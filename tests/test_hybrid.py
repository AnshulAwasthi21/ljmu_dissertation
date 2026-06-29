"""
tests/test_hybrid.py

Phase 6 — tests for the hybrid RRF retriever. Mirrors test_dense.py rigour:
  - RRF math correctness against HAND-COMPUTED toy rankings
  - union pool: a chunk in only one arm contributes 1/(k+rank) + 0
  - fuse-with-self ≈ that ranking (idempotent ordering)
  - deterministic tie-break by chunk_id
  - default k == 60
  - interface parity with BM25Retriever / DenseRetriever (real arms, fake encoder)
  - traceability (metadata + per-arm ranks preserved)
  - k edge cases (0, > corpus, > candidate_depth)
  - integration with the UNCHANGED evaluate_retriever harness
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "retrieval"
sys.path.insert(0, str(SRC))

from bm25_retriever import BM25Retriever, evaluate_retriever  # noqa: E402
from dense_retriever import DenseRetriever  # noqa: E402
from hybrid_retriever import (  # noqa: E402
    HybridRetriever,
    fuse_rankings,
    rrf_contribution,
    DEFAULT_RRF_K,
)


# ----------------------------------------------------------------------
# Toy fixtures: a 5-chunk corpus + a fake dense encoder we fully control
# ----------------------------------------------------------------------
@pytest.fixture
def toy_chunks():
    rows = [
        # text is engineered so BM25 rankings are predictable per query
        {"chunk_id": "edgar_0001_section_7_chunk_000",
         "text": "revenue increased due to higher product sales and gross margin",
         "source": "edgar", "subtype": "section_7", "metadata": '{"cik": "111"}'},
        {"chunk_id": "edgar_0001_section_7_chunk_001",
         "text": "operating expenses rose on research and development spending",
         "source": "edgar", "subtype": "section_7", "metadata": '{"cik": "111"}'},
        {"chunk_id": "edgar_0002_section_1A_chunk_000",
         "text": "risk factors could have a material adverse effect on our business",
         "source": "edgar", "subtype": "section_1A", "metadata": '{"cik": "222"}'},
        {"chunk_id": "earnings_0003_qa_chunk_000",
         "text": "the combined ratio outlook for the full year remains favorable",
         "source": "earnings", "subtype": "qa", "metadata": '{"ticker": "ABC"}'},
        {"chunk_id": "earnings_0003_qa_chunk_001",
         "text": "we expect the effective tax rate to be around twenty one percent",
         "source": "earnings", "subtype": "qa", "metadata": '{"ticker": "ABC"}'},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def toy_embeddings():
    # 5 chunks x 4 dims, L2-normalised; engineered so a chosen query vector has a
    # controllable nearest neighbour. Orthogonal-ish basis vectors keep it simple.
    raw = np.array(
        [[1.0, 0.0, 0.0, 0.0],   # chunk 0
         [0.0, 1.0, 0.0, 0.0],   # chunk 1
         [0.0, 0.0, 1.0, 0.0],   # chunk 2
         [0.0, 0.0, 0.0, 1.0],   # chunk 3
         [0.5, 0.0, 0.0, 0.5]],  # chunk 4 (between 0 and 3)
        dtype=np.float32,
    )
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


def make_dense(toy_chunks, toy_embeddings, query_vec):
    """DenseRetriever with a fake encoder that returns a fixed query vector,
    so dense rankings are deterministic and need no model download."""
    qv = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    qv = qv / np.linalg.norm(qv)

    def fake_encoder(texts):
        return np.repeat(qv, len(list(texts)), axis=0)

    return DenseRetriever(
        toy_chunks, embeddings=toy_embeddings, query_encoder=fake_encoder,
    )


# ======================================================================
# 1. Pure RRF math — hand-computed
# ======================================================================
def test_rrf_contribution_one_indexed():
    # rank 1 with k=60 -> 1/61; rank 100 -> 1/160
    assert rrf_contribution(1, k=60) == pytest.approx(1 / 61)
    assert rrf_contribution(100, k=60) == pytest.approx(1 / 160)
    assert rrf_contribution(None, k=60) == 0.0       # absent arm
    assert rrf_contribution(0, k=60) == 0.0          # guard against 0-index misuse


def test_default_k_is_60():
    assert DEFAULT_RRF_K == 60


def test_fuse_hand_computed_small_k():
    # Two arms, k=10 chosen so every fused score is distinct and easy to verify.
    arm1 = ["A", "B", "C", "D"]   # ranks 1,2,3,4
    arm2 = ["C", "A", "E"]        # ranks 1,2,3
    fused = fuse_rankings([arm1, arm2], k=10)
    score = {cid: s for cid, s, _ in fused}

    expected = {
        "A": 1 / 11 + 1 / 12,   # arm1 r1, arm2 r2  = 0.174242
        "C": 1 / 13 + 1 / 11,   # arm1 r3, arm2 r1  = 0.167832
        "B": 1 / 12,            # arm1 r2 only      = 0.083333
        "E": 1 / 13,            # arm2 r3 only      = 0.076923
        "D": 1 / 14,            # arm1 r4 only      = 0.071429
    }
    for cid, exp in expected.items():
        assert score[cid] == pytest.approx(exp)

    # full ranking order
    assert [cid for cid, _, _ in fused] == ["A", "C", "B", "E", "D"]

    # per-arm ranks reported correctly (union order: arm1 first, then arm2 extras)
    per = {cid: ranks for cid, _, ranks in fused}
    assert per["A"] == [1, 2]
    assert per["C"] == [3, 1]
    assert per["B"] == [2, None]
    assert per["E"] == [None, 3]
    assert per["D"] == [4, None]


def test_missing_arm_contributes_zero():
    # A appears only in arm1 at rank 1 -> exactly 1/(k+1), no penalty floor.
    fused = fuse_rankings([["A"], ["B"]], k=60)
    score = {cid: s for cid, s, _ in fused}
    assert score["A"] == pytest.approx(1 / 61)
    assert score["B"] == pytest.approx(1 / 61)


def test_fuse_with_self_preserves_order():
    # Fusing a ranking with an identical copy must preserve that ranking's order
    # (every chunk simply gets doubled, monotonic in rank -> order unchanged).
    L = ["X", "Y", "Z", "W"]
    fused = fuse_rankings([L, L], k=60)
    assert [cid for cid, _, _ in fused] == L
    # and each score is exactly 2/(k+rank)
    for rank, (cid, s, _) in enumerate(fused, start=1):
        assert s == pytest.approx(2 / (60 + rank))


def test_deterministic_tiebreak_by_chunk_id():
    # B and A tie (mirror ranks); chunk_id ascending breaks the tie -> A before B.
    fused = fuse_rankings([["A", "B"], ["B", "A"]], k=60)
    ids = [cid for cid, _, _ in fused]
    assert ids == ["A", "B"]
    s = {cid: sc for cid, sc, _ in fused}
    assert s["A"] == pytest.approx(s["B"])  # genuinely tied on score


# ======================================================================
# 2. Interface parity + plumbing (real BM25 + real dense, fake encoder)
# ======================================================================
def test_retrieve_payload_shape(toy_chunks, toy_embeddings):
    bm25 = BM25Retriever(toy_chunks)
    dense = make_dense(toy_chunks, toy_embeddings, [1, 0, 0, 0])
    hybrid = HybridRetriever(bm25, dense)
    res = hybrid.retrieve("revenue gross margin product sales", k=3)
    assert len(res) == 3
    r0 = res[0]
    for key in ("chunk_id", "text", "score", "rank",
                "source", "subtype", "metadata",
                "bm25_rank", "dense_rank", "bm25_score", "dense_score"):
        assert key in r0, f"missing key {key}"
    # ranks are sequential 1..k
    assert [r["rank"] for r in res] == [1, 2, 3]


def test_retrieve_ids_matches_retrieve(toy_chunks, toy_embeddings):
    bm25 = BM25Retriever(toy_chunks)
    dense = make_dense(toy_chunks, toy_embeddings, [0, 0, 0, 1])
    hybrid = HybridRetriever(bm25, dense)
    full = hybrid.retrieve("combined ratio outlook full year", k=4)
    ids = hybrid.retrieve_ids("combined ratio outlook full year", k=4)
    assert ids == [r["chunk_id"] for r in full]


def test_traceability_metadata_preserved(toy_chunks, toy_embeddings):
    bm25 = BM25Retriever(toy_chunks)
    dense = make_dense(toy_chunks, toy_embeddings, [1, 0, 0, 0])
    hybrid = HybridRetriever(bm25, dense)
    res = hybrid.retrieve("revenue", k=1)
    assert res[0]["metadata"] == '{"cik": "111"}'
    assert res[0]["source"] == "edgar"


def test_determinism_repeated_calls(toy_chunks, toy_embeddings):
    bm25 = BM25Retriever(toy_chunks)
    dense = make_dense(toy_chunks, toy_embeddings, [0.5, 0, 0, 0.5])
    hybrid = HybridRetriever(bm25, dense)
    a = hybrid.retrieve_ids("revenue tax rate risk", k=5)
    b = hybrid.retrieve_ids("revenue tax rate risk", k=5)
    assert a == b


# ======================================================================
# 3. Edge cases
# ======================================================================
def test_k_zero_returns_empty(toy_chunks, toy_embeddings):
    bm25 = BM25Retriever(toy_chunks)
    dense = make_dense(toy_chunks, toy_embeddings, [1, 0, 0, 0])
    hybrid = HybridRetriever(bm25, dense)
    assert hybrid.retrieve("revenue", k=0) == []
    assert hybrid.retrieve_ids("revenue", k=0) == []


def test_k_larger_than_corpus(toy_chunks, toy_embeddings):
    bm25 = BM25Retriever(toy_chunks)
    dense = make_dense(toy_chunks, toy_embeddings, [1, 0, 0, 0])
    hybrid = HybridRetriever(bm25, dense)
    res = hybrid.retrieve("revenue", k=999)
    # union of both arms' top-100 can be at most the 5-chunk corpus
    assert len(res) <= len(toy_chunks)
    assert len({r["chunk_id"] for r in res}) == len(res)  # no duplicates


def test_k_exceeds_candidate_depth_deepens_pull(toy_chunks, toy_embeddings):
    # candidate_depth deliberately tiny; asking for more must still work by
    # deepening the per-arm pull to k.
    bm25 = BM25Retriever(toy_chunks)
    dense = make_dense(toy_chunks, toy_embeddings, [1, 0, 0, 0])
    hybrid = HybridRetriever(bm25, dense, candidate_depth=1)
    res = hybrid.retrieve("revenue", k=4)
    assert len(res) == 4


def test_invalid_construction():
    class Dummy:
        def retrieve(self, q, k=10):
            return []
        def __len__(self):
            return 0
    with pytest.raises(ValueError):
        HybridRetriever(Dummy(), Dummy(), k=0)
    with pytest.raises(ValueError):
        HybridRetriever(Dummy(), Dummy(), candidate_depth=0)


# ======================================================================
# 4. Integration with the UNCHANGED Phase-4 metric harness
# ======================================================================
def test_integration_with_evaluate_retriever(toy_chunks, toy_embeddings):
    bm25 = BM25Retriever(toy_chunks)
    dense = make_dense(toy_chunks, toy_embeddings, [1, 0, 0, 0])
    hybrid = HybridRetriever(bm25, dense)

    golden = pd.DataFrame([
        {"query_id": "q000", "source": "edgar", "subtype": "section_7",
         "query_text": "revenue gross margin product sales",
         "relevant_chunk_ids": ["edgar_0001_section_7_chunk_000"]},
        {"query_id": "q001", "source": "earnings", "subtype": "qa",
         "query_text": "combined ratio outlook full year favorable",
         "relevant_chunk_ids": ["earnings_0003_qa_chunk_000"]},
    ])

    per_q = evaluate_retriever(hybrid, golden, k_values=(1, 3, 5))
    assert len(per_q) == 2
    for col in ("precision@1", "recall@1", "mrr@1",
                "precision@5", "recall@5", "mrr@5"):
        assert col in per_q.columns
        assert per_q[col].between(0.0, 1.0).all()
    # single-positive recall@k is 0/1; mrr in (0,1]; all finite
    assert per_q["recall@5"].isin([0.0, 1.0]).all()


def test_fuse_with_self_via_retriever_matches_single_arm(toy_chunks, toy_embeddings):
    # Composing BM25 with a clone of itself (as both arms) must reproduce BM25's
    # own ordering — the retriever-level analogue of test_fuse_with_self.
    bm25a = BM25Retriever(toy_chunks)
    bm25b = BM25Retriever(toy_chunks)
    hybrid = HybridRetriever(bm25a, bm25b)
    q = "operating expenses research development"
    assert hybrid.retrieve_ids(q, k=5) == bm25a.retrieve_ids(q, k=5)
