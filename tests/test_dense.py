"""
tests/test_dense.py — Phase 5 dense-retriever tests.

Mirrors the rigour of test_bm25.py. All tests are MODEL-FREE: they use synthetic
embeddings and a fake encoder, so the suite runs offline with no model download
and no GPU. Model-dependent behaviour (the actual BGE encode) is verified once in
the notebook, not here, to keep tests fast and deterministic.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "src/retrieval")

import dense_retriever as dr
from bm25_retriever import evaluate_retriever


# ----------------------------------------------------------------------
# Fixtures: a tiny corpus with one-hot embeddings so retrieval is exact
# and hand-checkable. chunk i has embedding e_i (the i-th basis vector).
# ----------------------------------------------------------------------
DIM = 6


def _corpus_df():
    rows = []
    for i in range(DIM):
        rows.append({
            "chunk_id": f"edgar_0000_section_1_chunk_{i:03d}",
            "doc_id": "edgar_0000_section_1",
            "source": "edgar" if i < 3 else "earnings",
            "subtype": "section_1" if i < 3 else "qa",
            "chunk_index": i,
            "chunk_size_cfg": 1500,
            "overlap_cfg": 200,
            "text": f"chunk text number {i}",
            "metadata": "{}",
        })
    return pd.DataFrame(rows)


def _onehot_embeddings():
    return np.eye(DIM, dtype=np.float32)


def _onehot_query_encoder(target_idx_lookup):
    """Return an encoder mapping a query string to the basis vector of the chunk
    index encoded in the query (e.g. 'want:2' -> e_2)."""
    def enc(texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for r, t in enumerate(texts):
            idx = target_idx_lookup(t)
            out[r, idx] = 1.0
        return out
    return enc


def _build(query_encoder):
    return dr.DenseRetriever(
        _corpus_df(), embeddings=_onehot_embeddings(),
        query_encoder=query_encoder,
    )


# ----------------------------------------------------------------------
# Construction guards
# ----------------------------------------------------------------------
def test_rejects_duplicate_ids():
    df = _corpus_df()
    df.loc[1, "chunk_id"] = df.loc[0, "chunk_id"]
    with pytest.raises(ValueError):
        dr.DenseRetriever(df, _onehot_embeddings(), lambda t: np.zeros((len(t), DIM)))


def test_rejects_row_mismatch():
    with pytest.raises(ValueError):
        dr.DenseRetriever(_corpus_df(), np.eye(DIM + 1, dtype=np.float32),
                          lambda t: np.zeros((len(t), DIM + 1)))


def test_rejects_missing_columns():
    df = _corpus_df().drop(columns=["text"])
    with pytest.raises(KeyError):
        dr.DenseRetriever(df, _onehot_embeddings(), lambda t: np.zeros((len(t), DIM)))


# ----------------------------------------------------------------------
# Core retrieval behaviour
# ----------------------------------------------------------------------
def test_retrieve_returns_target_first():
    enc = _onehot_query_encoder(lambda t: int(t.split(":")[1]))
    r = _build(enc)
    res = r.retrieve("want:4", k=3)
    assert res[0]["chunk_id"] == "edgar_0000_section_1_chunk_004"
    assert res[0]["rank"] == 1
    assert res[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_results_ranked_desc_by_score():
    # query equally close to 0 and 1, then identity dominates: build a query that
    # is a normalised mix so order is determined by similarity.
    def enc(texts):
        v = np.zeros((1, DIM), dtype=np.float32)
        v[0, 2] = 0.9
        v[0, 5] = 0.4
        return v
    r = _build(enc)
    res = r.retrieve("mix", k=DIM)
    scores = [x["score"] for x in res]
    assert scores == sorted(scores, reverse=True)
    assert res[0]["chunk_id"].endswith("_chunk_002")  # weight 0.9
    assert res[1]["chunk_id"].endswith("_chunk_005")  # weight 0.4
    assert [x["rank"] for x in res] == list(range(1, DIM + 1))


def test_retrieve_ids_matches_retrieve():
    enc = _onehot_query_encoder(lambda t: int(t.split(":")[1]))
    r = _build(enc)
    full = [x["chunk_id"] for x in r.retrieve("want:1", k=4)]
    ids = r.retrieve_ids("want:1", k=4)
    assert ids == full


def test_payload_preserves_metadata_and_traceability():
    enc = _onehot_query_encoder(lambda t: 0)
    r = _build(enc)
    top = r.retrieve("want:0", k=1)[0]
    for col in ("chunk_id", "doc_id", "source", "subtype", "metadata", "text"):
        assert col in top
    assert "score" in top and "rank" in top


def test_k_larger_than_corpus_returns_corpus_size():
    enc = _onehot_query_encoder(lambda t: 0)
    r = _build(enc)
    assert len(r.retrieve("want:0", k=999)) == DIM


def test_k_zero_or_negative_returns_empty():
    enc = _onehot_query_encoder(lambda t: 0)
    r = _build(enc)
    assert r.retrieve("want:0", k=0) == []
    assert r.retrieve("want:0", k=-3) == []


def test_determinism_repeated_calls():
    enc = _onehot_query_encoder(lambda t: 3)
    r = _build(enc)
    a = r.retrieve_ids("want:3", k=DIM)
    b = r.retrieve_ids("want:3", k=DIM)
    assert a == b


# ----------------------------------------------------------------------
# Backend equivalence: FAISS path and NumPy fallback must agree exactly
# ----------------------------------------------------------------------
def test_faiss_and_numpy_backends_agree():
    enc = _onehot_query_encoder(lambda t: int(t.split(":")[1]))
    r = _build(enc)
    primary = r.retrieve_ids("want:2", k=DIM)
    # Force the NumPy fallback path and compare.
    r.backend = "numpy"
    r._index = None
    fallback = r.retrieve_ids("want:2", k=DIM)
    assert primary == fallback


# ----------------------------------------------------------------------
# Embedding helpers / caching / prefix correctness
# ----------------------------------------------------------------------
def test_l2_normalize_unit_norm():
    m = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = dr._l2_normalize(m)
    assert out[0] == pytest.approx([0.6, 0.8], abs=1e-6)
    assert np.all(np.isfinite(out))  # zero row handled, no nan


def test_fingerprint_order_independent():
    a = dr.chunk_ids_fingerprint(["c", "a", "b"])
    b = dr.chunk_ids_fingerprint(["a", "b", "c"])
    assert a == b


class _FakeModel:
    """Records every text it is asked to encode; returns deterministic vectors."""
    def __init__(self, dim=DIM):
        self.dim = dim
        self.seen = []

    def encode(self, texts, **kw):
        self.seen.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, hash(t) % self.dim] = 1.0
        return out


def test_query_prefix_applied_to_query_not_passages():
    model = _FakeModel()
    df = _corpus_df()
    r = dr.DenseRetriever.from_model(
        df, model=model, use_cache=False,
        query_prefix="QP: ", passage_prefix="",
    )
    # Passages encoded so far: none should carry the query prefix.
    passage_batches = [b for b in model.seen]
    assert all(not any(s.startswith("QP: ") for s in batch) for batch in passage_batches)
    # Now a query — it MUST carry the prefix.
    r.retrieve("a question", k=1)
    assert any(any(s.startswith("QP: ") for s in batch) for batch in model.seen)


def test_embed_corpus_cache_roundtrip(tmp_path):
    model = _FakeModel()
    df = _corpus_df()
    emb1 = dr.embed_corpus(df, model=model, model_name="fake/model",
                           cache_dir=str(tmp_path), use_cache=True)
    calls_after_first = len(model.seen)
    # Second call must hit cache: no new encode calls, identical array.
    emb2 = dr.embed_corpus(df, model=model, model_name="fake/model",
                           cache_dir=str(tmp_path), use_cache=True)
    assert len(model.seen) == calls_after_first  # no re-embedding
    assert np.array_equal(emb1, emb2)


def test_cache_invalidates_on_corpus_change(tmp_path):
    model = _FakeModel()
    df = _corpus_df()
    dr.embed_corpus(df, model=model, model_name="fake/model",
                    cache_dir=str(tmp_path), use_cache=True)
    n1 = len(model.seen)
    df2 = df.copy()
    df2.loc[0, "chunk_id"] = "edgar_9999_section_1_chunk_000"  # changes fingerprint
    dr.embed_corpus(df2, model=model, model_name="fake/model",
                    cache_dir=str(tmp_path), use_cache=True)
    assert len(model.seen) > n1  # re-embedded because corpus changed


# ----------------------------------------------------------------------
# Integration: the UNCHANGED Phase-4 harness drives the dense retriever
# ----------------------------------------------------------------------
def test_evaluate_retriever_integration():
    enc = _onehot_query_encoder(lambda t: int(t.split(":")[1]))
    r = _build(enc)
    golden = pd.DataFrame([
        {"query_id": "q000", "query_text": "want:0", "source": "edgar",
         "subtype": "section_1",
         "relevant_chunk_ids": ["edgar_0000_section_1_chunk_000"]},
        {"query_id": "q001", "query_text": "want:5", "source": "earnings",
         "subtype": "qa",
         "relevant_chunk_ids": ["edgar_0000_section_1_chunk_005"]},
    ])
    per_q = evaluate_retriever(r, golden, k_values=(1, 3, 5))
    # Perfect retrieval on this toy set: every relevant chunk ranked first.
    assert per_q["recall@1"].mean() == pytest.approx(1.0)
    assert per_q["mrr@1"].mean() == pytest.approx(1.0)
    assert set(per_q.columns) >= {"query_id", "source", "subtype", "n_relevant",
                                  "recall@1", "mrr@1", "precision@1"}
