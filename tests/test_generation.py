"""
tests/test_generation.py

Phase 7 generation tests. Fully offline: a fake retriever and a fake
``complete_fn`` stand in for the real retrievers and OpenAI, so the suite
exercises prompt assembly, context formatting/ordering, provenance recording,
``gold_in_context`` logic, abstention detection, the matrix sweep, determinism,
save/load round-trip and content hashing WITHOUT touching the network. The real
API calls and the real golden-hash assertion live in 07_generation.ipynb.

Mirrors the rigour of test_hybrid.py: pure-function correctness, edge cases,
fixed-seed/fake-client determinism, and hash stability.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.generation.generator import (
    ABSTENTION_TEXT,
    GENERATION_COLUMNS,
    GENERATION_SYSTEM_PROMPT,
    PROMPT_VERSION,
    build_context_block,
    build_user_prompt,
    generate_one,
    gold_in_context,
    hash_generations,
    is_abstention,
    load_generations,
    make_openai_complete_fn,
    prompt_fingerprint,
    run_generation_matrix,
    save_generations,
)
from src.retrieval.golden_dataset_builder import hash_golden_set


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeRetriever:
    """Returns a fixed ranked list per query, in the BM25/dense/hybrid dict shape
    (chunk_id + text + score + rank). Deterministic, offline."""

    def __init__(self, ranking):
        # ranking: dict[query_text] -> list[(chunk_id, text)] best-first
        self._ranking = ranking

    def retrieve(self, query, k=10):
        hits = self._ranking.get(query, [])[:k]
        out = []
        for rank, (cid, text) in enumerate(hits, start=1):
            out.append({
                "chunk_id": cid,
                "text": text,
                "source": "edgar" if cid.startswith("edgar") else "earnings",
                "subtype": "section_7",
                "score": float(100 - rank),  # descending, deterministic
                "rank": rank,
            })
        return out

    def retrieve_ids(self, query, k=10):
        return [h["chunk_id"] for h in self.retrieve(query, k)]


def fake_complete_echo(system, user):
    """Deterministic fake LLM: echoes a marker plus a citation so provenance and
    parsing paths are exercised. Records model + fingerprint like the real one."""
    return {
        "text": "Synthesised answer grounded in the passages [1].",
        "model": "fake-gen",
        "system_fingerprint": "fp_test_0001",
    }


def fake_complete_abstain(system, user):
    return {"text": ABSTENTION_TEXT, "model": "fake-gen", "system_fingerprint": "fp_x"}


# ----------------------------------------------------------------------
# Prompt assembly
# ----------------------------------------------------------------------
def test_context_block_numbered_in_rank_order():
    hits = [
        {"chunk_id": "edgar_0001_section_7_chunk_000", "text": "Alpha revenue rose."},
        {"chunk_id": "edgar_0001_section_7_chunk_001", "text": "Beta margin fell."},
    ]
    block = build_context_block(hits)
    assert block.index("[1]") < block.index("[2]")
    assert "Alpha revenue rose." in block
    assert "Beta margin fell." in block


def test_context_block_does_not_leak_chunk_ids():
    # chunk ids encode source/section and must NOT reach the model.
    hits = [{"chunk_id": "edgar_0004_section_7_chunk_020", "text": "Some content."}]
    block = build_context_block(hits)
    assert "edgar_0004_section_7_chunk_020" not in block
    assert "section_7" not in block


def test_context_block_handles_missing_text():
    hits = [{"chunk_id": "x", "text": None}, {"chunk_id": "y"}]
    block = build_context_block(hits)
    assert "[1]" in block and "[2]" in block  # no crash on None / missing


def test_user_prompt_contains_query_and_context():
    up = build_user_prompt("What drove revenue?", "[1]\nRevenue text")
    assert "What drove revenue?" in up
    assert "[1]" in up and "Revenue text" in up
    assert "QUESTION:" in up and "CONTEXT PASSAGES:" in up


def test_system_prompt_carries_key_instructions():
    # the control must instruct abstention and citation, and embed the sentinel
    assert ABSTENTION_TEXT in GENERATION_SYSTEM_PROMPT
    assert "square brackets" in GENERATION_SYSTEM_PROMPT
    assert "ONLY" in GENERATION_SYSTEM_PROMPT


def test_prompt_fingerprint_is_stable_hex():
    fp = prompt_fingerprint()
    assert isinstance(fp, str) and len(fp) == 16
    assert fp == prompt_fingerprint()  # deterministic


# ----------------------------------------------------------------------
# gold_in_context / abstention
# ----------------------------------------------------------------------
def test_gold_in_context_present_returns_rank():
    present, rank = gold_in_context(["a", "b", "gold", "c"], ["gold"])
    assert present is True and rank == 3


def test_gold_in_context_absent_returns_none():
    present, rank = gold_in_context(["a", "b", "c"], ["gold"])
    assert present is False and rank is None


def test_gold_in_context_first_match_wins_multipositive():
    present, rank = gold_in_context(["a", "g2", "g1"], ["g1", "g2"])
    assert present is True and rank == 2  # first relevant encountered


def test_is_abstention_exact_and_tolerant():
    assert is_abstention(ABSTENTION_TEXT)
    assert is_abstention(f'  "{ABSTENTION_TEXT}"  ')   # quote/space tolerant
    assert not is_abstention("Revenue rose 12% [1].")
    assert not is_abstention(None)


# ----------------------------------------------------------------------
# generate_one: provenance correctness
# ----------------------------------------------------------------------
def _ranking_with_gold_at(rank_pos):
    ids = [f"edgar_0001_section_7_chunk_{i:03d}" for i in range(5)]
    gold = "edgar_0009_section_7_chunk_099"
    chunks = [(cid, f"text {cid}") for cid in ids]
    chunks.insert(rank_pos - 1, (gold, "the gold passage text"))
    return {"q": chunks[:5]}, gold


def test_generate_one_records_exact_provenance():
    ranking, gold = _ranking_with_gold_at(rank_pos=3)
    retr = FakeRetriever(ranking)
    rec = generate_one(
        query_text="q", relevant_ids=[gold], retriever=retr,
        retriever_name="bm25", complete_fn=fake_complete_echo,
        query_id="q000", source="edgar", subtype="section_7", k=5,
    )
    assert rec["retriever"] == "bm25"
    assert rec["retrieved_chunk_ids"] == retr.retrieve_ids("q", 5)  # rank order
    assert rec["gold_in_context"] is True
    assert rec["gold_rank"] == 3
    assert rec["n_context"] == 5
    assert rec["answer"].startswith("Synthesised answer")
    assert rec["abstained"] is False
    assert rec["system_fingerprint"] == "fp_test_0001"
    assert rec["prompt_version"] == PROMPT_VERSION
    assert len(rec["retrieved_scores"]) == 5


def test_generate_one_gold_absent_and_abstains():
    ids = [f"edgar_0001_section_7_chunk_{i:03d}" for i in range(5)]
    retr = FakeRetriever({"q": [(c, f"t{c}") for c in ids]})
    rec = generate_one(
        query_text="q", relevant_ids=["edgar_9999_section_7_chunk_000"],
        retriever=retr, retriever_name="dense", complete_fn=fake_complete_abstain, k=5,
    )
    assert rec["gold_in_context"] is False
    assert rec["gold_rank"] is None
    assert rec["abstained"] is True


def test_generate_one_respects_k():
    ids = [f"edgar_0001_section_7_chunk_{i:03d}" for i in range(10)]
    retr = FakeRetriever({"q": [(c, f"t{c}") for c in ids]})
    rec = generate_one(
        query_text="q", relevant_ids=[ids[0]], retriever=retr,
        retriever_name="bm25", complete_fn=fake_complete_echo, k=3,
    )
    assert rec["n_context"] == 3
    assert len(rec["retrieved_chunk_ids"]) == 3


# ----------------------------------------------------------------------
# run_generation_matrix
# ----------------------------------------------------------------------
def _mini_golden():
    return pd.DataFrame([
        {"query_id": "q000", "query_text": "qa", "source": "edgar",
         "subtype": "section_7", "relevant_chunk_ids": ["edgar_a_chunk_000"]},
        {"query_id": "q001", "query_text": "qb", "source": "earnings",
         "subtype": "qa", "relevant_chunk_ids": ["earnings_b_chunk_000"]},
    ])


def _mini_retrievers():
    r1 = FakeRetriever({
        "qa": [("edgar_a_chunk_000", "gold a"), ("edgar_a_chunk_001", "x")],
        "qb": [("earnings_b_chunk_000", "gold b"), ("earnings_b_chunk_001", "y")],
    })
    r2 = FakeRetriever({
        "qa": [("edgar_a_chunk_002", "z"), ("edgar_a_chunk_000", "gold a")],
        "qb": [("earnings_b_chunk_009", "w"), ("earnings_b_chunk_008", "v")],
    })
    return {"bm25": r1, "dense": r2, "hybrid": r1}


def test_matrix_shape_and_columns():
    df = run_generation_matrix(_mini_golden(), _mini_retrievers(),
                               fake_complete_echo, k=2)
    assert len(df) == 2 * 3  # queries x retrievers
    assert list(df.columns) == GENERATION_COLUMNS
    assert set(df["retriever"]) == {"bm25", "dense", "hybrid"}
    # gold landed at rank 1 for bm25/hybrid on both queries, rank 2 for dense-qa
    bm25_qa = df[(df.retriever == "bm25") & (df.query_id == "q000")].iloc[0]
    assert bm25_qa["gold_rank"] == 1
    dense_qa = df[(df.retriever == "dense") & (df.query_id == "q000")].iloc[0]
    assert dense_qa["gold_rank"] == 2
    dense_qb = df[(df.retriever == "dense") & (df.query_id == "q001")].iloc[0]
    assert not dense_qb["gold_in_context"]  # gold not in dense's top-2 (numpy bool)


def test_matrix_is_deterministic_under_fake_client():
    g, r = _mini_golden(), _mini_retrievers()
    h1 = hash_generations(run_generation_matrix(g, r, fake_complete_echo, k=2))
    h2 = hash_generations(run_generation_matrix(g, r, fake_complete_echo, k=2))
    assert h1 == h2


# ----------------------------------------------------------------------
# hashing + save/load
# ----------------------------------------------------------------------
def test_hash_is_order_independent():
    df = run_generation_matrix(_mini_golden(), _mini_retrievers(),
                               fake_complete_echo, k=2)
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert hash_generations(df) == hash_generations(shuffled)


def test_save_load_roundtrip_preserves_lists_and_hash(tmp_path):
    df = run_generation_matrix(_mini_golden(), _mini_retrievers(),
                               fake_complete_echo, k=2)
    p = tmp_path / "gens.parquet"
    h_saved = save_generations(df, p)
    reloaded = load_generations(p)
    assert isinstance(reloaded["retrieved_chunk_ids"].iloc[0], list)
    assert isinstance(reloaded["relevant_chunk_ids"].iloc[0], list)
    assert hash_generations(reloaded) == h_saved  # frozen artifact stable


# ----------------------------------------------------------------------
# OpenAI factory wiring (with a fake client object, still offline)
# ----------------------------------------------------------------------
class _FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.model = "gpt-4o-mini-fake"
        self.system_fingerprint = "fp_real_like"


class _FakeCompletions:
    def __init__(self, recorder):
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder.update(kwargs)
        return _FakeResp("hello from fake [1]")


class _FakeClient:
    def __init__(self, recorder):
        self.chat = type("C", (), {"completions": _FakeCompletions(recorder)})


def test_openai_complete_fn_passes_config_and_returns_metadata():
    recorder = {}
    client = _FakeClient(recorder)
    fn = make_openai_complete_fn(client, model="gpt-4o-mini", temperature=0.0, seed=42)
    out = fn("SYS", "USER")
    assert out["text"] == "hello from fake [1]"
    assert out["system_fingerprint"] == "fp_real_like"
    # fixed decoding config was forwarded to the API call
    assert recorder["model"] == "gpt-4o-mini"
    assert recorder["temperature"] == 0.0
    assert recorder["seed"] == 42
    assert recorder["messages"][0]["role"] == "system"
    assert recorder["messages"][1]["role"] == "user"


# ----------------------------------------------------------------------
# Golden-set hash reuse (proves the Phase-4 contract function is intact here)
# ----------------------------------------------------------------------
def test_golden_hash_is_order_independent():
    g = pd.DataFrame([
        {"query_id": "q001", "query_text": "a", "relevant_chunk_ids": ["c2", "c1"]},
        {"query_id": "q000", "query_text": "b", "relevant_chunk_ids": ["c3"]},
    ])
    g2 = g.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert hash_golden_set(g) == hash_golden_set(g2)
