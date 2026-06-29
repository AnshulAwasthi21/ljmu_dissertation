"""
tests/test_golden_dataset.py

Phase 4 golden-set builder tests. The LLM call is exercised with an INJECTED
fake client so the suite runs offline and deterministically.

Run:  pytest tests/test_golden_dataset.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.golden_dataset_builder import (
    sample_seed_chunks,
    find_overlap_siblings,
    build_candidate_set,
    load_curated_golden_set,
    save_candidate_set,
    save_golden_set,
    load_golden_set,
    hash_golden_set,
    _allocate_proportional,
    _parse_id_list,
    GENERATION_METHOD,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def corpus() -> pd.DataFrame:
    rows = []
    # EDGAR: two docs, several subtypes
    for di, (doc, sub) in enumerate([("e0", "section_1"), ("e1", "section_1A"),
                                     ("e2", "section_7"), ("e3", "section_8")]):
        for ci in range(4):
            rows.append({
                "chunk_id": f"{doc}_chunk_{ci:03d}", "doc_id": doc, "source": "edgar",
                "subtype": sub, "chunk_index": ci,
                "text": f"Edgar {sub} chunk {ci} " + "content " * 80,
                "char_length": 600,
            })
    # Earnings
    for doc, sub in [("a0", "qa"), ("a1", "prepared_remarks"), ("a2", "full")]:
        for ci in range(4):
            rows.append({
                "chunk_id": f"{doc}_chunk_{ci:03d}", "doc_id": doc, "source": "earnings",
                "subtype": sub, "chunk_index": ci,
                "text": f"Earnings {sub} turn {ci} " + "dialogue " * 80,
                "char_length": 600,
            })
    return pd.DataFrame(rows)


class FakeChoice:
    def __init__(self, content): self.message = type("M", (), {"content": content})()


class FakeResp:
    def __init__(self, content): self.choices = [FakeChoice(content)]


class FakeClient:
    """Mimics client.chat.completions.create -> returns a JSON array of queries."""
    def __init__(self, payload='["What does this passage report?"]'):
        self.payload = payload
        self.calls = 0
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls += 1
        return FakeResp(self.payload)


# ----------------------------------------------------------------------
# Allocation + sampling
# ----------------------------------------------------------------------
def test_allocate_proportional_sums_to_n():
    alloc = _allocate_proportional({"a": 100, "b": 50, "c": 10}, 16)
    assert sum(alloc.values()) == 16


def test_allocate_never_exceeds_available():
    alloc = _allocate_proportional({"a": 2, "b": 2}, 10)
    assert alloc["a"] <= 2 and alloc["b"] <= 2


def test_sample_seed_chunks_counts_and_balance(corpus):
    s = sample_seed_chunks(corpus, n_edgar=6, n_earnings=6, min_chars=100)
    assert (s["source"] == "edgar").sum() == 6
    assert (s["source"] == "earnings").sum() == 6


def test_sample_seed_chunks_is_deterministic(corpus):
    a = sample_seed_chunks(corpus, n_edgar=5, n_earnings=5, min_chars=100, seed=42)
    b = sample_seed_chunks(corpus, n_edgar=5, n_earnings=5, min_chars=100, seed=42)
    assert a["chunk_id"].tolist() == b["chunk_id"].tolist()


def test_sample_filters_short_chunks(corpus):
    corpus.loc[corpus["chunk_id"] == "e0_chunk_000", "char_length"] = 50
    s = sample_seed_chunks(corpus, n_edgar=16, n_earnings=0, min_chars=400)
    assert "e0_chunk_000" not in s["chunk_id"].tolist()


# ----------------------------------------------------------------------
# Sibling detection
# ----------------------------------------------------------------------
def test_overlap_siblings_same_doc_adjacent(corpus):
    sibs = find_overlap_siblings("e0_chunk_001", corpus, window=1)
    assert set(sibs) == {"e0_chunk_000", "e0_chunk_002"}


def test_overlap_siblings_excludes_other_docs(corpus):
    sibs = find_overlap_siblings("e0_chunk_000", corpus, window=1)
    assert all(s.startswith("e0_") for s in sibs)
    assert "e1_chunk_000" not in sibs


# ----------------------------------------------------------------------
# Candidate assembly with fake LLM
# ----------------------------------------------------------------------
def test_build_candidate_set_shape(corpus):
    seeds = sample_seed_chunks(corpus, n_edgar=3, n_earnings=3, min_chars=100)
    fake = FakeClient()
    cand = build_candidate_set(seeds, corpus, client=fake, n_questions=1)
    assert len(cand) == 6
    assert fake.calls == 6
    # seed chunk pre-filled as its own relevant id; curated text starts blank
    assert (cand["relevant_chunk_ids"] == cand["seed_chunk_id"]).all()
    assert (cand["query_text_curated"] == "").all()
    assert (cand["keep"] == 1).all()


def test_parse_id_list_pipe_and_list():
    assert _parse_id_list("a|b|c") == ["a", "b", "c"]
    assert _parse_id_list(["a", "b"]) == ["a", "b"]
    assert _parse_id_list("") == []


# ----------------------------------------------------------------------
# Curated-set validation
# ----------------------------------------------------------------------
def _curated_frame():
    return pd.DataFrame([
        {"query_id": "q000", "keep": 1, "query_text_raw": "raw a",
         "query_text_curated": "How large was the merger?", "source": "edgar",
         "subtype": "section_1", "seed_chunk_id": "e0_chunk_000",
         "relevant_chunk_ids": "e0_chunk_000|e0_chunk_001",
         "candidate_siblings": "e0_chunk_001", "notes": ""},
        {"query_id": "q001", "keep": 0, "query_text_raw": "raw b",
         "query_text_curated": "dropped", "source": "edgar", "subtype": "section_1",
         "seed_chunk_id": "e1_chunk_000", "relevant_chunk_ids": "e1_chunk_000",
         "candidate_siblings": "", "notes": "weak"},
    ])


def test_load_curated_drops_keep_zero(tmp_path, corpus):
    p = tmp_path / "cand.csv"
    save_candidate_set(_curated_frame(), p)
    g = load_curated_golden_set(p, corpus)
    assert g["query_id"].tolist() == ["q000"]          # q001 dropped
    assert g.loc[0, "relevant_chunk_ids"] == ["e0_chunk_000", "e0_chunk_001"]
    assert g.loc[0, "generation_method"] == GENERATION_METHOD


def test_load_curated_rejects_missing_chunk(tmp_path, corpus):
    bad = _curated_frame().iloc[[0]].copy()
    bad.loc[0, "relevant_chunk_ids"] = "e0_chunk_000|does_not_exist"
    p = tmp_path / "bad.csv"
    save_candidate_set(bad, p)
    with pytest.raises(ValueError, match="not in corpus"):
        load_curated_golden_set(p, corpus)


def test_load_curated_rejects_empty_paraphrase(tmp_path, corpus):
    bad = _curated_frame().iloc[[0]].copy()
    bad.loc[0, "query_text_curated"] = ""
    p = tmp_path / "bad2.csv"
    save_candidate_set(bad, p)
    with pytest.raises(ValueError, match="paraphrase required"):
        load_curated_golden_set(p, corpus)


# ----------------------------------------------------------------------
# Persist + hash
# ----------------------------------------------------------------------
def test_golden_set_roundtrips_parquet(tmp_path, corpus):
    p = tmp_path / "cand.csv"
    save_candidate_set(_curated_frame(), p)
    g = load_curated_golden_set(p, corpus)
    out = tmp_path / "golden.parquet"
    h = save_golden_set(g, out)
    reloaded = load_golden_set(out)
    assert reloaded.loc[0, "relevant_chunk_ids"] == ["e0_chunk_000", "e0_chunk_001"]
    assert hash_golden_set(reloaded) == h


def test_hash_is_order_independent(tmp_path, corpus):
    p = tmp_path / "cand.csv"
    save_candidate_set(_curated_frame(), p)
    g = load_curated_golden_set(p, corpus)
    h1 = hash_golden_set(g)
    h2 = hash_golden_set(g.iloc[::-1].reset_index(drop=True))
    assert h1 == h2


def test_hash_changes_when_relevance_changes(tmp_path, corpus):
    p = tmp_path / "cand.csv"
    save_candidate_set(_curated_frame(), p)
    g = load_curated_golden_set(p, corpus)
    h1 = hash_golden_set(g)
    g2 = g.copy()
    g2.at[0, "relevant_chunk_ids"] = ["e0_chunk_000"]   # drop a positive
    assert hash_golden_set(g2) != h1
