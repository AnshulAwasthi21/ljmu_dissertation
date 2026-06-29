"""
tests/test_bm25.py

Phase 4 BM25 + metric tests. Same rigour as the 14 Phase-3 chunking tests:
tokenisation, index build, retrieval contract, traceability, exact-term ranking,
metric correctness on tiny known fixtures, and regression locks on the key
decisions (decimal-number tokenisation, tie-break determinism).

Run:  pytest tests/test_bm25.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.bm25_retriever import (
    BM25Retriever,
    tokenize,
    precision_at_k,
    recall_at_k,
    mrr_at_k,
    evaluate_retriever,
    metric_columns,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def tiny_corpus() -> pd.DataFrame:
    """Five chunks with deliberately controlled vocabulary so retrieval ranking
    is predictable. Mirrors the real schema (chunk_id, doc_id, source, subtype,
    chunk_index, text, char_length, metadata-as-string)."""
    recs = [
        ("d0_chunk_000", "d0", "edgar", "section_1", 0,
         "Dominion Energy completed the SCANA combination valued at 13.4 billion dollars."),
        ("d0_chunk_001", "d0", "edgar", "section_1", 1,
         "Renewable generation includes utility scale solar and offshore wind projects."),
        ("d1_chunk_000", "d1", "earnings", "qa", 0,
         "Operator the first question comes from the line of an analyst about revenue."),
        ("d1_chunk_001", "d1", "earnings", "qa", 1,
         "Operating margin increased 395 basis points over the prior second quarter."),
        ("d2_chunk_000", "d2", "edgar", "section_7", 0,
         "The unique token zzqxmarker appears only here for exact match testing."),
    ]
    rows = [
        {
            "chunk_id": c, "doc_id": d, "source": s, "subtype": st,
            "chunk_index": i, "text": t, "char_length": len(t),
            "metadata": '{"filename": "test.htm", "cik": "999"}',
        }
        for (c, d, s, st, i, t) in recs
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def retriever(tiny_corpus) -> BM25Retriever:
    return BM25Retriever(tiny_corpus)


# ----------------------------------------------------------------------
# Tokenisation
# ----------------------------------------------------------------------
def test_tokenize_lowercases():
    assert tokenize("Dominion ENERGY") == ["dominion", "energy"]


def test_tokenize_keeps_decimal_numbers_intact():
    # REGRESSION LOCK: "13.4" must stay one token, not split into "13" and "4".
    # This is the documented decision that lets BM25 match exact financial figures.
    assert "13.4" in tokenize("valued at 13.4 billion")
    assert "13" not in tokenize("valued at 13.4 billion")


def test_tokenize_splits_on_punctuation_and_drops_symbols():
    assert tokenize("revenue, margin; growth!") == ["revenue", "margin", "growth"]


def test_tokenize_stopword_removal_optional():
    assert tokenize("the revenue of the company", remove_stopwords=True) == ["revenue", "company"]
    # default keeps them
    assert "the" in tokenize("the revenue")


def test_tokenize_handles_non_string():
    assert tokenize(None) == []
    assert tokenize(123) == []


# ----------------------------------------------------------------------
# Index build + retrieval contract
# ----------------------------------------------------------------------
def test_index_builds_with_all_chunks(retriever, tiny_corpus):
    assert len(retriever) == len(tiny_corpus)
    assert retriever.chunk_ids == tiny_corpus["chunk_id"].tolist()


def test_duplicate_chunk_ids_rejected(tiny_corpus):
    dup = pd.concat([tiny_corpus, tiny_corpus.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        BM25Retriever(dup)


def test_missing_columns_rejected():
    with pytest.raises(KeyError):
        BM25Retriever(pd.DataFrame({"foo": ["bar"]}))


def test_retrieve_returns_k_results(retriever):
    assert len(retriever.retrieve("revenue analyst question", k=3)) == 3


def test_retrieve_caps_at_corpus_size(retriever):
    # ask for more than exist -> get all, not an error
    assert len(retriever.retrieve("energy", k=999)) == len(retriever)


def test_retrieve_k_zero_returns_empty(retriever):
    assert retriever.retrieve("energy", k=0) == []


def test_retrieve_results_descending_score(retriever):
    res = retriever.retrieve("solar wind renewable generation", k=5)
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)
    ranks = [r["rank"] for r in res]
    assert ranks == [1, 2, 3, 4, 5]


def test_retrieved_result_preserves_traceability(retriever):
    # Every result must carry chunk_id + metadata back to its filing/call.
    res = retriever.retrieve("solar wind", k=1)[0]
    for field in ("chunk_id", "doc_id", "source", "subtype", "metadata", "text", "score", "rank"):
        assert field in res
    assert res["source"] in {"edgar", "earnings"}


def test_exact_unique_term_ranks_its_chunk_first(retriever):
    # The chunk containing the unique token must be rank 1 for that token —
    # the lexical-precision property BM25 is meant to deliver.
    res = retriever.retrieve("zzqxmarker", k=1)
    assert res[0]["chunk_id"] == "d2_chunk_000"


def test_exact_decimal_figure_retrieves_its_chunk(retriever):
    # "13.4" as one token should surface the SCANA chunk.
    res = retriever.retrieve("13.4 billion", k=1)
    assert res[0]["chunk_id"] == "d0_chunk_000"


# ----------------------------------------------------------------------
# Metric correctness on known fixtures
# ----------------------------------------------------------------------
def test_precision_at_k_known():
    retrieved = ["a", "b", "c", "d"]
    assert precision_at_k(retrieved, {"b", "d"}, 4) == pytest.approx(0.5)
    assert precision_at_k(retrieved, {"a"}, 1) == pytest.approx(1.0)
    # single positive => P@4 capped at 1/4
    assert precision_at_k(retrieved, {"a"}, 4) == pytest.approx(0.25)


def test_recall_at_k_known():
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, {"b", "x"}, 4) == pytest.approx(0.5)  # found 1 of 2
    assert recall_at_k(retrieved, {"a"}, 1) == pytest.approx(1.0)
    assert recall_at_k(retrieved, {"z"}, 4) == pytest.approx(0.0)


def test_recall_empty_relevant_is_zero():
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_mrr_at_k_known():
    retrieved = ["x", "a", "b"]
    assert mrr_at_k(retrieved, {"a"}, 5) == pytest.approx(0.5)   # first rel at rank 2
    assert mrr_at_k(retrieved, {"x"}, 5) == pytest.approx(1.0)   # rank 1
    assert mrr_at_k(retrieved, {"z"}, 5) == pytest.approx(0.0)   # absent


def test_mrr_respects_k_cutoff():
    retrieved = ["x", "y", "a"]
    # relevant 'a' is at rank 3, so MRR@2 must be 0
    assert mrr_at_k(retrieved, {"a"}, 2) == 0.0
    assert mrr_at_k(retrieved, {"a"}, 3) == pytest.approx(1 / 3)


# ----------------------------------------------------------------------
# Full evaluation harness
# ----------------------------------------------------------------------
def test_evaluate_retriever_structure(retriever):
    golden = pd.DataFrame([
        {"query_id": "q000", "query_text": "13.4 billion SCANA combination",
         "source": "edgar", "subtype": "section_1",
         "relevant_chunk_ids": ["d0_chunk_000"]},
        {"query_id": "q001", "query_text": "operating margin basis points quarter",
         "source": "earnings", "subtype": "qa",
         "relevant_chunk_ids": ["d1_chunk_001"]},
    ])
    res = evaluate_retriever(retriever, golden, k_values=(1, 3, 5))
    assert len(res) == 2
    for col in ["query_id", "source", "subtype", "n_relevant"] + metric_columns((1, 3, 5)):
        assert col in res.columns
    # Both queries are lexically strong known-items -> should be found at rank 1.
    assert res["mrr@1"].mean() == pytest.approx(1.0)


def test_evaluate_breakdown_groupby_is_pandas3_safe(retriever):
    # Guards the notebook pattern: groupby().mean() on metric cols, NOT groupby.apply.
    golden = pd.DataFrame([
        {"query_id": "q000", "query_text": "13.4 billion SCANA",
         "source": "edgar", "subtype": "section_1", "relevant_chunk_ids": ["d0_chunk_000"]},
        {"query_id": "q001", "query_text": "operating margin basis points",
         "source": "earnings", "subtype": "qa", "relevant_chunk_ids": ["d1_chunk_001"]},
    ])
    per_q = evaluate_retriever(retriever, golden, k_values=(1, 3))
    cols = metric_columns((1, 3))
    by_source = per_q.groupby("source")[cols].mean()
    assert set(by_source.index) == {"edgar", "earnings"}
