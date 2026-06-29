"""
src/retrieval/bm25_retriever.py

Phase 4 — BM25 sparse-retrieval baseline.

Indexes the `text` column of chunks_n200.parquet (the shared retrieval control
from Phase 3). Every retrieved result preserves chunk_id + inherited metadata so
it is traceable back to its filing / call / section / sector — required for the
formal-vs-conversational and sector analyses in Phase 8.

Comparative-validity contract
------------------------------
Dense (Phase 5) and hybrid (Phase 6) MUST index this exact chunk file and be
scored on the exact frozen golden set. This module therefore keeps indexing and
metrics separable and stateless w.r.t. the retriever (metric functions operate on
ranked id lists), so the same evaluation harness drives all three configs.

Tokenisation decision (locked by a regression test)
----------------------------------------------------
Lowercase; tokens are decimal numbers OR alphabetic runs:  \\d+(?:\\.\\d+)?|[a-z]+
Numbers with decimals are kept INTACT ("13.4" stays one token, not "13"+"4")
because exact numerical facts are precisely where sparse retrieval is expected to
beat dense (proposal §7.4). No stemming, no stopword removal by default: IDF
already down-weights common terms, and adding a stemmer would introduce an
uncontrolled variable into the cross-retriever comparison. Stopword removal is
available as an opt-in flag, off by default, documented as such.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set

import pandas as pd
from rank_bm25 import BM25Okapi

# Decimal number OR alphabetic run. Order matters: numbers first so "13.4" is
# consumed whole before the alphabetic alternative can fire.
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[a-z]+")

# Minimal English stopword list (opt-in only). Kept short and explicit rather
# than pulling nltk, to avoid an extra dependency and keep the decision visible.
_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "by", "with", "as", "at", "that", "this", "it",
    "from", "we", "our", "their", "its", "has", "have", "had", "will", "would",
}


def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """Lowercase + regex tokenise. Decimal numbers stay intact. See module docstring."""
    if not isinstance(text, str):
        return []
    toks = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


class BM25Retriever:
    """BM25Okapi index over a chunk DataFrame, returning fully-traceable results.

    Parameters
    ----------
    chunks_df : DataFrame with at least `id_col` and `text_col`. Any extra
        columns (source, subtype, metadata, ...) are preserved and returned.
    """

    def __init__(
        self,
        chunks_df: pd.DataFrame,
        text_col: str = "text",
        id_col: str = "chunk_id",
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        remove_stopwords: bool = False,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if text_col not in chunks_df.columns or id_col not in chunks_df.columns:
            raise KeyError(f"chunks_df must contain '{id_col}' and '{text_col}'")
        if chunks_df[id_col].duplicated().any():
            raise ValueError(f"{id_col} must be unique to keep results traceable")

        self.text_col = text_col
        self.id_col = id_col
        self.remove_stopwords = remove_stopwords
        self.k1 = k1
        self.b = b
        self._tok = tokenizer or (lambda t: tokenize(t, remove_stopwords))

        # Frozen, position-aligned arrays: row i in every array is the same chunk.
        self._df = chunks_df.reset_index(drop=True).copy()
        self.chunk_ids: List[str] = self._df[id_col].astype(str).tolist()
        self._meta_cols = [c for c in self._df.columns if c != text_col]

        self._corpus_tokens: List[List[str]] = [
            self._tok(t) for t in self._df[text_col].fillna("").astype(str).tolist()
        ]
        self.bm25 = BM25Okapi(self._corpus_tokens, k1=k1, b=b)

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def _row_payload(self, i: int, score: float, rank: int) -> Dict:
        row = self._df.iloc[i]
        payload = {c: row[c] for c in self._meta_cols}
        payload["text"] = row[self.text_col]
        payload["score"] = float(score)
        payload["rank"] = rank
        return payload

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """Return the top-`k` chunks as dicts with full metadata + score + rank.

        Results are ordered by descending BM25 score; ties broken by corpus
        position for determinism. Returns up to k (fewer if corpus < k).
        """
        if k <= 0 or len(self) == 0:
            return []
        q_tokens = self._tok(query)
        scores = self.bm25.get_scores(q_tokens)
        # argsort desc, stable on position for tie determinism
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        top = order[:k]
        return [self._row_payload(i, scores[i], rank + 1) for rank, i in enumerate(top)]

    def retrieve_ids(self, query: str, k: int = 10) -> List[str]:
        """Lightweight path: top-`k` chunk_ids only (used by the metric loop)."""
        return [r[self.id_col] for r in self.retrieve(query, k)]


# ----------------------------------------------------------------------
# Metric functions — operate on ranked id lists, retriever-agnostic
# ----------------------------------------------------------------------
def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """|relevant ∩ top-k| / k.

    NOTE: with single-positive judgements Precision@K is capped at 1/k, so a
    perfect retriever scores 1/k. Interpret alongside Recall/MRR (see
    phase_04_results.md). Multi-positive queries make it informative.
    """
    if k <= 0:
        return 0.0
    rel = set(relevant_ids)
    topk = list(retrieved_ids)[:k]
    hits = sum(1 for c in topk if c in rel)
    return hits / k


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """|relevant ∩ top-k| / |relevant|. With one positive this equals Hit Rate / Success@k."""
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    topk = list(retrieved_ids)[:k]
    hits = sum(1 for c in topk if c in rel)
    return hits / len(rel)


def mrr_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Reciprocal rank of the FIRST relevant chunk within top-k; 0 if none."""
    rel = set(relevant_ids)
    for i, c in enumerate(list(retrieved_ids)[:k]):
        if c in rel:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retriever(
    retriever: BM25Retriever,
    golden_df: pd.DataFrame,
    k_values: Sequence[int] = (1, 3, 5, 10),
    query_col: str = "query_text",
    relevant_col: str = "relevant_chunk_ids",
) -> pd.DataFrame:
    """Score every golden query, returning a per-query DataFrame.

    Columns: query_id, source, subtype, n_relevant, then precision@k / recall@k /
    mrr@k for each k. Aggregate / breakdown is left to the caller (use stable
    groupby().mean() — no groupby.apply, which is removed in pandas 3).
    """
    max_k = max(k_values)
    rows = []
    for _, q in golden_df.iterrows():
        retrieved = retriever.retrieve_ids(q[query_col], k=max_k)
        rel = list(q[relevant_col])
        row = {
            "query_id": q.get("query_id"),
            "source": q.get("source"),
            "subtype": q.get("subtype"),
            "n_relevant": len(rel),
        }
        for k in k_values:
            row[f"precision@{k}"] = precision_at_k(retrieved, rel, k)
            row[f"recall@{k}"] = recall_at_k(retrieved, rel, k)
            row[f"mrr@{k}"] = mrr_at_k(retrieved, rel, k)
        rows.append(row)
    return pd.DataFrame(rows)


def metric_columns(k_values: Sequence[int] = (1, 3, 5, 10)) -> List[str]:
    cols = []
    for k in k_values:
        cols += [f"precision@{k}", f"recall@{k}", f"mrr@{k}"]
    return cols


# ----------------------------------------------------------------------
# Convenience loader (mirrors chunkers.load_corpus metadata handling)
# ----------------------------------------------------------------------
def load_chunks(path) -> pd.DataFrame:
    """Load chunks_n200.parquet. Metadata stays a JSON string here (BM25 does not
    need it parsed); the notebook parses it only when displaying a result."""
    return pd.read_parquet(Path(path))
