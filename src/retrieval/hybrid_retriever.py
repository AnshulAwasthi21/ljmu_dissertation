"""
src/retrieval/hybrid_retriever.py

Phase 6 — Hybrid sparse+dense retrieval via Reciprocal Rank Fusion (RRF).

Composes the EXISTING Phase-4 ``BM25Retriever`` and Phase-5 ``DenseRetriever``
(it does NOT re-implement either) and fuses their per-query rankings with RRF,
exposing the SAME ``retrieve(query, k)`` / ``retrieve_ids(query, k)`` interface so
the Phase-4 ``evaluate_retriever`` harness drives all three configs UNCHANGED.
Any cross-retriever difference is therefore the retriever, not the corpus, the
eval set, or the metric code.

Design decisions (see phase_06_retrieval_decisions.md for the why)
------------------------------------------------------------------
- **Fusion = RRF, parameter-free k=60** (Cormack et al. 2009; proposal §7.4.1).
  For a chunk ``d``:  RRFscore(d) = Σ_arm 1 / (k + rank_arm(d)), rank 1-indexed.
  Ranks (not raw scores) are fused because BM25 scores (~tens) and dense cosine
  (~0..1) live on incomparable scales; rank position is the common currency.
  k=60 is left UNTUNED — tuning it on the single eval set would be circular and
  would smuggle a free variable into the clean three-way comparison, exactly as
  BM25 (k1=1.5,b=0.75) and the general-purpose dense encoder are left untuned.
  NOTE: RRF's ``k`` (fusion smoothing) is unrelated to BM25's ``k1`` — same letter,
  different knob. They never interact.

- **Candidate depth = top-100 per arm, fused over the UNION.** Each arm is asked
  for its top-``candidate_depth`` chunks; the fusion pool is the UNION of the two
  lists. A chunk present in only one arm contributes 1/(k+rank) from that arm and
  exactly 0 (no penalty floor) from the missing arm — this is what lets a BM25
  exact-match win that dense missed survive fusion, and vice-versa. Intersection
  (keep only chunks in both) is rejected: it discards single-arm finds and guts
  recall. Depth 100 is sufficient at this corpus size: beyond rank ~100 a single
  arm contributes <1/160 ≈ 0.006, too little to lift a chunk into the final top-10
  unless it is also ranked well by the other arm — in which case depth 100 already
  caught it. ``candidate_depth`` is configurable for the optional robustness check.

- **Deterministic tie-break by ``(-rrf_score, chunk_id)``.** RRF produces exact
  ties more often than score-based retrievers (two chunks hitting the same pair of
  ranks). BM25/dense tie-break by corpus position; the fused pool has no single
  corpus position, so we tie-break by ``chunk_id`` (lexicographic) — fully
  deterministic and reproducible, which is what the contract requires. Asserted in
  tests/test_hybrid.py.

Traceability
------------
Each returned dict carries ``chunk_id`` + inherited metadata + ``text`` (identical
in shape to BM25/dense), plus ``score`` (= the RRF score) and ``rank``. It ALSO
carries ``bm25_rank`` / ``dense_rank`` / ``bm25_score`` / ``dense_score`` (``None``
when an arm did not surface the chunk) — the raw material for the Phase-6
"where do the arms disagree?" analysis. ``score`` is on the RRF scale (not BM25's
nor cosine's); the metric harness only reads rank order + id membership, so the
scale is irrelevant to the metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # imported only for type hints; runtime uses duck typing
    from .bm25_retriever import BM25Retriever
    from .dense_retriever import DenseRetriever

DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_DEPTH = 100


# ======================================================================
# Pure RRF math (no retrievers, no DataFrame) — directly unit-testable
# ======================================================================
def rrf_contribution(rank: Optional[int], k: int = DEFAULT_RRF_K) -> float:
    """One arm's RRF contribution for a chunk at 1-indexed ``rank`` (``None`` ->
    chunk absent from that arm -> 0.0). Top result is rank 1, so the denominator
    for the best chunk is k+1 (e.g. k=60 -> 1/61 ≈ 0.01639)."""
    if rank is None or rank < 1:
        return 0.0
    return 1.0 / (k + rank)


def fuse_rankings(
    ranked_id_lists: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
) -> List[Tuple[str, float, List[Optional[int]]]]:
    """Fuse N ranked id-lists with RRF over their UNION.

    Parameters
    ----------
    ranked_id_lists : sequence of ranked chunk_id lists. Position 0 in a list is
        rank 1 for that arm. A chunk absent from a list is treated as rank=None
        (contributes 0 from that arm — no penalty floor).
    k : RRF smoothing constant (default 60).

    Returns
    -------
    list of ``(chunk_id, rrf_score, per_arm_ranks)`` sorted by
    ``(-rrf_score, chunk_id)`` (descending score, then chunk_id ascending for a
    deterministic, reproducible tie-break). ``per_arm_ranks[i]`` is the 1-indexed
    rank of the chunk in ``ranked_id_lists[i]`` or ``None`` if absent.
    """
    n_arms = len(ranked_id_lists)
    # Per-arm rank maps (first occurrence wins, though ids are unique per arm).
    rank_maps: List[Dict[str, int]] = []
    for lst in ranked_id_lists:
        rmap: Dict[str, int] = {}
        for pos, cid in enumerate(lst):
            if cid not in rmap:
                rmap[cid] = pos + 1  # 1-indexed
        rank_maps.append(rmap)

    # Union of all ids across arms.
    union_ids: List[str] = []
    seen = set()
    for lst in ranked_id_lists:
        for cid in lst:
            if cid not in seen:
                seen.add(cid)
                union_ids.append(cid)

    fused: List[Tuple[str, float, List[Optional[int]]]] = []
    for cid in union_ids:
        per_arm = [rank_maps[a].get(cid) for a in range(n_arms)]
        score = sum(rrf_contribution(r, k) for r in per_arm)
        fused.append((cid, score, per_arm))

    fused.sort(key=lambda t: (-t[1], t[0]))
    return fused


# ======================================================================
# Hybrid retriever (composition over BM25Retriever + DenseRetriever)
# ======================================================================
class HybridRetriever:
    """RRF fusion of two retrievers that each expose ``retrieve(query, k)``.

    Parameters
    ----------
    bm25, dense : already-constructed retrievers. Only ``retrieve(query, k)`` and
        ``id_col`` are required, so any retriever honouring the Phase-4 interface
        contract can be plugged in (duck-typed; no hard import).
    k : RRF smoothing constant (default 60, untuned).
    candidate_depth : how many chunks to pull from EACH arm before fusing
        (default 100). For a final-``k`` larger than this, the pull is deepened to
        ``max(candidate_depth, k)`` so the fused pool can always satisfy the request.
    id_col : chunk-id column name (default 'chunk_id'), must match both arms.
    """

    def __init__(
        self,
        bm25: "BM25Retriever",
        dense: "DenseRetriever",
        k: int = DEFAULT_RRF_K,
        candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
        id_col: str = "chunk_id",
    ):
        if k < 1:
            raise ValueError("RRF k must be >= 1")
        if candidate_depth < 1:
            raise ValueError("candidate_depth must be >= 1")
        self.bm25 = bm25
        self.dense = dense
        self.k = k
        self.candidate_depth = candidate_depth
        self.id_col = id_col

    def __len__(self) -> int:
        # Both arms index the same corpus; report whichever is available.
        for arm in (self.bm25, self.dense):
            try:
                return len(arm)
            except TypeError:
                continue
        return 0

    def _arm_results(self, query: str, depth: int) -> Tuple[List[Dict], List[Dict]]:
        bm25_hits = self.bm25.retrieve(query, k=depth)
        dense_hits = self.dense.retrieve(query, k=depth)
        return bm25_hits, dense_hits

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """Top-``k`` fused chunks as dicts: full metadata + ``text`` + ``score``
        (RRF) + ``rank``, plus per-arm ``bm25_rank``/``dense_rank``/``bm25_score``/
        ``dense_score`` (``None`` if an arm missed the chunk). Ordered by RRF score
        descending, ties broken by ``chunk_id``."""
        if k <= 0 or len(self) == 0:
            return []

        depth = max(self.candidate_depth, k)
        bm25_hits, dense_hits = self._arm_results(query, depth)

        bm25_ids = [h[self.id_col] for h in bm25_hits]
        dense_ids = [h[self.id_col] for h in dense_hits]

        # Per-arm score lookups for traceability (rank comes from fuse_rankings).
        bm25_score = {h[self.id_col]: float(h.get("score")) for h in bm25_hits}
        dense_score = {h[self.id_col]: float(h.get("score")) for h in dense_hits}

        # Payload lookups: prefer BM25's payload for metadata/text, fall back to
        # dense's. They are the same chunk, so metadata is identical; this just
        # makes the choice deterministic.
        payloads: Dict[str, Dict] = {}
        for h in dense_hits:
            payloads.setdefault(h[self.id_col], h)
        for h in bm25_hits:  # bm25 wins ties on payload source
            payloads[h[self.id_col]] = h

        fused = fuse_rankings([bm25_ids, dense_ids], k=self.k)

        out: List[Dict] = []
        for rank, (cid, rrf, per_arm) in enumerate(fused[:k], start=1):
            src = payloads[cid]
            payload = {c: src[c] for c in src
                       if c not in ("score", "rank")}
            payload["score"] = float(rrf)        # RRF score (different scale!)
            payload["rank"] = rank
            payload["bm25_rank"] = per_arm[0]
            payload["dense_rank"] = per_arm[1]
            payload["bm25_score"] = bm25_score.get(cid)
            payload["dense_score"] = dense_score.get(cid)
            out.append(payload)
        return out

    def retrieve_ids(self, query: str, k: int = 10) -> List[str]:
        """Lightweight path: top-``k`` fused chunk_ids only (used by the metric loop)."""
        return [r[self.id_col] for r in self.retrieve(query, k)]


# ----------------------------------------------------------------------
# Convenience loader (mirrors bm25_retriever.load_chunks / dense_retriever)
# ----------------------------------------------------------------------
def load_chunks(path) -> pd.DataFrame:
    """Load chunks_n200.parquet. Metadata stays a JSON string here (hybrid does
    not need it parsed); the notebook parses it only when displaying a result."""
    from pathlib import Path
    return pd.read_parquet(Path(path))
