"""
src/retrieval/dense_retriever.py

Phase 5 — Dense semantic-retrieval baseline.

Indexes the SAME `text` column of chunks_n200.parquet that BM25 (Phase 4)
indexed — the shared retrieval control from Phase 3. Every retrieved result
preserves chunk_id + inherited metadata, identical in shape to
``BM25Retriever.retrieve`` so the Phase-4 ``evaluate_retriever`` harness drives
this retriever UNCHANGED. Any cross-retriever difference is therefore the
retriever, not the corpus, the eval set, or the metric code.

Design decisions (see phase_05_retrieval_decisions.md for the why)
------------------------------------------------------------------
- **Model:** ``BAAI/bge-small-en-v1.5`` (384-dim, 512-token window, MIT). The
  512-token window covers the 1500-char chunks WITHOUT truncation — a 256-token
  model (e.g. all-MiniLM-L6-v2) would clip ~1/3 of every chunk and clip EDGAR
  (longer, denser) harder than earnings, biasing the very contrast we measure.
- **Asymmetric prefix (correctness-critical):** BGE expects a retrieval
  instruction on the QUERY only; passages get no prefix. ``query_prefix`` is
  applied to queries at retrieve-time and NEVER to the corpus. Getting this
  backwards silently degrades dense and would corrupt the EDGAR-vs-earnings
  conclusion, so it is centralised here and asserted in tests.
- **Index:** FAISS ``IndexFlatIP`` over L2-normalised vectors == EXACT cosine.
  Exactness keeps the comparison honest (BM25 is exact too — an ANN index would
  compare exact-sparse vs approximate-dense). At 21k–63k vectors a flat scan is
  sub-millisecond, so approximation buys nothing. Falls back to a NumPy matmul
  (identical exact result) if faiss is unavailable.
- **Caching:** passage embeddings (the only expensive step) are cached to disk,
  keyed by model + chunk config + a hash of the sorted chunk_ids. A re-run (or
  the optional n=300 robustness run) loads instead of re-embedding; if the chunk
  set changes the key changes and the cache invalidates — you cannot silently
  score on stale vectors.

Interface parity with BM25Retriever
-----------------------------------
``retrieve(query, k) -> List[Dict]`` (metadata + text + score + rank) and
``retrieve_ids(query, k) -> List[str]`` match the Phase-4 signatures exactly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Defaults — change the model here and everything downstream follows.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# BGE s2p (short query -> long passage) retrieval instruction. Query-only.
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_PASSAGE_PREFIX = ""  # passages get no prefix for BGE
DEFAULT_BATCH_SIZE = 64
DEFAULT_CACHE_DIR = "data/embeddings"


# ======================================================================
# Embedding helpers (sentence-transformers imported lazily)
# ======================================================================
def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; safe against zero rows."""
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def load_model(model_name: str = DEFAULT_MODEL):
    """Load a SentenceTransformer. Lazy import so the module loads (and the
    metric/index logic is testable) without sentence-transformers installed."""
    from sentence_transformers import SentenceTransformer  # lazy
    return SentenceTransformer(model_name)


def _encode(model, texts: List[str], batch_size: int, prefix: str) -> np.ndarray:
    """Encode texts with an optional prefix, returning L2-normalised float32.
    Normalisation is done here (not relying on the model flag) so the contract
    is explicit and identical for the FAISS and NumPy backends."""
    prepared = [f"{prefix}{t}" if prefix else t for t in texts]
    emb = model.encode(
        prepared,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,  # we normalise explicitly below
    )
    return _l2_normalize(np.asarray(emb, dtype=np.float32))


def chunk_ids_fingerprint(chunk_ids: Sequence[str]) -> str:
    """Order-independent short hash of the chunk-id set. Part of the cache key:
    if the corpus changes, cached embeddings are invalidated automatically."""
    blob = "\n".join(sorted(map(str, chunk_ids)))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _cache_paths(cache_dir, model_name: str, chunk_cfg: str, fp: str):
    safe_model = model_name.replace("/", "__")
    stem = f"emb__{safe_model}__{chunk_cfg}__{fp}"
    base = Path(cache_dir)
    return base / f"{stem}.npy", base / f"{stem}.meta.json"


def embed_corpus(
    chunks_df: pd.DataFrame,
    model=None,
    model_name: str = DEFAULT_MODEL,
    text_col: str = "text",
    id_col: str = "chunk_id",
    passage_prefix: str = DEFAULT_PASSAGE_PREFIX,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> np.ndarray:
    """Embed the corpus passages, position-aligned to ``chunks_df`` rows.

    Caches to ``cache_dir`` keyed by model + chunk config + chunk-id fingerprint.
    Returns an (n_rows, dim) L2-normalised float32 array. ``model`` may be passed
    in to avoid reloading; otherwise it is loaded lazily IF a cache miss occurs.
    """
    ids = chunks_df[id_col].astype(str).tolist()
    fp = chunk_ids_fingerprint(ids)
    # chunk config string for the cache key (kept human-readable)
    cs = chunks_df["chunk_size_cfg"].iloc[0] if "chunk_size_cfg" in chunks_df.columns else "na"
    ov = chunks_df["overlap_cfg"].iloc[0] if "overlap_cfg" in chunks_df.columns else "na"
    chunk_cfg = f"cs{cs}_ov{ov}_n{len(ids)}"

    npy_path = meta_path = None
    if use_cache and cache_dir is not None:
        npy_path, meta_path = _cache_paths(cache_dir, model_name, chunk_cfg, fp)
        if npy_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            emb = np.load(npy_path)
            if meta.get("fingerprint") == fp and emb.shape[0] == len(ids):
                return emb.astype(np.float32, copy=False)

    if model is None:
        model = load_model(model_name)
    texts = chunks_df[text_col].fillna("").astype(str).tolist()
    emb = _encode(model, texts, batch_size=batch_size, prefix=passage_prefix)

    if use_cache and cache_dir is not None:
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, emb)
        meta_path.write_text(json.dumps({
            "model_name": model_name,
            "chunk_cfg": chunk_cfg,
            "fingerprint": fp,
            "n_rows": int(emb.shape[0]),
            "dim": int(emb.shape[1]),
            "passage_prefix": passage_prefix,
            "normalized": True,
        }, indent=2))
    return emb


# ======================================================================
# Dense retriever
# ======================================================================
class DenseRetriever:
    """Exact dense retriever (FAISS IndexFlatIP / cosine) over a chunk DataFrame.

    Parameters
    ----------
    chunks_df : DataFrame with at least ``id_col`` and ``text_col``. Extra columns
        (source, subtype, metadata, ...) are preserved and returned, matching
        BM25Retriever.
    embeddings : (n_rows, dim) array, L2-normalised, position-aligned to
        ``chunks_df`` after ``reset_index``. Produced by ``embed_corpus``.
    query_encoder : callable(str | List[str]) -> np.ndarray returning
        L2-normalised query vectors. For production use ``from_model`` wires this
        to the model (with the query prefix); tests may inject a fake encoder so
        the index/metric logic is exercised without downloading a model.
    """

    def __init__(
        self,
        chunks_df: pd.DataFrame,
        embeddings: np.ndarray,
        query_encoder: Callable[[List[str]], np.ndarray],
        text_col: str = "text",
        id_col: str = "chunk_id",
        model_name: str = DEFAULT_MODEL,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
    ):
        if text_col not in chunks_df.columns or id_col not in chunks_df.columns:
            raise KeyError(f"chunks_df must contain '{id_col}' and '{text_col}'")
        if chunks_df[id_col].duplicated().any():
            raise ValueError(f"{id_col} must be unique to keep results traceable")
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.shape[0] != len(chunks_df):
            raise ValueError(
                f"embeddings rows ({emb.shape[0]}) != chunks_df rows ({len(chunks_df)})"
            )

        self.text_col = text_col
        self.id_col = id_col
        self.model_name = model_name
        self.query_prefix = query_prefix
        self._query_encoder = query_encoder

        self._df = chunks_df.reset_index(drop=True).copy()
        self.chunk_ids: List[str] = self._df[id_col].astype(str).tolist()
        self._meta_cols = [c for c in self._df.columns if c != text_col]
        self._emb = emb
        self.dim = int(emb.shape[1])
        self._index = self._build_index(emb)

    # -- index backends -------------------------------------------------
    def _build_index(self, emb: np.ndarray):
        """FAISS IndexFlatIP if available, else a NumPy fallback object. Both are
        EXACT inner-product search; on L2-normalised vectors that is cosine."""
        try:
            import faiss  # lazy
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            self.backend = "faiss"
            return index
        except Exception:
            self.backend = "numpy"
            return None  # _search uses self._emb directly

    def _search(self, q_vecs: np.ndarray, k: int):
        """Return (scores, idxs) for each query row, top-k by cosine desc.
        Tie-break is deterministic by corpus position to match BM25Retriever."""
        n = len(self.chunk_ids)
        kk = min(k, n)
        if self.backend == "faiss":
            scores, idxs = self._index.search(q_vecs, kk)
        else:
            sims = q_vecs @ self._emb.T  # (n_queries, n_chunks)
            idxs = np.argsort(-sims, axis=1)[:, :kk]
            scores = np.take_along_axis(sims, idxs, axis=1)
        # Deterministic re-sort by (-score, position); negligible at float
        # precision but guarantees the same stable contract BM25 gives on ties.
        out_scores, out_idxs = [], []
        for row_s, row_i in zip(scores, idxs):
            pairs = [(float(s), int(i)) for s, i in zip(row_s, row_i) if int(i) != -1]
            pairs.sort(key=lambda p: (-p[0], p[1]))
            out_scores.append([p[0] for p in pairs])
            out_idxs.append([p[1] for p in pairs])
        return out_scores, out_idxs

    # -- public API (mirrors BM25Retriever) -----------------------------
    def __len__(self) -> int:
        return len(self.chunk_ids)

    def _encode_query(self, query: str) -> np.ndarray:
        vec = self._query_encoder([query])
        return np.asarray(vec, dtype=np.float32).reshape(1, -1)

    def _row_payload(self, i: int, score: float, rank: int) -> Dict:
        row = self._df.iloc[i]
        payload = {c: row[c] for c in self._meta_cols}
        payload["text"] = row[self.text_col]
        payload["score"] = float(score)
        payload["rank"] = rank
        return payload

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """Top-`k` chunks as dicts with full metadata + cosine score + rank,
        ordered by descending similarity. Returns up to k (fewer if corpus < k)."""
        if k <= 0 or len(self) == 0:
            return []
        q = self._encode_query(query)
        scores, idxs = self._search(q, k)
        s0, i0 = scores[0], idxs[0]
        return [self._row_payload(i, s, rank + 1)
                for rank, (s, i) in enumerate(zip(s0, i0))]

    def retrieve_ids(self, query: str, k: int = 10) -> List[str]:
        """Lightweight path: top-`k` chunk_ids only (used by the metric loop)."""
        return [r[self.id_col] for r in self.retrieve(query, k)]

    # -- construction from a model (with caching) -----------------------
    @classmethod
    def from_model(
        cls,
        chunks_df: pd.DataFrame,
        model_name: str = DEFAULT_MODEL,
        text_col: str = "text",
        id_col: str = "chunk_id",
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        passage_prefix: str = DEFAULT_PASSAGE_PREFIX,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        use_cache: bool = True,
        model=None,
    ) -> "DenseRetriever":
        """Load the model (once), embed the corpus (cached), and build the index.
        The SAME model instance encodes both passages and queries, with the query
        prefix applied to queries only."""
        if model is None:
            model = load_model(model_name)
        emb = embed_corpus(
            chunks_df, model=model, model_name=model_name, text_col=text_col,
            id_col=id_col, passage_prefix=passage_prefix, batch_size=batch_size,
            cache_dir=cache_dir, use_cache=use_cache,
        )

        def query_encoder(texts: List[str]) -> np.ndarray:
            return _encode(model, list(texts), batch_size=batch_size, prefix=query_prefix)

        return cls(
            chunks_df, embeddings=emb, query_encoder=query_encoder,
            text_col=text_col, id_col=id_col, model_name=model_name,
            query_prefix=query_prefix,
        )


# ======================================================================
# Convenience loader (mirrors bm25_retriever.load_chunks)
# ======================================================================
def load_chunks(path) -> pd.DataFrame:
    """Load chunks_n200.parquet. Metadata stays a JSON string here (dense does
    not need it parsed); the notebook parses it only when displaying a result."""
    return pd.read_parquet(Path(path))
