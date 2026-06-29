"""
src/chunking/chunkers.py

Phase 3 chunking for the thesis corpus.

Two strategies, dispatched on the `source` column of the Phase-2 corpus:
  - EDGAR (formal prose, flat text)  -> recursive character splitting
  - Earnings (conversational)        -> speaker-turn-aware greedy packing

Design constraints (carried from Phase 2):
  - Chunks NEVER span document units (each parent doc_id is chunked independently).
  - Every chunk inherits its parent doc_id + metadata for retrieval traceability.
  - Target chunk size 1500 chars, overlap 200 chars.
  - Baseline only: no hierarchical / node-based chunking yet.

IMPORTANT -- speaker-turn detection
-----------------------------------
Phase 2 (log item #10) established that the source earnings transcripts contain
NO newlines between speaker turns; the text arrives as a single flat string.
`preserve_newlines=True` therefore has no practical effect on this dataset.

Consequently speaker turns CANNOT be detected with a newline-anchored regex --
that would yield zero turns and collapse each call into one unsplittable blob.
We detect turns by the source's actual convention: a capitalised speaker label
immediately followed by " : " (space-colon-space), e.g. "Michael Petters : ...".

Invariants guaranteed by this module (verified in tests + the notebook):
  - No chunk exceeds `chunk_size` characters (overlap is counted within budget).
  - No empty chunks are emitted.
  - chunk_id always begins with its parent doc_id (a chunk cannot merge two docs).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_OVERLAP = 200

# Recursive splitter separator hierarchy, finest-last. Paragraph/line
# separators are kept at the top for forward-compatibility with scale-up data
# that may carry structure; EDGAR text in this corpus is flat, so in practice
# ". " and " " do the work. The empty string "" must remain last: it guarantees
# we can always reduce a stubborn span to sub-chunk-size pieces, which is what
# keeps the "no oversized chunk" invariant true.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

# Speaker label = 1-4 capitalised tokens immediately followed by " : ".
# Tokens may contain apostrophes/hyphens ("O'Brien", "Smith-Jones") but NOT
# digits, so "Q3 :" or "2016 :" never match. We deliberately EXCLUDE the
# period from the token class: including it lets a token swallow the previous
# sentence's full stop (".. to Mike." + " Michael Petters :" -> one label),
# shifting the turn boundary by a word. Titles like "Mr." appear only in
# references, never as turn labels, so dropping "." is safe here.
# The space-before-colon is the source's convention and is a strong,
# low-false-positive signal once the text has been whitespace-normalised
# in Phase 2.
SPEAKER_LABEL_RE = re.compile(
    r"[A-Z][A-Za-z\u2019'\-]+(?:\s+[A-Z][A-Za-z\u2019'\-]+){0,3}\s:\s"
)


# ======================================================================
# Recursive character splitter (EDGAR / formal prose)
# ======================================================================
def _split_keep_separator(text: str, sep: str) -> List[str]:
    """Split on `sep`, keeping the separator attached to the preceding piece
    so that lengths and reconstruction stay faithful. sep == "" -> chars."""
    if sep == "":
        return list(text)
    parts = text.split(sep)
    out: List[str] = []
    for i, p in enumerate(parts):
        out.append(p + sep if i < len(parts) - 1 else p)
    return [p for p in out if p != ""]


def _merge_splits(splits: List[str], chunk_size: int, overlap: int) -> List[str]:
    """Greedily merge pieces (each already carrying its trailing separator,
    so we join with "") into chunks <= chunk_size, retaining up to `overlap`
    chars of trailing context at each boundary."""
    docs: List[str] = []
    current: List[str] = []
    total = 0
    for d in splits:
        dlen = len(d)
        if current and total + dlen > chunk_size:
            doc = "".join(current).strip()
            if doc:
                docs.append(doc)
            # Drop from the front until we are within `overlap` AND the
            # incoming piece will fit. This is what carries context forward.
            while current and (total > overlap or total + dlen > chunk_size):
                total -= len(current[0])
                current.pop(0)
        current.append(d)
        total += dlen
    doc = "".join(current).strip()
    if doc:
        docs.append(doc)
    return docs


def _hard_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Last-resort fixed-window split (only reached if a separator hierarchy
    without a trailing "" is supplied). Kept for safety."""
    step = max(1, chunk_size - overlap)
    return [text[i:i + chunk_size] for i in range(0, len(text), step)]


def _recursive(text: str, separators: List[str], chunk_size: int,
               overlap: int) -> List[str]:
    final: List[str] = []
    # Choose the first separator that occurs ("" always "occurs").
    sep = separators[-1]
    rest: List[str] = []
    for i, s in enumerate(separators):
        if s == "":
            sep, rest = "", []
            break
        if s in text:
            sep, rest = s, separators[i + 1:]
            break

    splits = _split_keep_separator(text, sep)
    good: List[str] = []
    for s in splits:
        if len(s) <= chunk_size:
            good.append(s)
        else:
            if good:
                final.extend(_merge_splits(good, chunk_size, overlap))
                good = []
            if rest:
                final.extend(_recursive(s, rest, chunk_size, overlap))
            else:
                final.extend(_hard_window(s, chunk_size, overlap))
    if good:
        final.extend(_merge_splits(good, chunk_size, overlap))
    return [c for c in final if c]


def recursive_character_split(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """Recursively split formal prose on a separator hierarchy, packing pieces
    up to `chunk_size` with `overlap` chars of trailing context carried over."""
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    return _recursive(text, separators or DEFAULT_SEPARATORS, chunk_size, overlap)


# ======================================================================
# Speaker-turn-aware packer (earnings / conversational)
# ======================================================================
def split_into_speaker_turns(text: str) -> List[str]:
    """Split a flat transcript string into speaker turns using the ' : ' label
    convention. Returns one segment per turn (label included). Any preamble
    before the first label (rare) is returned as its own leading segment.

    If NO speaker label is found, returns the whole text as a single segment
    rather than silently dropping content -- but note that this means the unit
    will be chunked as one blob, which is the failure mode we are guarding
    against; the tests assert real transcripts produce many turns.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    matches = list(SPEAKER_LABEL_RE.finditer(text))
    if not matches:
        return [text.strip()]

    turns: List[str] = []
    first = matches[0].start()
    if first > 0:
        pre = text[:first].strip()
        if pre:
            turns.append(pre)
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg = text[start:end].strip()
        if seg:
            turns.append(seg)
    return turns


def _packed_len(parts: List[str]) -> int:
    """Length of parts joined by single spaces."""
    if not parts:
        return 0
    return sum(len(p) for p in parts) + (len(parts) - 1)


def _build_overlap_carry(parts: List[str], overlap: int) -> List[str]:
    """Take whole trailing turns whose combined length fits within `overlap`."""
    carry: List[str] = []
    for prev in reversed(parts):
        if _packed_len(carry) + (1 if carry else 0) + len(prev) <= overlap:
            carry.insert(0, prev)
        else:
            break
    return carry


def chunk_by_speaker_turns(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """Greedy-pack whole speaker turns into chunks <= chunk_size, carrying
    whole trailing turns (up to `overlap` chars) into the next chunk for
    context overlap. Turns longer than `chunk_size` (e.g. a long CEO answer)
    are sub-split with the recursive splitter so the size invariant holds."""
    turns = split_into_speaker_turns(text)
    if not turns:
        return []

    # Expand oversized turns so no atomic unit exceeds chunk_size.
    units: List[str] = []
    for t in turns:
        if len(t) <= chunk_size:
            units.append(t)
        else:
            units.extend(recursive_character_split(t, chunk_size, overlap))

    chunks: List[str] = []
    current: List[str] = []
    for u in units:
        projected = _packed_len(current) + (1 if current else 0) + len(u)
        if current and projected > chunk_size:
            chunks.append(" ".join(current))
            carry = _build_overlap_carry(current, overlap)
            # If the overlap carry plus the incoming turn would exceed the
            # budget, drop the carry for this boundary rather than emit an
            # oversized chunk. Overlap is therefore best-effort, the size
            # cap is hard.
            if _packed_len(carry) + (1 if carry else 0) + len(u) > chunk_size:
                carry = []
            current = carry
        current.append(u)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


# ======================================================================
# Dispatch + corpus-level chunking
# ======================================================================
def chunk_document(
    text: str,
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """Route a single document unit to the strategy for its corpus."""
    if source == "earnings":
        return chunk_by_speaker_turns(text, chunk_size, overlap)
    # edgar and any unknown source fall back to recursive prose splitting.
    return recursive_character_split(text, chunk_size, overlap)


def strategy_for_source(source: str) -> str:
    return "speaker_turn" if source == "earnings" else "recursive_char"


def chunk_corpus(
    df: pd.DataFrame,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> pd.DataFrame:
    """Chunk a Phase-2 corpus DataFrame into a one-row-per-chunk DataFrame.

    Expected input columns (Phase-2 unified schema):
        doc_id, source, corpus_type, subtype, text, char_length, metadata
    `metadata` may be a dict (in-memory) or a JSON string (loaded from parquet).

    Output columns:
        chunk_id, doc_id, source, corpus_type, subtype, chunk_index, n_chunks,
        chunk_strategy, chunk_size_cfg, overlap_cfg, text, char_length, metadata
    """
    rows = []
    for _, doc in df.iterrows():
        doc_id = doc.get("doc_id")
        source = doc.get("source", "")
        text = doc.get("text", "")

        meta = doc.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}

        strategy = strategy_for_source(source)
        pieces = chunk_document(text, source, chunk_size, overlap)
        n = len(pieces)
        for i, piece in enumerate(pieces):
            rows.append({
                "chunk_id": f"{doc_id}_chunk_{i:03d}",
                "doc_id": doc_id,
                "source": source,
                "corpus_type": doc.get("corpus_type", ""),
                "subtype": doc.get("subtype", ""),
                "chunk_index": i,
                "n_chunks": n,
                "chunk_strategy": strategy,
                "chunk_size_cfg": chunk_size,
                "overlap_cfg": overlap,
                "text": piece,
                "char_length": len(piece),
                "metadata": meta,  # inherited parent metadata (dict)
            })

    cols = [
        "chunk_id", "doc_id", "source", "corpus_type", "subtype",
        "chunk_index", "n_chunks", "chunk_strategy", "chunk_size_cfg",
        "overlap_cfg", "text", "char_length", "metadata",
    ]
    return pd.DataFrame(rows, columns=cols)


def save_chunks(df: pd.DataFrame, path) -> None:
    """Persist chunks to parquet, JSON-serialising metadata (Phase-2 convention
    -- avoids the mixed-schema Arrow error documented in phase_02_log #5)."""
    out = df.copy()
    out["metadata"] = out["metadata"].apply(lambda m: json.dumps(m, default=str))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def load_corpus(path) -> pd.DataFrame:
    """Load a Phase-2 corpus parquet and deserialise the JSON metadata column."""
    df = pd.read_parquet(path)
    if "metadata" in df.columns:
        df["metadata"] = df["metadata"].apply(
            lambda m: json.loads(m) if isinstance(m, str) else (m or {})
        )
    return df
