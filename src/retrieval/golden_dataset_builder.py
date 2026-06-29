"""
src/retrieval/golden_dataset_builder.py

Phase 4 — Golden relevance-judgement set for retrieval evaluation.

Strategy (Option C, locked in phase_04_retrieval_decisions.md):
  LLM seeds candidate queries from a STRATIFIED sample of real chunks; the
  researcher then CURATES them (paraphrases away lexical echo of the source
  chunk, drops weak ones, confirms relevant chunks). The frozen, hashed output
  is reused byte-for-byte by dense (Phase 5) and hybrid (Phase 6) so all three
  retrievers are scored on the identical query set.

Why a manual curation break exists
-----------------------------------
The single most important mitigation in this design is rewriting each generated
query so it does NOT reuse the source chunk's vocabulary. LLM-from-chunk queries
otherwise hand BM25 a free lexical match and bias the *BM25-vs-dense delta* — the
headline comparison of the thesis. Curation is therefore methodological, not
cosmetic, and cannot be automated away.

Why we freeze the OUTPUT, not the generation step
--------------------------------------------------
LLM generation is only best-effort reproducible (temperature + provider `seed`
reduce but do not eliminate variation). We do NOT rely on re-running generation
to reproduce the set. We persist the curated parquet and a content hash; that
artifact is the reproducible object. seed=42 governs the *chunk sampling*, which
is fully deterministic.

Pipeline:
  1. sample_seed_chunks(...)            -> deterministic stratified seed sample
  2. build_candidate_set(...)           -> calls LLM, writes golden_candidates.csv
  3. [MANUAL] researcher edits the CSV  -> paraphrase queries, keep/drop, confirm chunks
  4. load_curated_golden_set(...)       -> validates, normalises -> DataFrame
  5. save_golden_set(...) / hash_golden_set(...) -> golden_queries.parquet + hash
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import pandas as pd

SEED = 42

# A chunk shorter than this is a poor seed (e.g. "Operator : Thank you.") — it
# cannot anchor a substantive analyst query. Filters the short-chunk tail
# documented in phase_03_results.md §5.
DEFAULT_MIN_SEED_CHARS = 400

GENERATION_METHOD = "llm_seed_curated"
DEFAULT_GEN_MODEL = "gpt-4o-mini"

# Pipe is the in-CSV separator for the relevant-chunk list, chosen because it
# never appears in a chunk_id (chunk_ids are "{doc_id}_chunk_{i:03d}").
LIST_SEP = "|"


# ----------------------------------------------------------------------
# 1. Deterministic stratified seed sampling
# ----------------------------------------------------------------------
def _allocate_proportional(counts: dict, n: int) -> dict:
    """Allocate `n` slots across keys proportionally to `counts`, deterministically.

    Largest-remainder method: floor proportional shares, then hand the leftover
    slots to the keys with the largest fractional remainder (ties broken by key
    name for determinism). Never allocates more than a key actually has.
    """
    total = sum(counts.values())
    if total == 0 or n <= 0:
        return {k: 0 for k in counts}
    raw = {k: (counts[k] * n / total) for k in counts}
    alloc = {k: min(int(v), counts[k]) for k, v in raw.items()}
    assigned = sum(alloc.values())
    leftover = n - assigned
    # Order candidates by (remainder desc, key asc) for stable top-up.
    order = sorted(counts.keys(), key=lambda k: (-(raw[k] - int(raw[k])), k))
    i = 0
    guard = 0
    while leftover > 0 and guard < 10_000:
        k = order[i % len(order)]
        if alloc[k] < counts[k]:
            alloc[k] += 1
            leftover -= 1
        i += 1
        guard += 1
    return alloc


def sample_seed_chunks(
    chunks_df: pd.DataFrame,
    n_edgar: int = 30,
    n_earnings: int = 30,
    seed: int = SEED,
    min_chars: int = DEFAULT_MIN_SEED_CHARS,
) -> pd.DataFrame:
    """Draw a deterministic, subtype-stratified seed sample from each corpus arm.

    Returns the sampled chunk rows (a subset of `chunks_df`) with a stable order.
    Stratification is proportional to each subtype's chunk count within its arm,
    so EDGAR and earnings are each represented across their subtypes.
    """
    df = chunks_df[chunks_df["char_length"] >= min_chars].copy()
    picked: List[pd.DataFrame] = []

    for source, n in (("edgar", n_edgar), ("earnings", n_earnings)):
        arm = df[df["source"] == source]
        if arm.empty or n <= 0:
            continue
        counts = arm["subtype"].value_counts().to_dict()
        alloc = _allocate_proportional(counts, min(n, len(arm)))
        for subtype, k in alloc.items():
            if k <= 0:
                continue
            pool = arm[arm["subtype"] == subtype]
            picked.append(pool.sample(n=min(k, len(pool)), random_state=seed))

    if not picked:
        return chunks_df.iloc[0:0].copy()

    out = pd.concat(picked, ignore_index=True)
    # Stable, reproducible ordering independent of concat order.
    out = out.sort_values("chunk_id").reset_index(drop=True)
    return out


# ----------------------------------------------------------------------
# 2. Overlap-sibling candidates (multi-positive support)
# ----------------------------------------------------------------------
def find_overlap_siblings(
    chunk_id: str, chunks_df: pd.DataFrame, window: int = 1
) -> List[str]:
    """Return chunk_ids adjacent to `chunk_id` within the SAME parent doc_id.

    Adjacent chunks share `overlap` chars (200) by construction, so they are
    strong multi-positive *candidates* — offered to the curator to confirm, never
    auto-marked relevant (a neighbour may genuinely change topic).
    """
    rows = chunks_df.loc[chunks_df["chunk_id"] == chunk_id]
    if rows.empty:
        return []
    row = rows.iloc[0]
    doc_id, idx = row["doc_id"], int(row["chunk_index"])
    sib = chunks_df[
        (chunks_df["doc_id"] == doc_id)
        & (chunks_df["chunk_index"].between(idx - window, idx + window))
        & (chunks_df["chunk_id"] != chunk_id)
    ]
    return sorted(sib["chunk_id"].tolist())


# ----------------------------------------------------------------------
# 3. LLM query generation (client is injectable for testing)
# ----------------------------------------------------------------------
_GEN_SYSTEM = (
    "You are a financial analyst writing retrieval-evaluation queries. Given a "
    "passage from a 10-K filing or an earnings call, write natural analyst "
    "questions whose answer is contained in THAT passage. Questions must be "
    "answerable from the passage alone, specific, and realistic. Return ONLY a "
    "JSON array of strings, no prose, no markdown fences."
)


def _build_gen_user_prompt(chunk_text: str, n_questions: int) -> str:
    snippet = chunk_text[:3500]
    return (
        f"Write {n_questions} analyst question(s) answerable from this passage. "
        f"Return a JSON array of {n_questions} string(s).\n\nPASSAGE:\n{snippet}"
    )


def _parse_json_array(raw: str) -> List[str]:
    """Robustly parse a JSON array of strings, tolerating ```json fences."""
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        # Last resort: pull the first [...] block.
        m = re.search(r"\[.*\]", txt, flags=re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(data, list):
        return [str(q).strip() for q in data if str(q).strip()]
    return []


def make_openai_client(api_key: Optional[str] = None):
    """Construct an OpenAI client. Imported lazily so the module loads without
    the openai package installed (tests inject a fake client instead)."""
    from openai import OpenAI  # lazy import

    return OpenAI(api_key=api_key) if api_key else OpenAI()


def generate_queries_for_chunk(
    chunk_text: str,
    client,
    model: str = DEFAULT_GEN_MODEL,
    n_questions: int = 1,
    temperature: float = 0.3,
    seed: int = SEED,
) -> List[str]:
    """Generate `n_questions` candidate queries for one chunk via the chat API.

    `client` must expose `client.chat.completions.create(...)` (the OpenAI v1
    interface). Injecting a fake client makes this unit-testable offline.
    """
    if not isinstance(chunk_text, str) or not chunk_text.strip():
        return []
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        seed=seed,
        messages=[
            {"role": "system", "content": _GEN_SYSTEM},
            {"role": "user", "content": _build_gen_user_prompt(chunk_text, n_questions)},
        ],
    )
    content = resp.choices[0].message.content
    return _parse_json_array(content)


# ----------------------------------------------------------------------
# 4. Candidate-set assembly (-> CSV for manual curation)
# ----------------------------------------------------------------------
CANDIDATE_COLUMNS = [
    "query_id",
    "keep",                 # 1 = keep, 0 = drop  (curator edits)
    "query_text_raw",       # LLM output (read-only reference)
    "query_text_curated",   # curator paraphrases here (THIS is what gets used)
    "source",
    "subtype",
    "seed_chunk_id",
    "relevant_chunk_ids",   # pipe-separated; pre-filled with seed chunk only
    "candidate_siblings",   # pipe-separated neighbours to optionally merge in
    "notes",
]


def build_candidate_set(
    seed_chunks: pd.DataFrame,
    chunks_df: pd.DataFrame,
    client,
    model: str = DEFAULT_GEN_MODEL,
    n_questions: int = 1,
    seed: int = SEED,
    sibling_window: int = 1,
) -> pd.DataFrame:
    """Generate candidate queries for each seed chunk and assemble the curation
    table. `relevant_chunk_ids` is pre-filled with the seed chunk only; adjacent
    overlap-siblings are surfaced in `candidate_siblings` for optional confirmation.
    """
    rows = []
    qn = 0
    for _, sc in seed_chunks.iterrows():
        cid = sc["chunk_id"]
        queries = generate_queries_for_chunk(
            sc["text"], client=client, model=model,
            n_questions=n_questions, seed=seed,
        )
        siblings = find_overlap_siblings(cid, chunks_df, window=sibling_window)
        for q in queries:
            rows.append({
                "query_id": f"q{qn:03d}",
                "keep": 1,
                "query_text_raw": q,
                "query_text_curated": "",  # curator fills this in
                "source": sc["source"],
                "subtype": sc["subtype"],
                "seed_chunk_id": cid,
                "relevant_chunk_ids": cid,                       # seed only by default
                "candidate_siblings": LIST_SEP.join(siblings),   # confirm in curation
                "notes": "",
            })
            qn += 1
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def save_candidate_set(df: pd.DataFrame, path) -> None:
    """Write the candidate table to CSV (UTF-8-SIG so Excel on Windows opens it
    cleanly) for manual curation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ----------------------------------------------------------------------
# 5. Load + validate the curated set -> frozen parquet
# ----------------------------------------------------------------------
GOLDEN_COLUMNS = [
    "query_id", "query_text", "source", "subtype",
    "seed_chunk_id", "relevant_chunk_ids", "generation_method",
    "curated", "seed", "notes",
]


def _parse_id_list(raw) -> List[str]:
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    if raw is None or (isinstance(raw, float)):
        return []
    return [tok.strip() for tok in str(raw).split(LIST_SEP) if tok.strip()]


def load_curated_golden_set(
    curation_csv,
    chunks_df: pd.DataFrame,
    seed: int = SEED,
    require_curated_text: bool = True,
) -> pd.DataFrame:
    """Read the curated CSV, validate it, and return the canonical golden set.

    Validation (fails loudly — a silently broken gold set corrupts every metric):
      - drops rows with keep == 0
      - uses `query_text_curated` (falls back to raw only if not required)
      - every relevant_chunk_id MUST exist in `chunks_df`
      - every query MUST have >= 1 relevant chunk
      - query_ids unique
    """
    raw = pd.read_csv(curation_csv, encoding="utf-8-sig")
    kept = raw[raw.get("keep", 1).fillna(1).astype(int) == 1].copy()

    valid_ids = set(chunks_df["chunk_id"])
    out_rows = []
    problems = []

    def _cell(r, key):
        v = r.get(key, "")
        return "" if pd.isna(v) else str(v).strip()

    for _, r in kept.iterrows():
        qid = str(r["query_id"]).strip()
        curated = _cell(r, "query_text_curated")
        rawq = _cell(r, "query_text_raw")
        if require_curated_text and not curated:
            problems.append(f"{qid}: empty query_text_curated (paraphrase required)")
            continue
        query_text = curated or rawq
        if not query_text:
            problems.append(f"{qid}: no query text at all")
            continue

        rel = _parse_id_list(r.get("relevant_chunk_ids"))
        # optionally merge confirmed siblings if the curator moved them in;
        # we only honour what is in relevant_chunk_ids (explicit), not candidates.
        rel = sorted(set(rel))
        missing = [c for c in rel if c not in valid_ids]
        if missing:
            problems.append(f"{qid}: relevant chunk(s) not in corpus: {missing}")
            continue
        if not rel:
            problems.append(f"{qid}: no relevant chunks")
            continue

        out_rows.append({
            "query_id": qid,
            "query_text": query_text,
            "source": str(r["source"]).strip(),
            "subtype": str(r["subtype"]).strip(),
            "seed_chunk_id": str(r["seed_chunk_id"]).strip(),
            "relevant_chunk_ids": rel,
            "generation_method": GENERATION_METHOD,
            "curated": bool(curated),
            "seed": seed,
            "notes": _cell(r, "notes"),
        })

    if problems:
        raise ValueError(
            "Golden-set validation failed:\n  - " + "\n  - ".join(problems)
        )

    df = pd.DataFrame(out_rows, columns=GOLDEN_COLUMNS)
    if df["query_id"].duplicated().any():
        dups = df.loc[df["query_id"].duplicated(), "query_id"].tolist()
        raise ValueError(f"Duplicate query_ids: {dups}")
    return df.sort_values("query_id").reset_index(drop=True)


# ----------------------------------------------------------------------
# 6. Persist + hash (the reproducible artifact)
# ----------------------------------------------------------------------
def hash_golden_set(df: pd.DataFrame) -> str:
    """Deterministic content hash of the golden set. Lets the viva prove that
    Phases 4/5/6 used the identical query set. Order-independent over queries."""
    items = []
    for _, r in df.iterrows():
        items.append({
            "query_id": r["query_id"],
            "query_text": r["query_text"],
            "relevant_chunk_ids": sorted(list(r["relevant_chunk_ids"])),
        })
    items.sort(key=lambda d: d["query_id"])
    blob = json.dumps(items, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def save_golden_set(df: pd.DataFrame, path) -> str:
    """Persist the golden set to parquet (list column survives via pyarrow).
    Returns the content hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    # ensure relevant_chunk_ids is a clean python list for pyarrow list<string>
    out["relevant_chunk_ids"] = out["relevant_chunk_ids"].apply(
        lambda v: list(v) if isinstance(v, (list, tuple)) else _parse_id_list(v)
    )
    out.to_parquet(path, index=False)
    return hash_golden_set(out)


def load_golden_set(path) -> pd.DataFrame:
    """Load the frozen golden set, normalising the list column back to python lists."""
    df = pd.read_parquet(path)
    df["relevant_chunk_ids"] = df["relevant_chunk_ids"].apply(
        lambda v: list(v) if v is not None else []
    )
    return df
