"""
src/generation/generator.py

Phase 7 — Retrieval-augmented query-focused summarisation (QFS) generation.

This is the ablation that connects the retrieval track (Phases 4-6) to answer
quality. For each of the 60 frozen golden queries and each of the three
retrievers {bm25, dense, hybrid}, we retrieve top-k context, assemble it under
ONE fixed prompt, and call a fixed generator (gpt-4o-mini, temperature 0,
seed 42). The retriever is the ONLY manipulated variable; the LLM, the prompt,
k, and the context ordering are all held constant. Any difference in answer
quality is therefore attributable to retrieval.

Why this module looks the way it does
-------------------------------------
- **Injectable client (testability).** The generator never imports `openai`. It
  takes a ``complete_fn(system, user) -> dict`` callable, exactly mirroring how
  ``DenseRetriever`` takes an injectable ``query_encoder``. Tests inject a fake
  ``complete_fn`` and exercise prompt assembly, context formatting, provenance,
  ``gold_in_context`` logic, abstention detection, save/load and hashing WITHOUT
  touching the network. The notebook wires ``complete_fn`` to the real API.

- **The fixed prompt is the control.** It is defined once here as a module
  constant and fingerprinted (``prompt_version`` + a hash of the system text) so
  the saved generations carry proof that every one of the 180 calls used the
  identical instruction. Editing the prompt changes the fingerprint, which is the
  signal that a re-run is a different experiment.

- **Provenance is recorded by US, not parsed from the model.** We control exactly
  which chunks entered each context, so ``retrieved_chunk_ids`` (rank order),
  ``gold_in_context`` and ``gold_rank`` are exact. The model is asked to cite
  ``[n]`` for human traceability only; NO metric depends on parsing its citations.

- **Chunk ids are NOT shown to the model.** The context block uses bare ``[1]..[k]``
  markers. Chunk ids encode metadata (``edgar_0004_section_7_chunk_020``) that
  would leak the source/section into the prompt and confound the comparison; the
  id<->position mapping lives only in our provenance record.

- **gold_in_context is the mediating variable.** It records whether the
  single-positive gold chunk reached the top-k context (and at what rank). It is
  what lets Phase 8 separate the *retrieval* effect ("did the right chunk get
  there?") from the *generation* effect ("given it was there, was the answer
  grounded?").

- **Determinism is by freezing, not re-derivation.** gpt-4o-mini at temperature 0
  + seed is best-effort deterministic, not bitwise (OpenAI exposes
  ``system_fingerprint`` for exactly this reason). We record the fingerprint and
  treat the saved, SHA-256'd generations file as the canonical reproducible
  artifact — the same discipline the golden set uses.

Public surface
--------------
- ``build_context_block`` / ``build_user_prompt`` — prompt assembly (pure).
- ``gold_in_context`` — (bool, rank|None) for a retrieved id list vs relevant ids.
- ``generate_one`` — one (query x retriever) generation record.
- ``run_generation_matrix`` — the full 60 x 3 sweep -> DataFrame.
- ``make_openai_complete_fn`` — factory wiring a real OpenAI client (lazy).
- ``save_generations`` / ``load_generations`` / ``hash_generations`` — frozen artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# ----------------------------------------------------------------------
# Fixed generation config (the control)
# ----------------------------------------------------------------------
DEFAULT_GEN_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42
DEFAULT_K = 5

# Exact sentinel the model is told to emit when the context cannot answer the
# query. Phase 8 detects abstention by exact (stripped) match, so a faithful
# model with a missing gold chunk is scored as "abstained", not "hallucinated".
ABSTENTION_TEXT = (
    "The provided context does not contain enough information to answer this question."
)

# Bump this string if the prompt is ever edited; it travels into every record and
# into phase_07_summary.json so a changed prompt is impossible to run silently.
PROMPT_VERSION = "p7_qfs_v1"

GENERATION_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the user's question using "
    "ONLY the numbered context passages provided.\n"
    "- Ground every statement in the passages; do not use any outside knowledge.\n"
    "- Cite the passage number(s) you used in square brackets immediately after "
    "each claim, e.g. [1] or [2][3].\n"
    "- Be specific: reproduce the exact figures, dates, and named terms from the "
    "passages when they are relevant to the question.\n"
    "- If the passages do not contain enough information to answer the question, "
    f'reply with exactly this sentence and nothing else: "{ABSTENTION_TEXT}"\n'
    "- Keep the answer concise (a few sentences)."
)


def prompt_fingerprint() -> str:
    """Short hash of the fixed system prompt + version. Proves all 180 calls used
    the identical instruction; changes the moment the prompt text changes."""
    blob = f"{PROMPT_VERSION}\n{GENERATION_SYSTEM_PROMPT}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------
# Prompt assembly (pure functions — directly unit-testable)
# ----------------------------------------------------------------------
def build_context_block(
    hits: Sequence[Dict],
    text_col: str = "text",
) -> str:
    """Format retrieved hits into a numbered context block, rank order preserved.

    Passages are labelled ``[1]..[k]`` in the order the retriever returned them
    (best first). The ordering is IDENTICAL across all three retrievers — it is a
    control, and fixing it sidesteps lost-in-the-middle position confounds.

    Chunk ids are deliberately NOT included: they encode source/section metadata
    that would leak into the prompt. The position<->id mapping is recorded in the
    provenance fields instead.
    """
    parts = []
    for i, h in enumerate(hits, start=1):
        text = "" if h.get(text_col) is None else str(h.get(text_col))
        parts.append(f"[{i}]\n{text.strip()}")
    return "\n\n".join(parts)


def build_user_prompt(query: str, context_block: str) -> str:
    """Assemble the fixed user message: the question, then the numbered context,
    then a fixed instruction. Fixed across every query and retriever."""
    return (
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT PASSAGES:\n{context_block}\n\n"
        "Using only the passages above, write a grounded, query-focused answer "
        "to the question. Cite the passage numbers you rely on."
    )


def gold_in_context(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> Tuple[bool, Optional[int]]:
    """Did a relevant (gold) chunk reach the retrieved context, and at what rank?

    Returns ``(present, rank)`` where ``rank`` is the 1-indexed position of the
    first relevant id in ``retrieved_ids`` (or ``None`` if absent). With the
    single-positive design ``relevant_ids`` has length 1, but this handles the
    multi-positive case for free.
    """
    rel = set(map(str, relevant_ids))
    for i, cid in enumerate(retrieved_ids, start=1):
        if str(cid) in rel:
            return True, i
    return False, None


def is_abstention(answer: str) -> bool:
    """True if the model emitted the exact abstention sentinel (whitespace- and
    quote-tolerant), i.e. it correctly declined rather than confabulating."""
    if not isinstance(answer, str):
        return False
    cleaned = answer.strip().strip('"').strip()
    return cleaned == ABSTENTION_TEXT


# ----------------------------------------------------------------------
# Single generation
# ----------------------------------------------------------------------
def generate_one(
    query_text: str,
    relevant_ids: Sequence[str],
    retriever,
    retriever_name: str,
    complete_fn: Callable[[str, str], Dict],
    *,
    query_id: Optional[str] = None,
    source: Optional[str] = None,
    subtype: Optional[str] = None,
    k: int = DEFAULT_K,
    id_col: str = "chunk_id",
    text_col: str = "text",
    model_label: str = DEFAULT_GEN_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """Produce ONE generation record for a (query x retriever) cell.

    Steps: retrieve top-k -> assemble the fixed prompt -> call ``complete_fn`` ->
    record the answer alongside exact, parser-free provenance.

    ``complete_fn(system, user)`` must return a dict with at least ``{"text": str}``
    and may include ``"system_fingerprint"`` / ``"model"`` (the OpenAI factory does).
    """
    hits = retriever.retrieve(query_text, k=k)
    retrieved_ids = [str(h[id_col]) for h in hits]
    retrieved_scores = [float(h.get("score")) if h.get("score") is not None else None
                        for h in hits]

    context_block = build_context_block(hits, text_col=text_col)
    user_prompt = build_user_prompt(query_text, context_block)

    present, gold_rank = gold_in_context(retrieved_ids, relevant_ids)

    result = complete_fn(GENERATION_SYSTEM_PROMPT, user_prompt)
    answer = result.get("text", "") if isinstance(result, dict) else str(result)
    answer = "" if answer is None else str(answer)

    return {
        "query_id": query_id,
        "retriever": retriever_name,
        "source": source,
        "subtype": subtype,
        "query_text": query_text,
        "k": k,
        "relevant_chunk_ids": list(map(str, relevant_ids)),
        "retrieved_chunk_ids": retrieved_ids,         # rank order, best first
        "retrieved_scores": retrieved_scores,
        "gold_in_context": present,
        "gold_rank": gold_rank,                        # 1-indexed, None if absent
        "n_context": len(hits),
        "context_text": context_block,                 # exact text shown to model
        "answer": answer,
        "abstained": is_abstention(answer),
        "gen_model": (result.get("model") if isinstance(result, dict) else None) or model_label,
        "temperature": temperature,
        "seed": seed,
        "system_fingerprint": (result.get("system_fingerprint")
                               if isinstance(result, dict) else None),
        "prompt_version": PROMPT_VERSION,
        "prompt_fingerprint": prompt_fingerprint(),
    }


# Canonical column order for the saved artifact.
GENERATION_COLUMNS = [
    "query_id", "retriever", "source", "subtype", "query_text", "k",
    "relevant_chunk_ids", "retrieved_chunk_ids", "retrieved_scores",
    "gold_in_context", "gold_rank", "n_context", "context_text",
    "answer", "abstained", "gen_model", "temperature", "seed",
    "system_fingerprint", "prompt_version", "prompt_fingerprint",
]


# ----------------------------------------------------------------------
# Full matrix sweep
# ----------------------------------------------------------------------
def run_generation_matrix(
    golden_df: pd.DataFrame,
    retrievers: Dict[str, object],
    complete_fn: Callable[[str, str], Dict],
    *,
    k: int = DEFAULT_K,
    id_col: str = "chunk_id",
    text_col: str = "text",
    query_col: str = "query_text",
    relevant_col: str = "relevant_chunk_ids",
    model_label: str = DEFAULT_GEN_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
    progress: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Run every (golden query x retriever) cell -> a one-row-per-generation frame.

    ``retrievers`` maps a name -> a constructed retriever exposing
    ``retrieve(query, k)``. With 60 queries and 3 retrievers this yields 180 rows.
    Retrievers are iterated in sorted-name order and queries in their golden order,
    so row order is deterministic (the saved hash is order-independent regardless).

    ``progress(done, total)`` is an optional callback for a notebook progress bar.
    """
    rows: List[Dict] = []
    total = len(golden_df) * len(retrievers)
    done = 0
    for name in sorted(retrievers.keys()):
        retr = retrievers[name]
        for _, q in golden_df.iterrows():
            rec = generate_one(
                query_text=q[query_col],
                relevant_ids=list(q[relevant_col]),
                retriever=retr,
                retriever_name=name,
                complete_fn=complete_fn,
                query_id=q.get("query_id"),
                source=q.get("source"),
                subtype=q.get("subtype"),
                k=k, id_col=id_col, text_col=text_col,
                model_label=model_label, temperature=temperature, seed=seed,
            )
            rows.append(rec)
            done += 1
            if progress is not None:
                progress(done, total)
    return pd.DataFrame(rows, columns=GENERATION_COLUMNS)


# ----------------------------------------------------------------------
# OpenAI client factory (lazy; mirrors golden_dataset_builder.make_openai_client)
# ----------------------------------------------------------------------
def make_openai_complete_fn(
    client,
    model: str = DEFAULT_GEN_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
) -> Callable[[str, str], Dict]:
    """Wrap an OpenAI client into a ``complete_fn(system, user) -> dict``.

    Captures the fixed decoding config (model, temperature, seed) in the closure so
    every call in the sweep is identical. Returns ``text`` plus ``model`` and
    ``system_fingerprint`` for the provenance record. The client is injected (the
    notebook builds it); this keeps the module importable without ``openai`` and
    the tests fully offline.
    """
    def complete(system: str, user: str) -> Dict:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            seed=seed,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = resp.choices[0]
        return {
            "text": choice.message.content,
            "model": getattr(resp, "model", model),
            "system_fingerprint": getattr(resp, "system_fingerprint", None),
        }

    return complete


# ----------------------------------------------------------------------
# Frozen artifact: save / load / hash
# ----------------------------------------------------------------------
def hash_generations(df: pd.DataFrame) -> str:
    """Deterministic, order-independent content hash of the generations.

    Covers the fields that define the experiment's output: (query_id, retriever,
    retrieved_chunk_ids, answer). Phase 8 asserts this hash so it can prove it
    scored the exact frozen generations — the same contract the golden set uses.
    """
    items = []
    for _, r in df.iterrows():
        items.append({
            "query_id": str(r["query_id"]),
            "retriever": str(r["retriever"]),
            "retrieved_chunk_ids": list(map(str, r["retrieved_chunk_ids"])),
            "answer": "" if r["answer"] is None else str(r["answer"]),
        })
    items.sort(key=lambda d: (d["query_id"], d["retriever"]))
    blob = json.dumps(items, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def save_generations(df: pd.DataFrame, path) -> str:
    """Persist generations to parquet (list columns survive via pyarrow) and
    return the content hash. This parquet is the canonical frozen artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in ("relevant_chunk_ids", "retrieved_chunk_ids", "retrieved_scores"):
        if col in out.columns:
            out[col] = out[col].apply(
                lambda v: list(v) if isinstance(v, (list, tuple)) else ([] if v is None else v)
            )
    out.to_parquet(path, index=False)
    return hash_generations(out)


def load_generations(path) -> pd.DataFrame:
    """Load the frozen generations parquet, normalising list columns back to
    python lists (pyarrow returns numpy arrays for list<...> columns)."""
    df = pd.read_parquet(Path(path))
    for col in ("relevant_chunk_ids", "retrieved_chunk_ids", "retrieved_scores"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: list(v) if v is not None else []
            )
    return df
