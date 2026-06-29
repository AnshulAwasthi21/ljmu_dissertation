"""
src/evaluation/evaluator.py

Phase 8 — Evaluation of the Phase-7 generations (QFS answers).

Scores the 180 frozen generations and answers the thesis question: does retrieval
choice change grounded-generation quality? TWO judge axes + a proxy + a human tier:

  AXIS 1 — FAITHFULNESS (vs the retrieved CONTEXT). Claim-level extract-then-verify:
           the judge sees query + context + answer (NEVER the gold answer) and labels
           each extracted claim supported/partial/unsupported. faithfulness = mean
           credit. This is INTRINSIC HALLUCINATION — did the model invent beyond what
           it was given? Phase-7 showed the model grounds in whatever it retrieves, so
           this is expected to be high and roughly retriever-independent.

  AXIS 2 — CORRECTNESS (vs the GOLD chunk). A SEPARATE pass: the judge sees query +
           the gold/reference passage + answer (NEVER the retrieved context) and rates
           correct/partial/incorrect. This is ANSWER ACCURACY — the retrieval-induced
           error mode. When the right chunk is not retrieved the model produces a
           confident, faithful-to-context but INCORRECT answer; only this axis sees it.
           The retrieval->generation coupling lives here, gated by gold_in_context.

  SECONDARY — ROUGE-L vs the gold chunk (a PROXY reference; interpretable only where
           gold_in_context == True). Self-contained LCS impl; never leads a claim.

  TERTIARY — HITL calibration: a stratified sheet hand-scored on the SAME rubrics,
           reported as human-judge agreement (validates BOTH automated axes).

Why two passes (not one): keeping faithfulness and correctness as separate calls means
the faithfulness judge never sees the answer key, so faithfulness cannot be contaminated
by the ground truth, and correctness is an explicit, separate comparison. This is the
viva-defensible design.

Design (mirrors the rest of the codebase)
-----------------------------------------
- **Injectable judges**: ``faith_fn(query, answer, context) -> dict`` and
  ``correct_fn(query, answer, gold_text) -> dict``, exactly like the generator's
  injectable ``complete_fn``. Tests inject fakes and run fully offline.
- **Abstention is a separate category**, never a faithfulness/correctness failure. An
  abstained answer has no claims and answers nothing, so both axes are N/A; it is
  reported as its own rate. Penalising it would reward confident confabulation.
- **gold_in_context decomposition** lands on CORRECTNESS (where the gap is), with
  faithfulness as the contrasting flat axis.
- **Hash gates**: the notebook asserts BOTH the golden hash and the Phase-7 generations
  hash before scoring.

No metric depends on parsing the model's ``[n]`` citations.

Single-positive correctness caveat (Limitations): correctness is judged against the ONE
designated gold chunk. An answer drawn from a sibling chunk that is factually right may be
graded against a reference that does not fully contain it; reported as a known limitation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

DEFAULT_JUDGE_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42

# Per-claim support label -> numeric credit. faithfulness = mean over claims.
SUPPORT_CREDIT = {"yes": 1.0, "partial": 0.5, "no": 0.0}
# Correctness label -> numeric credit.
CORRECTNESS_CREDIT = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


# ======================================================================
# ROUGE-L (self-contained, LCS-based F1 — no external dependency)
# ======================================================================
_RL_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")


def _rouge_tokens(text: str) -> List[str]:
    """Lowercase tokenise; decimals kept intact (consistent with the BM25 tokenizer)."""
    if not isinstance(text, str):
        return []
    return _RL_TOKEN.findall(text.lower())


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence (classic DP)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_fscore(candidate: str, reference: str) -> float:
    """Standard ROUGE-L F1 (Lin 2004): LCS-based precision/recall, beta=1. 0.0 if
    either side empty. Self-contained, deterministic, dependency-free."""
    cand, ref = _rouge_tokens(candidate), _rouge_tokens(reference)
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


# ======================================================================
# AXIS 1 — Faithfulness judge (vs context). Claim-level extract-then-verify.
# ======================================================================
JUDGE_SYSTEM_PROMPT = (
    "You are a meticulous financial-analysis evaluator. You judge whether an "
    "ANSWER is grounded in a set of numbered CONTEXT passages and whether it "
    "addresses the QUESTION. The context passages are the ONLY ground truth; do "
    "not use outside knowledge.\n\n"
    "Procedure:\n"
    "1. If the answer declines to answer or states the context is insufficient, "
    'set "is_abstention" true and return an empty "claims" list.\n'
    "2. Otherwise decompose the answer into its distinct factual claims (each a "
    "single checkable assertion: a figure, date, named entity, or statement).\n"
    "3. For each claim decide whether the CONTEXT supports it:\n"
    '   "yes"     = directly stated or unambiguously entailed by a passage;\n'
    '   "partial" = partly supported, or right topic but wrong magnitude/qualifier;\n'
    '   "no"      = not supported by any passage (a hallucination w.r.t. context).\n'
    "   Record the supporting passage number, or null.\n"
    '4. Rate "relevance" in [0,1]: how well the answer addresses the QUESTION, '
    "regardless of factual correctness.\n\n"
    "Return ONLY a JSON object (no markdown, no prose) with keys exactly:\n"
    '{"is_abstention": bool, "claims": [{"claim": str, "supported": '
    '"yes"|"partial"|"no", "passage": int|null}], "relevance": number, '
    '"notes": str}'
)


def build_judge_user_prompt(query: str, answer: str, context_text: str) -> str:
    return (
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT PASSAGES:\n{context_text}\n\n"
        f"ANSWER TO EVALUATE:\n{answer}\n\n"
        "Evaluate the answer against the context passages as instructed. "
        "Return only the JSON object."
    )


def parse_judge_json(raw: str) -> Dict:
    """Robustly parse the faithfulness judge's JSON, tolerating fences/prose. Safe
    defaults on failure so one bad reply degrades one row, not the sweep."""
    default = {"is_abstention": False, "claims": [], "relevance": None,
               "notes": "", "parse_error": True}
    if not isinstance(raw, str) or not raw.strip():
        return default
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE).strip()
    obj = None
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        return default
    claims = obj.get("claims", []) or []
    clean = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        sup = str(c.get("supported", "")).strip().lower()
        if sup not in SUPPORT_CREDIT:
            sup = "no"
        clean.append({"claim": str(c.get("claim", "")).strip(),
                      "supported": sup, "passage": c.get("passage")})
    rel = obj.get("relevance")
    try:
        rel = float(rel) if rel is not None else None
    except (TypeError, ValueError):
        rel = None
    return {"is_abstention": bool(obj.get("is_abstention", False)),
            "claims": clean, "relevance": rel,
            "notes": str(obj.get("notes", "")).strip(), "parse_error": False}


def compute_faithfulness(claims: Sequence[Dict]) -> Optional[float]:
    """Mean per-claim credit (yes=1, partial=0.5, no=0). None when no claims."""
    if not claims:
        return None
    return sum(SUPPORT_CREDIT.get(c.get("supported", "no"), 0.0) for c in claims) / len(claims)


# ======================================================================
# AXIS 2 — Correctness judge (vs gold chunk). Separate pass; no context shown.
# ======================================================================
CORRECTNESS_SYSTEM_PROMPT = (
    "You are a meticulous financial-analysis evaluator assessing ANSWER ACCURACY. "
    "You are given a QUESTION, a REFERENCE passage that is the ground-truth source "
    "for the question, and an ANSWER. Judge ONLY whether the answer's factual "
    "content is correct and consistent with the REFERENCE. The answer may have been "
    "generated from other material; that is irrelevant — compare its facts to the "
    "reference only. Do not use outside knowledge.\n\n"
    "Procedure:\n"
    "1. If the answer declines / says the context is insufficient, set "
    '"is_abstention" true and "correct" to "incorrect" (it did not answer).\n'
    "2. Otherwise rate factual correctness against the REFERENCE:\n"
    '   "correct"   = the key facts/figures asked for match the reference;\n'
    '   "partial"   = partially correct, or correct topic but wrong/missing a key '
    "figure or qualifier;\n"
    '   "incorrect" = the answer\'s key facts conflict with or are absent from the '
    "reference (a retrieval-induced or hallucinated error).\n\n"
    "Return ONLY a JSON object (no markdown, no prose) with keys exactly:\n"
    '{"is_abstention": bool, "correct": "correct"|"partial"|"incorrect", '
    '"note": str}'
)


def build_correctness_user_prompt(query: str, answer: str, gold_text: str) -> str:
    return (
        f"QUESTION:\n{query}\n\n"
        f"REFERENCE (ground-truth source passage):\n{gold_text}\n\n"
        f"ANSWER TO EVALUATE:\n{answer}\n\n"
        "Judge the answer's factual correctness against the REFERENCE as instructed. "
        "Return only the JSON object."
    )


def parse_correctness_json(raw: str) -> Dict:
    """Parse the correctness judge's JSON, safe defaults on failure."""
    default = {"is_abstention": False, "correct": "incorrect", "note": "",
               "parse_error": True}
    if not isinstance(raw, str) or not raw.strip():
        return default
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE).strip()
    obj = None
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        return default
    label = str(obj.get("correct", "")).strip().lower()
    if label not in CORRECTNESS_CREDIT:
        label = "incorrect"
    return {"is_abstention": bool(obj.get("is_abstention", False)),
            "correct": label, "note": str(obj.get("note", "")).strip(),
            "parse_error": False}


# ======================================================================
# Scoring one generation + the full sweep
# ======================================================================
SCORE_COLUMNS = [
    "query_id", "retriever", "source", "subtype", "gold_in_context", "gold_rank",
    "abstained",
    # faithfulness axis
    "judge_is_abstention", "n_claims", "n_supported", "n_partial", "n_unsupported",
    "faithfulness", "relevance", "unsupported_claims", "judge_notes",
    "judge_parse_error",
    # correctness axis
    "correctness", "correctness_label", "correctness_note", "correctness_parse_error",
    # proxy
    "rouge_l",
]


def score_one(
    row,
    judge_fn: Callable[[str, str, str], Dict],
    correctness_fn: Optional[Callable[[str, str, str], Dict]] = None,
    gold_text: Optional[str] = None,
) -> Dict:
    """Score one generation on both axes.

    Abstentions (Phase-7 exact-match flag) are NOT sent to either judge: faithfulness,
    relevance and correctness are all N/A (None); the row counts only in the abstention
    rate. Non-abstentions get the faithfulness pass (vs context) and, if ``correctness_fn``
    and ``gold_text`` are provided, the correctness pass (vs gold). ROUGE-L uses gold_text.
    """
    abstained = bool(row["abstained"])
    base = {
        "query_id": row["query_id"], "retriever": row["retriever"],
        "source": row["source"], "subtype": row["subtype"],
        "gold_in_context": bool(row["gold_in_context"]),
        "gold_rank": row["gold_rank"], "abstained": abstained,
    }

    if abstained:
        base.update({
            "judge_is_abstention": True, "n_claims": 0, "n_supported": 0,
            "n_partial": 0, "n_unsupported": 0, "faithfulness": None,
            "relevance": None, "unsupported_claims": [],
            "judge_notes": "abstained (not judged)", "judge_parse_error": False,
            "correctness": None, "correctness_label": "abstained",
            "correctness_note": "abstained (not judged)", "correctness_parse_error": False,
            "rouge_l": None,
        })
        return base

    # --- faithfulness pass (vs context) ---
    v = judge_fn(row["query_text"], row["answer"], row["context_text"])
    claims = v.get("claims", [])
    n_yes = sum(1 for c in claims if c.get("supported") == "yes")
    n_par = sum(1 for c in claims if c.get("supported") == "partial")
    n_no = sum(1 for c in claims if c.get("supported") == "no")
    unsupported = [c["claim"] for c in claims if c.get("supported") in ("no", "partial")]

    # --- correctness pass (vs gold) ---
    correctness = correctness_label = correctness_note = None
    correctness_parse_error = False
    if correctness_fn is not None and gold_text is not None:
        cv = correctness_fn(row["query_text"], row["answer"], gold_text)
        correctness_label = cv.get("correct", "incorrect")
        correctness = CORRECTNESS_CREDIT.get(correctness_label, 0.0)
        correctness_note = cv.get("note", "")
        correctness_parse_error = bool(cv.get("parse_error", False))

    base.update({
        "judge_is_abstention": bool(v.get("is_abstention", False)),
        "n_claims": len(claims), "n_supported": n_yes, "n_partial": n_par,
        "n_unsupported": n_no,
        "faithfulness": compute_faithfulness(claims),
        "relevance": v.get("relevance"),
        "unsupported_claims": unsupported,
        "judge_notes": v.get("notes", ""),
        "judge_parse_error": bool(v.get("parse_error", False)),
        "correctness": correctness, "correctness_label": correctness_label,
        "correctness_note": correctness_note,
        "correctness_parse_error": correctness_parse_error,
        "rouge_l": rouge_l_fscore(row["answer"], gold_text) if gold_text is not None else None,
    })
    return base


def run_evaluation(
    gen_df: pd.DataFrame,
    judge_fn: Callable[[str, str, str], Dict],
    correctness_fn: Optional[Callable[[str, str, str], Dict]] = None,
    gold_text_by_chunk: Optional[Dict[str, str]] = None,
    *,
    progress: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Score every generation -> one-row-per-generation scores DataFrame, on both axes.

    ``gold_text_by_chunk`` maps chunk_id -> text (from chunks_n200.parquet); used for the
    correctness reference and the ROUGE-L proxy. ``correctness_fn`` optional (None -> skip
    the correctness axis, e.g. in offline faithfulness-only tests)."""
    rows: List[Dict] = []
    total = len(gen_df)
    for i, (_, row) in enumerate(gen_df.iterrows(), start=1):
        gold_text = None
        if gold_text_by_chunk is not None:
            rel = list(row["relevant_chunk_ids"])
            if rel:
                gold_text = gold_text_by_chunk.get(str(rel[0]))
        rows.append(score_one(row, judge_fn, correctness_fn=correctness_fn, gold_text=gold_text))
        if progress is not None:
            progress(i, total)
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


# ======================================================================
# OpenAI judge factories (lazy; injected by the notebook)
# ======================================================================
def make_openai_judge_fn(client, model: str = DEFAULT_JUDGE_MODEL,
                         temperature: float = DEFAULT_TEMPERATURE,
                         seed: int = DEFAULT_SEED) -> Callable[[str, str, str], Dict]:
    """Faithfulness judge: judge_fn(query, answer, context) -> dict (JSON response)."""
    def judge(query: str, answer: str, context: str) -> Dict:
        resp = client.chat.completions.create(
            model=model, temperature=temperature, seed=seed,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                      {"role": "user", "content": build_judge_user_prompt(query, answer, context)}],
        )
        return parse_judge_json(resp.choices[0].message.content)
    return judge


def make_openai_correctness_fn(client, model: str = DEFAULT_JUDGE_MODEL,
                               temperature: float = DEFAULT_TEMPERATURE,
                               seed: int = DEFAULT_SEED) -> Callable[[str, str, str], Dict]:
    """Correctness judge: correct_fn(query, answer, gold_text) -> dict (JSON response)."""
    def judge(query: str, answer: str, gold_text: str) -> Dict:
        resp = client.chat.completions.create(
            model=model, temperature=temperature, seed=seed,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": CORRECTNESS_SYSTEM_PROMPT},
                      {"role": "user", "content": build_correctness_user_prompt(query, answer, gold_text)}],
        )
        return parse_correctness_json(resp.choices[0].message.content)
    return judge


# ======================================================================
# Aggregation (descriptive only — no significance testing)
# ======================================================================
def _mean_no_groupby_apply(df: pd.DataFrame, by, col) -> Dict:
    """Mean of `col` grouped by `by`, skip NaN, plain dict. Uses groupby()[col].mean()
    (NOT groupby.apply — removed in pandas 3)."""
    g = df.groupby(by)[col].mean()
    out = {}
    for k, v in g.items():
        key = "/".join(map(str, k)) if isinstance(k, tuple) else str(k)
        out[key] = (None if pd.isna(v) else round(float(v), 4))
    return out


def summarize_scores(scored_df: pd.DataFrame) -> Dict:
    """Headline aggregates for phase_08_summary.json. Faithfulness/relevance/correctness
    means are over NON-abstained rows. The gold_in_context decomposition is reported for
    BOTH axes (the contrast is the result: faithfulness ~flat, correctness gated)."""
    answered = scored_df[~scored_df["abstained"]].copy()
    gic = answered[answered["gold_in_context"]]

    def by_gic(metric):
        return {
            "gold_in_context_true": _mean_no_groupby_apply(
                answered[answered["gold_in_context"]], "retriever", metric),
            "gold_in_context_false": _mean_no_groupby_apply(
                answered[~answered["gold_in_context"]], "retriever", metric),
        }

    return {
        "n_scored": int(len(scored_df)),
        "n_answered": int(len(answered)),
        "n_abstained": int(scored_df["abstained"].sum()),
        "judge_parse_errors": int(scored_df["judge_parse_error"].sum()),
        "correctness_parse_errors": int(scored_df["correctness_parse_error"].sum()),
        "faithfulness": {
            "by_retriever": _mean_no_groupby_apply(answered, "retriever", "faithfulness"),
            "by_retriever_source": _mean_no_groupby_apply(
                answered, ["retriever", "source"], "faithfulness"),
            "by_gold_in_context": by_gic("faithfulness"),
        },
        "correctness": {
            "by_retriever": _mean_no_groupby_apply(answered, "retriever", "correctness"),
            "by_retriever_source": _mean_no_groupby_apply(
                answered, ["retriever", "source"], "correctness"),
            "by_gold_in_context": by_gic("correctness"),
        },
        "relevance": {
            "by_retriever": _mean_no_groupby_apply(answered, "retriever", "relevance"),
        },
        "abstention_rate": _mean_no_groupby_apply(scored_df, "retriever", "abstained"),
        "rouge_l": {
            "by_retriever_all": _mean_no_groupby_apply(answered, "retriever", "rouge_l"),
            "by_retriever_gold_in_context": _mean_no_groupby_apply(gic, "retriever", "rouge_l"),
        },
    }


# ======================================================================
# HITL calibration sheet + human-judge agreement (both axes)
# ======================================================================
def build_hitl_sheet(scored_df: pd.DataFrame, gen_df: pd.DataFrame,
                     gold_text_by_chunk: Dict[str, str], n_per_cell: int = 3,
                     seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Stratified hand-scoring sheet (n_per_cell per retriever×source). Includes the
    query, gold chunk text, the context the model saw, the answer, the judge scores
    (for agreement), and BLANK human_faithful / human_correct / human_relevant cols."""
    merged = scored_df.merge(
        gen_df[["query_id", "retriever", "query_text", "context_text",
                "answer", "relevant_chunk_ids"]],
        on=["query_id", "retriever"], how="left")
    pool = merged[~merged["abstained"]]
    picks = []
    for retr in sorted(merged["retriever"].unique()):
        for src in sorted(merged["source"].unique()):
            cell = pool[(pool["retriever"] == retr) & (pool["source"] == src)]
            if len(cell) < n_per_cell:
                extra = merged[(merged["retriever"] == retr) & (merged["source"] == src)]
                cell = pd.concat([cell, extra]).drop_duplicates(["query_id", "retriever"])
            picks.append(cell.sample(n=min(n_per_cell, len(cell)), random_state=seed))
    sheet = pd.concat(picks, ignore_index=True)

    sheet["gold_chunk_text"] = sheet["relevant_chunk_ids"].apply(
        lambda r: gold_text_by_chunk.get(str(list(r)[0])) if len(list(r)) else None)
    sheet["judge_faithful_bin"] = (sheet["faithfulness"] >= 0.5).astype("Int64")
    sheet["judge_correct_bin"] = (sheet["correctness"] >= 0.5).astype("Int64")
    sheet["judge_relevant_bin"] = (sheet["relevance"] >= 0.5).astype("Int64")
    sheet["human_faithful"] = pd.NA
    sheet["human_correct"] = pd.NA
    sheet["human_relevant"] = pd.NA
    sheet["human_notes"] = ""

    cols = ["query_id", "retriever", "source", "subtype", "gold_in_context",
            "query_text", "gold_chunk_text", "context_text", "answer",
            "faithfulness", "correctness", "relevance",
            "judge_faithful_bin", "judge_correct_bin", "judge_relevant_bin",
            "human_faithful", "human_correct", "human_relevant", "human_notes"]
    return sheet[cols].sort_values(["retriever", "source", "query_id"]).reset_index(drop=True)


def _cohen_kappa(human: Sequence[int], judge: Sequence[int]) -> Optional[float]:
    pairs = [(int(h), int(j)) for h, j in zip(human, judge)
             if h is not None and j is not None and not pd.isna(h) and not pd.isna(j)]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for h, j in pairs if h == j) / n
    ph1 = sum(h for h, _ in pairs) / n
    pj1 = sum(j for _, j in pairs) / n
    pe = ph1 * pj1 + (1 - ph1) * (1 - pj1)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def human_judge_agreement(filled_sheet: pd.DataFrame) -> Dict:
    """Agreement between researcher hand-scores and the automated judges on the HITL
    sheet — validation of BOTH axes. Plain agreement % + Cohen's kappa each."""
    out = {}
    for dim, hcol, jcol in (("faithfulness", "human_faithful", "judge_faithful_bin"),
                            ("correctness", "human_correct", "judge_correct_bin"),
                            ("relevance", "human_relevant", "judge_relevant_bin")):
        if hcol not in filled_sheet.columns or jcol not in filled_sheet.columns:
            out[dim] = {"n": 0, "agreement": None, "cohen_kappa": None}
            continue
        sub = filled_sheet[[hcol, jcol]].dropna()
        n = len(sub)
        if n == 0:
            out[dim] = {"n": 0, "agreement": None, "cohen_kappa": None}
            continue
        agree = float((sub[hcol].astype(int) == sub[jcol].astype(int)).mean())
        k = _cohen_kappa(sub[hcol].tolist(), sub[jcol].tolist())
        out[dim] = {"n": int(n), "agreement": round(agree, 4),
                    "cohen_kappa": (round(k, 4) if k is not None else None)}
    return out


# ======================================================================
# Hash gates + IO
# ======================================================================
def assert_generations_hash(gen_df: pd.DataFrame, expected_sha256: str) -> str:
    """Recompute the Phase-7 generations hash and assert it matches."""
    from src.generation.generator import hash_generations
    actual = hash_generations(gen_df)
    if actual != expected_sha256:
        raise AssertionError(
            f"GENERATIONS HASH MISMATCH\n expected {expected_sha256}\n got      {actual}")
    return actual


def save_scores(df: pd.DataFrame, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "unsupported_claims" in out.columns:
        out["unsupported_claims"] = out["unsupported_claims"].apply(
            lambda v: list(v) if isinstance(v, (list, tuple)) else [])
    out.to_parquet(path, index=False)


def load_scores(path) -> pd.DataFrame:
    df = pd.read_parquet(Path(path))
    if "unsupported_claims" in df.columns:
        df["unsupported_claims"] = df["unsupported_claims"].apply(
            lambda v: list(v) if v is not None else [])
    return df


def build_gold_text_lookup(chunks_df: pd.DataFrame, id_col: str = "chunk_id",
                           text_col: str = "text") -> Dict[str, str]:
    """chunk_id -> text map for the correctness reference and ROUGE-L proxy."""
    return dict(zip(chunks_df[id_col].astype(str), chunks_df[text_col].astype(str)))
