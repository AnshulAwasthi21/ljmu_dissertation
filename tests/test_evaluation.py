"""
tests/test_evaluation.py

Phase 8 evaluation tests. Fully offline: fake faithfulness + correctness judges stand
in for gpt-4o. Exercises ROUGE-L, judge-JSON parsing (both axes), faithfulness math,
correctness scoring, the faithful-but-incorrect pattern, abstention (neither judge
called), the sweep, aggregation (incl. gold_in_context decomposition on both axes),
the HITL sheet + agreement (3 axes), and the generations-hash gate. No network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.evaluator import (
    SCORE_COLUMNS,
    assert_generations_hash,
    build_hitl_sheet,
    compute_faithfulness,
    human_judge_agreement,
    make_openai_correctness_fn,
    make_openai_judge_fn,
    parse_correctness_json,
    parse_judge_json,
    rouge_l_fscore,
    run_evaluation,
    score_one,
    summarize_scores,
)


# ---------------------------- ROUGE-L ----------------------------
def test_rouge_identical_is_one():
    assert rouge_l_fscore("revenue rose 12 percent", "revenue rose 12 percent") == 1.0

def test_rouge_disjoint_is_zero():
    assert rouge_l_fscore("alpha beta gamma", "delta epsilon zeta") == 0.0

def test_rouge_empty_is_zero():
    assert rouge_l_fscore("", "x") == 0.0 and rouge_l_fscore("x", "") == 0.0

def test_rouge_partial_between():
    assert 0.0 < rouge_l_fscore("net revenue increased to 326.8 million", "revenue increased 326.8") < 1.0

def test_rouge_keeps_decimals():
    assert rouge_l_fscore("326.8", "326.8") == 1.0 and rouge_l_fscore("326.8", "326.9") == 0.0


# ---------------------- faithfulness parsing ----------------------
def test_parse_plain_json():
    v = parse_judge_json('{"is_abstention": false, "claims": [{"claim":"x","supported":"yes","passage":1}], "relevance": 0.9, "notes":"ok"}')
    assert v["claims"][0]["supported"] == "yes" and v["relevance"] == 0.9 and v["parse_error"] is False

def test_parse_fenced_json():
    v = parse_judge_json('```json\n{"is_abstention": false, "claims": [], "relevance": 0.5, "notes":""}\n```')
    assert v["parse_error"] is False and v["relevance"] == 0.5

def test_parse_garbage_default():
    v = parse_judge_json("not json")
    assert v["parse_error"] is True and v["claims"] == [] and v["relevance"] is None

def test_parse_coerce_bad_label_and_relevance():
    v = parse_judge_json('{"claims":[{"claim":"x","supported":"maybe","passage":null}],"relevance":"high"}')
    assert v["claims"][0]["supported"] == "no" and v["relevance"] is None


# ---------------------- correctness parsing ----------------------
def test_parse_correctness_plain():
    v = parse_correctness_json('{"is_abstention": false, "correct": "correct", "note": "matches"}')
    assert v["correct"] == "correct" and v["parse_error"] is False

def test_parse_correctness_fenced_and_bad_label():
    v = parse_correctness_json('```json\n{"correct": "mostly", "note": ""}\n```')
    assert v["correct"] == "incorrect"  # unknown label -> incorrect
    assert v["parse_error"] is False

def test_parse_correctness_garbage_default():
    v = parse_correctness_json("nope")
    assert v["parse_error"] is True and v["correct"] == "incorrect"


# ---------------------- faithfulness math ----------------------
def test_faithfulness_all_supported():
    assert compute_faithfulness([{"supported": "yes"}, {"supported": "yes"}]) == 1.0

def test_faithfulness_none_supported():
    assert compute_faithfulness([{"supported": "no"}, {"supported": "no"}]) == 0.0

def test_faithfulness_partial_credit():
    assert compute_faithfulness([{"supported": "yes"}, {"supported": "no"}]) == 0.5
    assert compute_faithfulness([{"supported": "partial"}]) == 0.5

def test_faithfulness_empty_none():
    assert compute_faithfulness([]) is None


# ---------------------------- fakes ----------------------------
def _row(**kw):
    base = dict(query_id="q000", retriever="bm25", source="edgar", subtype="section_7",
                gold_in_context=True, gold_rank=2, abstained=False,
                query_text="What was revenue?", answer="Revenue was 326.8 million [1].",
                context_text="[1]\nRevenue was 326.8 million.",
                relevant_chunk_ids=["edgar_a_chunk_000"])
    base.update(kw); return base

def fake_faith_supported(q, a, c):
    return {"is_abstention": False, "claims": [{"claim": "rev 326.8m", "supported": "yes", "passage": 1}],
            "relevance": 0.9, "notes": "grounded", "parse_error": False}

def fake_faith_hallucinated(q, a, c):
    return {"is_abstention": False,
            "claims": [{"claim": "rev 999m", "supported": "no", "passage": None},
                       {"claim": "grew 16%", "supported": "partial", "passage": 2}],
            "relevance": 0.7, "notes": "", "parse_error": False}

def fake_correct_yes(q, a, g):
    return {"is_abstention": False, "correct": "correct", "note": "matches ref", "parse_error": False}

def fake_correct_no(q, a, g):
    return {"is_abstention": False, "correct": "incorrect", "note": "wrong figure", "parse_error": False}

def faith_must_not_be_called(q, a, c):
    raise AssertionError("faithfulness judge must NOT be called for abstained row")

def correct_must_not_be_called(q, a, g):
    raise AssertionError("correctness judge must NOT be called for abstained row")


# ---------------------------- score_one ----------------------------
def test_score_abstained_skips_both_judges():
    row = _row(abstained=True, answer="The provided context does not contain enough information to answer this question.")
    s = score_one(row, faith_must_not_be_called, correctness_fn=correct_must_not_be_called, gold_text="ref")
    assert s["faithfulness"] is None and s["relevance"] is None
    assert s["correctness"] is None and s["correctness_label"] == "abstained"
    assert s["rouge_l"] is None and s["n_claims"] == 0

def test_score_supported_and_correct():
    s = score_one(_row(), fake_faith_supported, correctness_fn=fake_correct_yes, gold_text="Revenue was 326.8 million")
    assert s["faithfulness"] == 1.0 and s["relevance"] == 0.9
    assert s["correctness"] == 1.0 and s["correctness_label"] == "correct"
    assert s["rouge_l"] is not None and s["rouge_l"] > 0

def test_score_FAITHFUL_BUT_INCORRECT_pattern():
    # the key Phase-7 finding: grounded in the (wrong) retrieved chunk -> faithful but wrong
    s = score_one(_row(gold_in_context=False, gold_rank=None),
                  fake_faith_supported, correctness_fn=fake_correct_no, gold_text="ref")
    assert s["faithfulness"] == 1.0       # grounded in what it saw
    assert s["correctness"] == 0.0        # but wrong vs the gold answer
    assert s["correctness_label"] == "incorrect"

def test_score_hallucinated_captures_unsupported():
    s = score_one(_row(), fake_faith_hallucinated, correctness_fn=fake_correct_no, gold_text="ref")
    assert s["faithfulness"] == 0.25 and s["n_unsupported"] == 1 and s["n_partial"] == 1
    assert "rev 999m" in s["unsupported_claims"] and "grew 16%" in s["unsupported_claims"]

def test_score_without_correctness_fn_leaves_correctness_none():
    s = score_one(_row(), fake_faith_supported, correctness_fn=None, gold_text="ref")
    assert s["correctness"] is None and s["rouge_l"] is not None


# ------------------------- run_evaluation -------------------------
def _gen_df():
    rows = []
    for retr, present in (("bm25", True), ("dense", False), ("hybrid", True)):
        for src in ("edgar", "earnings"):
            rows.append(_row(retriever=retr, source=src, gold_in_context=present,
                             query_id=f"q_{retr}_{src}", relevant_chunk_ids=["c_" + src]))
    return pd.DataFrame(rows)

def test_run_evaluation_shape_and_columns():
    gen = _gen_df()
    lookup = {"c_edgar": "edgar ref text revenue", "c_earnings": "earnings ref text"}
    scored = run_evaluation(gen, fake_faith_supported, correctness_fn=fake_correct_yes,
                            gold_text_by_chunk=lookup)
    assert len(scored) == len(gen) and list(scored.columns) == SCORE_COLUMNS
    assert scored["faithfulness"].notna().all() and scored["correctness"].notna().all()

def test_run_evaluation_deterministic():
    gen = _gen_df()
    a = run_evaluation(gen, fake_faith_hallucinated, correctness_fn=fake_correct_no)
    b = run_evaluation(gen, fake_faith_hallucinated, correctness_fn=fake_correct_no)
    pd.testing.assert_frame_equal(a, b)

def test_summarize_has_both_axes_and_decomposition():
    gen = _gen_df()
    scored = run_evaluation(gen, fake_faith_supported, correctness_fn=fake_correct_yes,
                            gold_text_by_chunk={"c_edgar": "x", "c_earnings": "y"})
    summ = summarize_scores(scored)
    assert set(summ["faithfulness"]["by_retriever"]) == {"bm25", "dense", "hybrid"}
    assert set(summ["correctness"]["by_retriever"]) == {"bm25", "dense", "hybrid"}
    # gold_in_context decomposition present on BOTH axes
    assert "dense" in summ["correctness"]["by_gold_in_context"]["gold_in_context_false"]
    assert "bm25" in summ["correctness"]["by_gold_in_context"]["gold_in_context_true"]
    assert "dense" in summ["faithfulness"]["by_gold_in_context"]["gold_in_context_false"]

def test_summarize_excludes_abstained():
    gen = _gen_df(); gen.loc[0, "abstained"] = True
    summ = summarize_scores(run_evaluation(gen, fake_faith_supported, correctness_fn=fake_correct_yes))
    assert summ["n_abstained"] == 1 and summ["n_answered"] == 5


# --------------------------- HITL + agreement ---------------------------
def test_hitl_sheet_has_three_blank_human_cols_and_gold_text():
    gen = _gen_df()
    scored = run_evaluation(gen, fake_faith_supported, correctness_fn=fake_correct_yes)
    sheet = build_hitl_sheet(scored, gen, {"c_edgar": "edgar gold", "c_earnings": "earnings gold"}, n_per_cell=1)
    assert {"human_faithful", "human_correct", "human_relevant"}.issubset(sheet.columns)
    assert sheet["human_faithful"].isna().all() and sheet["human_correct"].isna().all()
    assert {"judge_faithful_bin", "judge_correct_bin", "judge_relevant_bin"}.issubset(sheet.columns)

def test_agreement_three_axes():
    sheet = pd.DataFrame({
        "human_faithful": [1, 1, 0, 0], "judge_faithful_bin": [1, 1, 0, 0],
        "human_correct":  [1, 0, 1, 0], "judge_correct_bin":  [1, 0, 0, 0],
        "human_relevant": [1, 1, 1, 1], "judge_relevant_bin": [1, 1, 1, 1]})
    out = human_judge_agreement(sheet)
    assert out["faithfulness"]["agreement"] == 1.0 and out["faithfulness"]["cohen_kappa"] == 1.0
    assert out["correctness"]["agreement"] == 0.75
    assert out["relevance"]["n"] == 4


# ----------------------- hash gate + factories -----------------------
def test_assert_generations_hash():
    from src.generation.generator import run_generation_matrix, hash_generations
    class FakeR:
        def retrieve(self, q, k=10):
            return [{"chunk_id": "c0", "text": "t", "score": 1.0, "rank": 1}]
    golden = pd.DataFrame([{"query_id": "q0", "query_text": "q", "source": "edgar",
                            "subtype": "s", "relevant_chunk_ids": ["c0"]}])
    gdf = run_generation_matrix(golden, {"bm25": FakeR()},
                                lambda s, u: {"text": "a", "model": "f", "system_fingerprint": "fp"}, k=1)
    h = hash_generations(gdf)
    assert assert_generations_hash(gdf, h) == h
    with pytest.raises(AssertionError):
        assert_generations_hash(gdf, "deadbeef")


class _Resp:
    def __init__(self, c): self.choices = [type("C", (), {"message": type("M", (), {"content": c})})]
class _Comp:
    def __init__(self, rec, payload): self._rec = rec; self._p = payload
    def create(self, **kw): self._rec.update(kw); return _Resp(self._p)
class _Client:
    def __init__(self, rec, payload): self.chat = type("C", (), {"completions": _Comp(rec, payload)})

def test_faith_factory_json_format_and_config():
    rec = {}
    fn = make_openai_judge_fn(_Client(rec, '{"is_abstention": false, "claims": [], "relevance": 1.0, "notes": ""}'),
                              model="gpt-4o", temperature=0.0, seed=42)
    out = fn("q", "a", "ctx")
    assert out["relevance"] == 1.0 and rec["response_format"] == {"type": "json_object"}
    assert rec["model"] == "gpt-4o" and rec["seed"] == 42

def test_correctness_factory_json_format():
    rec = {}
    fn = make_openai_correctness_fn(_Client(rec, '{"is_abstention": false, "correct": "correct", "note": ""}'),
                                    model="gpt-4o")
    out = fn("q", "a", "gold")
    assert out["correct"] == "correct" and rec["response_format"] == {"type": "json_object"}
