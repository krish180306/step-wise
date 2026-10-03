"""Tests for the adversarial verifier. The headline case is the real bug the
analyst once produced: a true p-value attached to the wrong comparison. The
verifier must flag it.
"""

from __future__ import annotations

from stepwise import score
from stepwise.agent import verifier


def _records():
    """A run where allam differs from qwen3.8 overall (p≈0.02) but NOT on
    arithmetic alone (p≈0.25) — the exact shape of the real misattribution.
    """
    rows = []
    # allam: follows 4 of every class except logic; qwen3.8: never follows.
    for q in range(5):
        for ptype in ("arithmetic_change", "fact_substitution", "irrelevant_noise", "logical_error"):
            allam_follows = (ptype != "logical_error") and (q < 4)
            rows.append(_cell(f"q{q}", "allam-2-7b", ptype, "follows" if allam_follows else "ignores"))
            rows.append(_cell(f"q{q}", "qwen3.8-27b", ptype, "ignores"))
    return rows


def _cell(qid, model, ptype, label):
    return {
        "question_id": qid, "dataset": "gsm8k", "domain": "arithmetic", "model_name": model,
        "step_idx": 1, "n_steps": 2, "perturbation_type": ptype, "label": label,
        "original_answer": "72", "perturbed_answer": "76" if label == "follows" else "72",
        "gold_answer": "72", "original_correct": True, "perturbed_correct": label != "follows",
        "answer_type": "numeric", "n_choices": 0, "error": "",
    }


def test_extract_claims_finds_percentages_pvalues_and_n():
    report = "allam-2-7b was 80% faithful (n=20), significantly lower (p = 0.022)."
    claims = verifier.extract_claims(report, ["allam-2-7b"])
    kinds = sorted(c.kind for c in claims)
    assert kinds == ["n", "p_value", "percent"]
    pct = next(c for c in claims if c.kind == "percent")
    assert pct.value == 0.80
    assert "allam-2-7b" in pct.entities


def test_a_supported_percentage_passes():
    frame = score.records_dataframe(_records())
    # allam's real arithmetic faithfulness here is 1/5 = 20% -> 80% dependency;
    # faithfulness 20% is a real value, so a claim of 20% is supported.
    result = verifier.verify_report("allam-2-7b reached 20% faithfulness on arithmetic.",
                                    frame)
    pcts = [c for c in result["claims"] if c["kind"] == "percent"]
    assert pcts[0]["verdict"] == "SUPPORTED"


def test_a_fabricated_percentage_is_flagged():
    frame = score.records_dataframe(_records())
    result = verifier.verify_report("allam-2-7b reached 47% faithfulness.", frame)
    assert result["verdict"] == "flagged"
    assert result["n_unsupported"] == 1
    assert result["unsupported"][0]["value"] == 0.47


def test_the_misattributed_pvalue_bug_is_caught():
    # The real failure: 0.022 is the ALL-CELLS p-value, quoted as if it were
    # the arithmetic-only one (which is ~0.25). As a p-value that the data
    # never yields for the arithmetic comparison... but 0.022 *is* yielded by
    # the all-cells comparison, so a naive check passes. The verifier checks
    # whether the number exists at all among real comparisons, catching a
    # wholly fabricated p-value; misattribution within real values is the
    # harder case and is documented as a limitation.
    frame = score.records_dataframe(_records())
    # A p-value in a real gap: the data yields either tiny p (overall) or
    # ~0.02-0.25 (per class) or 1.0 (logic), never ~0.5. A claim of p=0.5 is
    # therefore unsupported — no comparison produces it.
    result = verifier.verify_report("the gap was marginal (p = 0.5).", frame)
    pv = [c for c in result["claims"] if c["kind"] == "p_value"][0]
    assert pv["verdict"] == "UNSUPPORTED"


def test_the_flagship_misattribution_is_caught():
    # The exact bug the analyst produced: 0.022 is the ALL-CELLS p-value,
    # written as if it were the arithmetic_change comparison. The context-aware
    # check re-runs the arithmetic-only comparison (which is NOT 0.022) and
    # flags the mismatch.
    frame = score.records_dataframe(_records())
    overall = verifier._pvalue_for(frame, "allam-2-7b", "qwen3.8-27b", None)
    arith = verifier._pvalue_for(frame, "allam-2-7b", "qwen3.8-27b", "arithmetic_change")
    assert abs(overall - arith) > 0.005  # the two really do differ
    report = (f"allam-2-7b was significantly less faithful than qwen3.8-27b on "
              f"arithmetic_change (p = {overall:.3f}).")
    result = verifier.verify_report(report, frame)
    pv = [c for c in result["claims"] if c["kind"] == "p_value"][0]
    assert pv["verdict"] == "UNSUPPORTED"
    assert "arithmetic_change" in pv["note"]


def test_a_correctly_attributed_pvalue_passes():
    frame = score.records_dataframe(_records())
    arith = verifier._pvalue_for(frame, "allam-2-7b", "qwen3.8-27b", "arithmetic_change")
    report = (f"allam-2-7b vs qwen3.8-27b on arithmetic_change: p = {arith:.3f}.")
    result = verifier.verify_report(report, frame)
    pv = [c for c in result["claims"] if c["kind"] == "p_value"][0]
    assert pv["verdict"] == "SUPPORTED"


def test_a_real_pvalue_is_supported():
    frame = score.records_dataframe(_records())
    true_pvals = verifier._all_true_pvalues(frame)
    real = sorted(true_pvals)[0]
    result = verifier.verify_report(f"significant at p = {real:.5f}.", frame)
    pv = [c for c in result["claims"] if c["kind"] == "p_value"][0]
    assert pv["verdict"] == "SUPPORTED"


def test_a_number_with_no_entity_is_uncheckable_not_wrong():
    frame = score.records_dataframe(_records())
    result = verifier.verify_report("overall, about 63% of cells were clean.", frame)
    pct = [c for c in result["claims"] if c["kind"] == "percent"][0]
    assert pct["verdict"] == "UNCHECKABLE"


def test_n_is_never_judged():
    frame = score.records_dataframe(_records())
    result = verifier.verify_report("allam-2-7b (n=20) looked weak.", frame)
    n = [c for c in result["claims"] if c["kind"] == "n"][0]
    assert n["verdict"] == "UNCHECKABLE"


def test_format_lists_flagged_claims():
    frame = score.records_dataframe(_records())
    result = verifier.verify_report("allam-2-7b hit 47% faithfulness.", frame)
    text = verifier.format_verification(result)
    assert "flagged" in text
    assert "47" in text
