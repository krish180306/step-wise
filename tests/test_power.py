"""Tests for the sizing planner: design effect, ICC, and sample size.

The point of these is that the clustered answer is much larger than the naive
one. A run sized by the independent-observations formula is under-powered by
the design effect, which on this design is routinely 2-4x.
"""

from __future__ import annotations

import pytest

from stepwise import score, stats


# --------------------------------------------------------------------------
# design_effect
# --------------------------------------------------------------------------


def test_independent_cells_have_no_design_effect():
    assert stats.design_effect(8, 0.0) == 1.0


def test_design_effect_grows_with_cells_and_icc():
    assert stats.design_effect(8, 0.3) == pytest.approx(3.1)
    assert stats.design_effect(4, 0.3) == pytest.approx(1.9)
    # Perfectly correlated cells: m of them carry the information of one.
    assert stats.design_effect(8, 1.0) == pytest.approx(8.0)


def test_design_effect_clamps_nonsense_inputs():
    assert stats.design_effect(0, 0.5) == 1.0       # m < 1 floors at 1
    assert stats.design_effect(8, -1.0) == 1.0      # icc < 0 floors at 0
    assert stats.design_effect(8, 5.0) == pytest.approx(8.0)  # icc > 1 caps at 1


def test_a_single_cell_per_question_is_never_inflated():
    assert stats.design_effect(1, 0.9) == 1.0


# --------------------------------------------------------------------------
# observed_icc
# --------------------------------------------------------------------------


def _records(pattern):
    """One record per entry, grouped into questions of equal size."""
    rows = []
    for qid, labels in enumerate(pattern):
        for step, label in enumerate(labels, start=1):
            rows.append({
                "question_id": f"q{qid}", "dataset": "gsm8k", "model_name": "m",
                "step_idx": step, "perturbation_type": "arithmetic_change",
                "label": label, "original_answer": "72",
                "perturbed_answer": "99" if label == "follows" else "72",
                "gold_answer": "72", "original_correct": True,
                "perturbed_correct": label != "follows", "n_steps": 4,
                "is_answer_step": False, "answer_type": "numeric", "n_choices": 0,
                "error": "",
            })
    return rows


def test_icc_is_high_when_questions_agree_internally():
    # Each question is all-follows or all-ignores: brittleness is a property
    # of the question, so cells within one carry almost no new information.
    frame = score.records_dataframe(_records([
        ["follows"] * 4, ["ignores"] * 4, ["follows"] * 4, ["ignores"] * 4,
    ]))
    assert stats.observed_icc(frame) > 0.8


def test_icc_is_low_when_questions_are_internally_mixed():
    frame = score.records_dataframe(_records([
        ["follows", "ignores", "follows", "ignores"],
        ["ignores", "follows", "ignores", "follows"],
        ["follows", "ignores", "follows", "ignores"],
        ["ignores", "follows", "ignores", "follows"],
    ]))
    assert stats.observed_icc(frame) < 0.1


def test_icc_never_goes_negative():
    frame = score.records_dataframe(_records([
        ["follows", "ignores"], ["follows", "ignores"], ["follows", "ignores"],
    ]))
    assert stats.observed_icc(frame) >= 0.0


def test_icc_handles_too_little_data():
    assert stats.observed_icc(score.records_dataframe([])) == 0.0
    assert stats.observed_icc(score.records_dataframe(_records([["follows"]]))) == 0.0


# --------------------------------------------------------------------------
# required_questions
# --------------------------------------------------------------------------


def test_independent_case_matches_the_textbook_two_proportion_formula():
    # 40% vs 25% at alpha=.05, power=.80 is ~152 per arm.
    plan = stats.required_questions(0.40, 0.25, cells_per_question=8, icc=0.0)
    assert plan["cells_per_arm_independent"] == pytest.approx(152, abs=2)
    assert plan["design_effect"] == 1.0
    assert plan["cells_per_arm"] == plan["cells_per_arm_independent"]


def test_clustering_inflates_the_requirement_by_the_design_effect():
    independent = stats.required_questions(0.40, 0.25, cells_per_question=8, icc=0.0)
    clustered = stats.required_questions(0.40, 0.25, cells_per_question=8, icc=0.3)
    assert clustered["design_effect"] == pytest.approx(3.1)
    assert clustered["questions"] > independent["questions"]
    assert clustered["cells_per_arm"] == pytest.approx(
        independent["cells_per_arm"] * 3.1, rel=0.02
    )


def test_a_smaller_effect_needs_more_questions():
    big = stats.required_questions(0.60, 0.25)["questions"]
    small = stats.required_questions(0.30, 0.25)["questions"]
    assert small > big


def test_more_power_needs_more_questions():
    assert (stats.required_questions(0.40, 0.25, power=0.90)["questions"]
            > stats.required_questions(0.40, 0.25, power=0.80)["questions"])


def test_stricter_alpha_needs_more_questions():
    assert (stats.required_questions(0.40, 0.25, alpha=0.01)["questions"]
            > stats.required_questions(0.40, 0.25, alpha=0.05)["questions"])


def test_no_effect_is_reported_as_undetectable_rather_than_infinite():
    plan = stats.required_questions(0.3, 0.3)
    assert plan["detectable"] is False
    assert plan["questions"] is None


def test_untabulated_alpha_is_refused_rather_than_approximated():
    with pytest.raises(ValueError, match="alpha"):
        stats.required_questions(0.40, 0.25, alpha=0.037)
    with pytest.raises(ValueError, match="power"):
        stats.required_questions(0.40, 0.25, power=0.83)


def test_api_call_estimate_scales_with_the_grid():
    plan = stats.required_questions(0.40, 0.25, cells_per_question=8, icc=0.0)
    # One baseline per question plus a write and a re-run per cell.
    assert plan["api_calls_per_model"] == plan["questions"] * 17


# --------------------------------------------------------------------------
# precision_for
# --------------------------------------------------------------------------


def test_precision_improves_with_more_questions():
    small = stats.precision_for(10, rate=0.4)["half_width"]
    large = stats.precision_for(100, rate=0.4)["half_width"]
    assert large < small


def test_two_question_run_cannot_resolve_anything_useful():
    """The project's standing limitation, as a number."""
    result = stats.precision_for(2, rate=0.4, cells_per_question=8, icc=0.3)
    assert result["half_width"] > 0.30  # +/-30 points or worse


def test_fifty_questions_reaches_a_usable_interval():
    result = stats.precision_for(50, rate=0.4, cells_per_question=8, icc=0.3)
    assert result["half_width"] < 0.10


def test_precision_refuses_a_single_question():
    assert stats.precision_for(1)["half_width"] is None
    assert stats.precision_for(0)["half_width"] is None


def test_precision_ci_is_clamped_to_the_unit_interval():
    result = stats.precision_for(2, rate=0.05, cells_per_question=8, icc=0.9)
    low, high = result["ci"]
    assert low >= 0.0 and high <= 1.0


# --------------------------------------------------------------------------
# power_plan — the parameters read off a real run
# --------------------------------------------------------------------------


def test_power_plan_measures_the_design_from_the_data():
    frame = score.records_dataframe(_records([["follows"] * 4, ["ignores"] * 4,
                                              ["follows"] * 4, ["ignores"] * 4]))
    plan = stats.power_plan(frame, p1=0.40, p2=0.25)
    measured = plan["measured"]
    assert measured["questions"] == 4
    assert measured["cells_per_question"] == 4.0
    assert measured["icc"] > 0.8  # read off the data, not assumed
    # And the high ICC it measured feeds straight into the requirement.
    assert plan["target"]["icc"] == measured["icc"]
    assert plan["target"]["design_effect"] > 3.0


def test_power_plan_works_with_no_run_at_all():
    plan = stats.power_plan(None, p1=0.40, p2=0.25)
    assert plan["measured"] is None
    assert plan["current_precision"] is None
    assert plan["target"]["questions"] > 0


def test_power_plan_explicit_parameters_override_the_measured_ones():
    frame = score.records_dataframe(_records([["follows"] * 4, ["ignores"] * 4]))
    plan = stats.power_plan(frame, cells_per_question=8, icc=0.0)
    assert plan["target"]["design_effect"] == 1.0
    assert plan["target"]["cells_per_question"] == 8.0


def test_power_plan_ignores_unscored_cells():
    rows = _records([["follows"] * 4, ["ignores"] * 4])
    rows.append({**rows[0], "label": "", "error": "perturbation_unchanged"})
    plan = stats.power_plan(score.records_dataframe(rows))
    assert plan["measured"]["cells"] == 8
