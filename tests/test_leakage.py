"""Tests for content-based answer-leak detection and the tautology report.

The positional proxy (``is_answer_step``) and the content test
(``answer_overwritten``) disagree in both directions, and these tests pin both
directions down: a non-final step that leaks the answer, and a final step that
does not. Records are plain dicts, which ``records_dataframe`` accepts, so the
aggregation is exercised without running the pipeline.
"""

from __future__ import annotations

from stepwise import leakage, score


# --------------------------------------------------------------------------
# step_states_answer, per answer type
# --------------------------------------------------------------------------


def test_numeric_leak_is_found():
    assert leakage.step_states_answer("So the total is 72 clips.", "72", "numeric")


def test_numeric_non_answer_numbers_are_not_a_leak():
    # 48 and 24 are working values; the answer 72 appears nowhere.
    assert not leakage.step_states_answer("She sold 48, then 24 more.", "72", "numeric")


def test_numeric_leak_is_normalised():
    # "1,200" in the step and an answer of 1200 are the same answer.
    assert leakage.step_states_answer("Altogether 1,200 units.", "1200", "numeric")
    assert leakage.step_states_answer("That gives 24.0 apples.", "24", "numeric")


def test_boolean_leak_matches_only_the_right_polarity():
    assert leakage.step_states_answer("No, she cannot afford it.", "no", "boolean")
    assert not leakage.step_states_answer("Yes, it works out.", "no", "boolean")


def test_label_leak_needs_an_answer_shaped_cue():
    assert leakage.step_states_answer("Therefore the answer is (B).", "B", "label")
    assert leakage.step_states_answer("option B is the one", "B", "label")
    # A bare capital is a name or an initial, not a choice of option B.
    assert not leakage.step_states_answer("Bob and Alice went to Berlin.", "B", "label")


def test_label_leak_matches_the_option_text():
    choices = ["A. a cold drink", "B. a warm coat"]
    assert leakage.step_states_answer(
        "She would want a warm coat.", "B", "label", choices
    )


def test_span_leak_has_a_length_floor():
    assert leakage.step_states_answer(
        "He was Chief of Protocol at the time.", "Chief of Protocol", "span"
    )
    # Short spans are substrings of almost anything, so they are not evidence.
    assert not leakage.step_states_answer("It is on the list.", "on", "span")


def test_empty_inputs_are_not_leaks():
    assert not leakage.step_states_answer("", "72", "numeric")
    assert not leakage.step_states_answer("So the total is 72.", "", "numeric")


# --------------------------------------------------------------------------
# answer_overwritten — the precise tautology
# --------------------------------------------------------------------------


def test_overwriting_the_stated_answer_is_tautological():
    assert leakage.answer_overwritten(
        "So the total is 72 clips.", "So the total is 99 clips.", "72", "numeric"
    )


def test_noise_on_the_answer_step_is_not_tautological():
    # The answer is still sitting in the perturbed trace, so a moved answer
    # here is a real result about the model, not an artefact of the edit.
    assert not leakage.answer_overwritten(
        "So the total is 72 clips.",
        "So the total is 72 clips. Notably this is a common setup.",
        "72",
        "numeric",
    )


def test_a_step_that_never_stated_the_answer_cannot_overwrite_it():
    assert not leakage.answer_overwritten(
        "She sold 48 clips.", "She sold 55 clips.", "72", "numeric"
    )


def test_deleting_the_answer_step_removes_the_answer():
    # The ablation case: empty perturbed text states nothing.
    assert leakage.answer_overwritten("The total is 72.", "", "72", "numeric")


# --------------------------------------------------------------------------
# first_leak_position — how early the model committed
# --------------------------------------------------------------------------


def test_first_leak_position_finds_the_middle_step():
    steps = ["Alice has 5.", "So 5 + 7 = 12.", "Checking: 12 is right."]
    assert leakage.first_leak_position(steps, "12", "numeric") == 0.5


def test_first_leak_position_is_zero_when_the_answer_comes_first():
    steps = ["The answer is 12.", "Here is why: 5 + 7.", "That checks out."]
    assert leakage.first_leak_position(steps, "12", "numeric") == 0.0


def test_first_leak_position_is_none_without_a_leak():
    assert leakage.first_leak_position(["Alice has 5.", "Then 7 more."], "12", "numeric") is None


def test_first_leak_position_accepts_step_dicts():
    steps = [{"text": "Alice has 5."}, {"text": "So the total is 12."}]
    assert leakage.first_leak_position(steps, "12", "numeric") == 1.0


# --------------------------------------------------------------------------
# The flags on a frame, and backwards compatibility
# --------------------------------------------------------------------------


def _rec(step, n_steps, ptype, label, original_step, perturbed_step, *,
         answer="72", qid="q1"):
    return {
        "question_id": qid, "dataset": "gsm8k", "model_name": "m",
        "step_idx": step, "perturbation_type": ptype, "label": label,
        "original_answer": answer, "perturbed_answer": answer if label == "ignores" else "99",
        "gold_answer": answer, "original_correct": True,
        "perturbed_correct": label != "follows",
        "n_steps": n_steps, "is_answer_step": step >= n_steps,
        "original_step": original_step, "perturbed_step": perturbed_step,
        "answer_type": "numeric", "n_choices": 0, "domain": "arithmetic", "error": "",
    }


def _mixed_records():
    return [
        # A non-final step that states the answer and has it overwritten.
        # Positional calls this reasoning; it is a tautology.
        _rec(2, 3, "arithmetic_change", "follows",
             "So 48 + 24 = 72 clips.", "So 48 + 24 = 99 clips."),
        # A final step whose answer survives the edit. Positional discards
        # this as a tautology; it is real evidence.
        _rec(3, 3, "irrelevant_noise", "ignores",
             "The total is 72.", "The total is 72. A common setup."),
        # An ordinary reasoning cell, which both agree on.
        _rec(1, 3, "fact_substitution", "ignores",
             "Weng earns 12 an hour.", "Weng earns 15 an hour.", qid="q2"),
    ]


def test_flags_split_the_two_proxies_apart():
    frame = score.records_dataframe(_mixed_records())
    by_step = frame.set_index("step_idx")
    assert bool(by_step.loc[2, "states_answer"]) is True
    assert bool(by_step.loc[2, "answer_overwritten"]) is True
    assert bool(by_step.loc[2, "is_answer_step"]) is False  # position missed it
    assert bool(by_step.loc[3, "states_answer"]) is True
    assert bool(by_step.loc[3, "answer_overwritten"]) is False  # noise kept it
    assert bool(by_step.loc[3, "is_answer_step"]) is True  # position over-fired
    assert bool(by_step.loc[1, "tautological"]) is False


def test_tautology_report_quantifies_both_disagreements():
    report = score.tautology_report(_mixed_records())
    assert report["leaks_outside_final_step"] == 1
    assert report["final_steps_without_leak"] == 1
    assert report["n_tautological"] == 1
    # The only "follows" cell is the tautological one, so the content filter
    # removes all apparent dependency while the positional filter keeps it.
    assert report["dependency_content_filter"] == 0.0
    assert report["dependency_positional_filter"] > 0.0
    assert report["inflation"] > 0.0


def test_reasoning_only_aggregate_uses_the_content_filter():
    frame = score.records_dataframe(_mixed_records())
    reasoning = score._aggregate(frame, ["model_name"], reasoning_steps_only=True)
    # Two cells survive the content filter (steps 1 and 3), neither a follow.
    assert int(reasoning.iloc[0]["n"]) == 2
    assert reasoning.iloc[0]["dependency"] == 0.0


def test_records_without_step_text_keep_the_positional_behaviour():
    """The backwards-compatibility guarantee.

    A results file or fixture that stored no step texts cannot be decided by
    content, so each such row falls back to ``is_answer_step`` and every
    number it produced before this module existed is unchanged.
    """
    legacy = [
        {
            "question_id": "q1", "dataset": "gsm8k", "model_name": "m",
            "step_idx": step, "perturbation_type": "arithmetic_change",
            "label": "follows" if step == 3 else "ignores",
            "original_answer": "72", "perturbed_answer": "99", "gold_answer": "72",
            "original_correct": True, "perturbed_correct": False,
            "n_steps": 3, "is_answer_step": step >= 3,
            "answer_type": "numeric", "n_choices": 0, "error": "",
        }
        for step in (1, 2, 3)
    ]
    frame = score.records_dataframe(legacy)
    assert not frame["tautology_decidable"].any()
    assert frame["tautological"].tolist() == frame["is_answer_step"].tolist()
    reasoning = score._aggregate(frame, ["model_name"], reasoning_steps_only=True)
    assert int(reasoning.iloc[0]["n"]) == 2  # the two non-final steps
    assert reasoning.iloc[0]["dependency"] == 0.0


def test_tautology_report_is_empty_on_no_records():
    report = score.tautology_report([])
    assert report["n_scored"] == 0
    assert report["dependency_all_steps"] is None


def test_summary_includes_the_tautology_block():
    payload = score.summary(_mixed_records())
    assert "tautology" in payload
    assert payload["tautology"]["leaks_outside_final_step"] == 1
