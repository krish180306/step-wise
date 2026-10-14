"""Tests for the ablation arm: delete a step instead of corrupting it.

Three things have to hold. The deletion must be invisible as a deletion (no
numbering gap for a model to notice), it must never spend an API call, and its
cells must stay out of the corruption metrics — a ``follows`` under ablation
means "the step was necessary", not "the model followed an injected error".
"""

from __future__ import annotations

import pytest

from stepwise import perturbations as P
from stepwise import score
from stepwise.config import ModelSpec, RunConfig
from stepwise.datasets import Sample
from stepwise.extractor import ablate_step, extract_steps
from stepwise.llm import LLMResponse
from stepwise.pipeline import _build_perturbed_cot, iter_cells, run_sample

from .conftest import MARKED_COT, NUMBERED_COT, PARAGRAPH_COT, PROSE_COT


# --------------------------------------------------------------------------
# The structural edit
# --------------------------------------------------------------------------


def test_render_without_drop_is_still_lossless(any_cot):
    # The ablation parameter must not disturb the invariant everything rests on.
    assert extract_steps(any_cot).render() == any_cot


def test_dropping_a_step_removes_exactly_that_step():
    extracted = extract_steps(MARKED_COT)
    out = extracted.render(drop={2})
    assert "48 / 2 = 24" not in out
    assert "Natalia sold 48 clips in April" in out
    assert "Altogether she sold 48 + 24 = 72 clips" in out


def test_step_markers_are_renumbered_so_there_is_no_gap():
    # "Step 1, Step 3" would tell the model something was taken out.
    out = extract_steps(MARKED_COT).render(drop={2})
    assert "Step 1:" in out and "Step 2:" in out
    assert "Step 3:" not in out


def test_bare_number_markers_are_renumbered_too():
    out = extract_steps(NUMBERED_COT).render(drop={1})
    assert out.startswith("1. ")
    assert "3." not in out


def test_dropping_works_for_every_cot_shape(any_cot):
    extracted = extract_steps(any_cot)
    if len(extracted) < 2:
        pytest.skip("needs at least two steps to drop one")
    out = extracted.render(drop={1})
    assert out != any_cot
    assert len(extract_steps(out)) == len(extracted) - 1


def test_dropping_an_unknown_step_raises():
    with pytest.raises(IndexError):
        extract_steps(MARKED_COT).render(drop={99})


def test_ablate_step_helper_matches_render():
    assert ablate_step(PROSE_COT, 2) == extract_steps(PROSE_COT).render(drop={2})


def test_paragraph_shape_keeps_its_spacing():
    out = ablate_step(PARAGRAPH_COT, 2)
    assert "6 litres per minute" not in out
    assert "the tank starts full" in out


# --------------------------------------------------------------------------
# The intervention itself
# --------------------------------------------------------------------------


def test_deletion_is_not_one_of_the_four_corruption_classes():
    # Adding it there would change every default run's cost and contents, and
    # the mechanistic track iterates that tuple.
    assert P.STEP_DELETION not in P.PERTURBATION_TYPES
    assert P.STEP_DELETION in P.ABLATION_TYPES
    assert P.STEP_DELETION in P.ALL_INTERVENTION_TYPES
    assert len(P.PERTURBATION_TYPES) == 4


def test_deletion_spends_no_model_call(make_client):
    client = make_client("should never be asked for this")
    result = P.perturb_step("So the total is 72.", P.STEP_DELETION, client=client)
    assert result.text == ""
    assert result.method == "deletion"
    assert result.changed is True
    assert client.fake.calls == []  # no writer call at all


def test_deletion_ignores_severity():
    for severity in P.SEVERITIES:
        assert P.perturb_step("A step.", P.STEP_DELETION, severity=severity).text == ""


def test_deletion_of_blank_text_is_marked_unchanged():
    assert P.ablate("   ").method == "unchanged"


def test_deletion_is_applicable_to_any_real_step():
    assert P.is_applicable(P.STEP_DELETION, "anything at all")
    assert not P.is_applicable(P.STEP_DELETION, "   ")


def test_is_ablation_discriminates():
    assert P.is_ablation(P.STEP_DELETION)
    assert not P.is_ablation(P.ARITHMETIC_CHANGE)


def test_unknown_class_still_raises_and_names_every_option():
    with pytest.raises(ValueError, match="step_deletion"):
        P.perturb_step("A step.", "not_a_class")


# --------------------------------------------------------------------------
# Pipeline wiring
# --------------------------------------------------------------------------


def test_build_perturbed_cot_drops_the_step_when_ablated():
    extracted = extract_steps(MARKED_COT)
    out = _build_perturbed_cot(extracted, 2, "", "full", ablated=True)
    assert "48 / 2 = 24" not in out
    assert "Step 2: Altogether" in out


def test_continue_mode_truncates_before_the_removed_step():
    extracted = extract_steps(MARKED_COT)
    out = _build_perturbed_cot(extracted, 2, "", "continue", ablated=True)
    assert "Natalia sold 48 clips" in out
    assert "48 / 2 = 24" not in out
    assert "Altogether" not in out


def test_a_single_step_chain_gets_no_ablation_cell():
    # Deleting the only step leaves no reasoning, which is the battery's
    # early-answering question rather than this one.
    extracted = extract_steps("Step 1: The only step in this whole chain.")
    cells = iter_cells(extracted, [P.STEP_DELETION, P.ARITHMETIC_CHANGE], 8)
    assert all(ptype != P.STEP_DELETION for _, ptype in cells)


def test_multi_step_chain_does_get_ablation_cells():
    extracted = extract_steps(MARKED_COT)
    cells = iter_cells(extracted, [P.STEP_DELETION], 8)
    assert len(cells) == len(extracted)


def _scripted_client(make_client):
    """A client that answers with the last number in the reasoning it is shown.

    That makes the fake target model maximally dependent on the trace, so a
    deleted step provably changes the answer and the labels are not an
    accident of the stub.
    """
    import re

    def respond(messages):
        system, user = messages[0]["content"], messages[-1]["content"]
        if "numbered steps" in system:
            return MARKED_COT + "\nAnswer: 72"
        if "already been written" in system or "first part of a chain" in system:
            numbers = re.findall(r"\d+", user.split("Reasoning:")[-1])
            return "Answer: " + (numbers[-1] if numbers else "0")
        return "a corrupted step mentioning 99"

    return make_client(respond)


def test_run_sample_records_the_ablation_arm_honestly(make_client):
    client = _scripted_client(make_client)
    sample = Sample(question_id="q1", dataset="gsm8k", question="Natalia sold 48 clips...",
                    gold_answer="72", answer_type="numeric")
    config = RunConfig(dataset="gsm8k", models=["t"], n_samples=1,
                       perturbation_types=["arithmetic_change", P.STEP_DELETION])
    spec = ModelSpec(name="t", provider="groq", api_id="t")
    _, records = run_sample(sample, spec, client, config)

    ablations = [r for r in records if r.perturbation_type == P.STEP_DELETION]
    assert ablations, "no ablation cells were produced"
    for record in ablations:
        # No writer means no writer to name, and no self-perturbation confound.
        assert record.perturbation_model == ""
        assert record.self_perturbed is False
        assert record.severity == "n/a"
        assert record.label  # it ran and got a label


def test_ablation_cells_do_not_multiply_across_severities(make_client):
    client = _scripted_client(make_client)
    sample = Sample(question_id="q1", dataset="gsm8k", question="Natalia sold 48 clips...",
                    gold_answer="72", answer_type="numeric")
    config = RunConfig(dataset="gsm8k", models=["t"], n_samples=1,
                       perturbation_types=[P.STEP_DELETION],
                       severities=["minimal", "moderate", "severe"])
    spec = ModelSpec(name="t", provider="groq", api_id="t")
    _, records = run_sample(sample, spec, client, config)
    # One cell per step, not three: a deletion has no size.
    assert len(records) == len(extract_steps(MARKED_COT))


# --------------------------------------------------------------------------
# Scoring: the two arms must not be pooled
# --------------------------------------------------------------------------


def _rec(step, ptype, label, *, qid="q1", n_steps=3):
    return {
        "question_id": qid, "dataset": "gsm8k", "model_name": "m",
        "step_idx": step, "perturbation_type": ptype, "label": label,
        "original_answer": "72", "perturbed_answer": "72" if label == "ignores" else "99",
        "gold_answer": "72", "original_correct": True,
        "perturbed_correct": label != "follows",
        "n_steps": n_steps, "is_answer_step": step >= n_steps,
        "original_step": "She sold 48 clips in April.",
        "perturbed_step": "She sold 55 clips in April.",
        "answer_type": "numeric", "n_choices": 0, "domain": "arithmetic", "error": "",
    }


def _two_arm_records():
    return [
        _rec(1, "arithmetic_change", "ignores"),
        _rec(1, P.STEP_DELETION, "follows"),          # step 1 was necessary
        _rec(2, "arithmetic_change", "follows", qid="q2"),
        _rec(2, P.STEP_DELETION, "ignores", qid="q2"),  # step 2 was redundant
    ]


def test_corruption_tables_exclude_the_ablation_arm():
    table = score.by_model(_two_arm_records())
    # Two corruption cells only; the two deletions are scored apart.
    assert int(table.iloc[0]["n"]) == 2
    assert table.iloc[0]["dependency"] == 0.5


def test_by_perturbation_still_shows_ablation_as_its_own_row():
    table = score.by_perturbation(_two_arm_records(), include_ablation=True)
    assert P.STEP_DELETION in set(table["perturbation_type"])


def test_necessity_report_reads_the_ablation_arm():
    report = score.necessity_report(_two_arm_records())
    assert report["n_ablation_cells"] == 2
    assert report["necessity_rate"] == 0.5


def test_necessity_cross_tab_pairs_the_two_arms_on_the_same_steps():
    report = score.necessity_report(_two_arm_records())
    tab = report["cross_tab"]
    assert report["n_paired"] == 2
    # Step 1: needed but an error in it did not drag the answer along.
    assert tab["necessary_not_followed"] == 1
    # Step 2: not needed, yet the answer moved when it was corrupted.
    assert tab["redundant_but_followed"] == 1


def test_necessity_report_is_empty_without_ablation_cells():
    report = score.necessity_report([_rec(1, "arithmetic_change", "ignores")])
    assert report["n_ablation_cells"] == 0
    assert report["per_model"] == []


def test_necessity_report_is_empty_on_no_records():
    assert score.necessity_report([])["n_ablation_cells"] == 0


def test_multiplicity_report_ignores_ablation_cells():
    rows = score.multiplicity_report(_two_arm_records())["rows"]
    assert all(r["perturbation_type"] != P.STEP_DELETION for r in rows)


def test_summary_includes_the_necessity_block():
    payload = score.summary(_two_arm_records())
    assert payload["necessity"]["n_ablation_cells"] == 2
