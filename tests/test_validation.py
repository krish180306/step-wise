"""Tests for measurement validation: Cohen's kappa, the annotation store, the
three-way agreement, the LLM judge (mocked), and the label regression.
"""

from __future__ import annotations

import pytest

from stepwise import judge, validation
from stepwise.llm import DiskCache, LLMClient, LLMResponse


# --------------------------------------------------------------------------
# Cohen's kappa
# --------------------------------------------------------------------------


def test_perfect_agreement_is_one():
    labels = ["follows", "ignores", "corrects", "ignores"]
    assert validation.cohen_kappa(labels, labels)["kappa"] == 1.0


def test_chance_level_agreement_is_near_zero():
    # Two raters labelling independently at the same base rate -> kappa ~ 0.
    import random

    rng = random.Random(0)
    opts = ["follows", "ignores"]
    a = [rng.choice(opts) for _ in range(400)]
    b = [rng.choice(opts) for _ in range(400)]
    k = validation.cohen_kappa(a, b)["kappa"]
    assert abs(k) < 0.15


def test_kappa_corrects_for_chance_below_raw_agreement():
    # 90% of cells are "ignores" for both; they agree 90% of the time, but
    # almost all of that is chance, so kappa is much lower than 0.9.
    a = ["ignores"] * 90 + ["follows"] * 10
    b = ["ignores"] * 90 + ["ignores"] * 5 + ["follows"] * 5
    r = validation.cohen_kappa(a, b)
    assert r["observed_agreement"] > 0.9
    # Chance correction bites: kappa (≈0.64) is far below the 0.95 raw rate.
    assert r["kappa"] < r["observed_agreement"] - 0.25


def test_kappa_strength_labels():
    assert "perfect" in validation._kappa_strength(0.9)
    assert validation._kappa_strength(0.5) == "moderate"
    assert validation._kappa_strength(0.1) == "slight or none"


def test_mismatched_lengths_are_rejected():
    assert validation.cohen_kappa(["follows"], ["follows", "ignores"])["kappa"] is None


# --------------------------------------------------------------------------
# Annotation store
# --------------------------------------------------------------------------


def test_annotation_store_roundtrips(tmp_path):
    store = validation.AnnotationStore.load(tmp_path / "ann.json")
    store.set("q1|m|1|arithmetic_change", "follows")
    store.set("q1|m|2|arithmetic_change", "ignores")
    store.save()
    reloaded = validation.AnnotationStore.load(tmp_path / "ann.json")
    assert len(reloaded) == 2
    assert reloaded.labels["q1|m|1|arithmetic_change"] == "follows"


def test_annotation_store_clears_and_validates(tmp_path):
    store = validation.AnnotationStore.load(tmp_path / "ann.json")
    store.set("k", "follows")
    store.set("k", "")  # clear
    assert "k" not in store.labels
    with pytest.raises(ValueError):
        store.set("k", "banana")


def test_validate_against_human_computes_kappa():
    records = [
        {"question_id": "q1", "model_name": "m", "step_idx": 1,
         "perturbation_type": "arithmetic_change", "label": "follows"},
        {"question_id": "q1", "model_name": "m", "step_idx": 2,
         "perturbation_type": "arithmetic_change", "label": "ignores"},
    ]
    store = validation.AnnotationStore(path=None, labels={
        validation.cell_key(records[0]): "follows",   # agree
        validation.cell_key(records[1]): "follows",   # disagree
    })
    result = validation.validate_against_human(records, store)
    assert result["annotated"] == 2
    assert result["disagreements"] == 1


# --------------------------------------------------------------------------
# Three-way agreement
# --------------------------------------------------------------------------


def test_three_way_agreement_compares_each_pair():
    rows = [
        {"auto_label": "follows", "human_label": "follows", "judge_label": "ignores"},
        {"auto_label": "ignores", "human_label": "ignores", "judge_label": "ignores"},
        {"auto_label": "follows", "human_label": "follows", "judge_label": "follows"},
    ]
    out = validation.three_way_agreement(rows)
    assert "auto_vs_human" in out
    assert "auto_vs_judge" in out
    assert out["auto_vs_human"]["kappa"] == 1.0  # auto and human agree here


# --------------------------------------------------------------------------
# LLM judge (mocked)
# --------------------------------------------------------------------------


def _client(reply):
    from tests.conftest import FakeProvider

    provider = FakeProvider(reply)
    return LLMClient(cache=DiskCache(enabled=False),
                     providers={"groq": provider, "openai": provider, "anthropic": provider})


def test_judge_parses_its_label():
    cell = {"question": "q", "original_answer": "72", "perturbed_answer": "76",
            "gold_answer": "72", "original_steps": [], "perturbed_steps": []}
    v = judge.judge_cell(cell, _client("LABEL: follows — the error propagated to the answer"))
    assert v["judge_label"] == "follows"
    assert "error propagated" in v["judge_reason"]


def test_judge_handles_a_messy_reply():
    cell = {"question": "q", "original_answer": "72", "perturbed_answer": "72",
            "gold_answer": "72", "original_steps": [], "perturbed_steps": []}
    v = judge.judge_cell(cell, _client("I think this one IGNORES the change entirely."))
    assert v["judge_label"] == "ignores"


def test_judge_marks_a_changed_step():
    steps = [{"marker": "Step 1:", "text": "ok", "changed": False},
             {"marker": "Step 2:", "text": "bad", "changed": True}]
    rendered = judge._steps(steps)
    assert "<-- changed" in rendered
    assert rendered.count("<-- changed") == 1


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------


def _reg_records(n_questions=30):
    """A dataset where the answer step drives following and reasoning steps
    do not — so the regression should find a big answer_step effect.
    """
    rows = []
    for q in range(n_questions):
        for step, n_steps in [(1, 3), (2, 3), (3, 3)]:
            is_answer = step == n_steps
            # answer step follows ~80% of the time; reasoning steps ~10%.
            follows = (is_answer and q % 5 != 0) or (not is_answer and q % 10 == 0)
            rows.append({
                "question_id": f"q{q}", "dataset": "gsm8k", "model_name": "m",
                "step_idx": step, "n_steps": n_steps, "is_answer_step": is_answer,
                "step_position": (step - 1) / 2, "perturbation_type": "arithmetic_change",
                "label": "follows" if follows else "ignores",
                "original_answer": "72", "perturbed_answer": "76" if follows else "72",
                "gold_answer": "72", "original_correct": True, "perturbed_correct": not follows,
                "original_step": "some reasoning text here", "answer_type": "numeric", "error": "",
            })
    return rows


def test_regression_finds_the_answer_step_effect():
    result = validation.label_regression(_reg_records())
    assert "error" not in result, result.get("error")
    answer_step = next(t for t in result["terms"] if t["term"] == "answer_step")
    # Corrupting the answer step hugely raises the odds of following.
    assert answer_step["odds_ratio"] > 1.0
    assert answer_step["p_value"] < 0.05


def test_regression_refuses_too_little_data():
    result = validation.label_regression(_reg_records(n_questions=2))
    assert "error" in result


def test_regression_refuses_no_outcome_variation():
    rows = _reg_records()
    for r in rows:
        r["label"] = "ignores"
    assert "error" in validation.label_regression(rows)


def test_format_regression_renders():
    text = validation.format_regression(validation.label_regression(_reg_records()))
    assert "answer_step" in text
    assert "odds ratio" in text
