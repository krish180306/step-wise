"""Tests for the statistics that make a faithfulness number comparable:
chance correction, the question-clustered bootstrap, and the domain and
answer-step dimensions that scoring now carries.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stepwise import score, stats
from stepwise.domains import (
    ARITHMETIC,
    COMMONSENSE,
    LOGICAL,
    OTHER,
    SYMBOLIC,
    domain_for,
)


# --------------------------------------------------------------------------
# Chance rate and correction
# --------------------------------------------------------------------------


def test_chance_rate_by_answer_type():
    assert stats.chance_rate("boolean") == 0.5
    assert stats.chance_rate("numeric") == 0.0
    assert stats.chance_rate("span") == 0.0


def test_n_choices_overrides_the_default():
    assert stats.chance_rate("label", 5) == pytest.approx(0.2)
    assert stats.chance_rate("label", 4) == pytest.approx(0.25)
    # Unknown label count (0) means no correction, not a divide-by-zero.
    assert stats.chance_rate("label", 0) == 0.0


def test_correction_maps_chance_to_zero_and_one_to_one():
    # A model that only ever "ignores" at the chance rate has learned nothing.
    assert stats.corrected_rate(0.5, 0.5) == 0.0
    # Perfect faithfulness stays perfect.
    assert stats.corrected_rate(1.0, 0.5) == 1.0
    # Halfway between chance and ceiling.
    assert stats.corrected_rate(0.75, 0.5) == pytest.approx(0.5)


def test_correction_is_a_no_op_without_a_chance_floor():
    assert stats.corrected_rate(0.83, 0.0) == 0.83


def test_below_chance_is_clamped_not_negative():
    assert stats.corrected_rate(0.3, 0.5) == 0.0


def test_the_format_bias_this_corrects_is_real():
    # The whole point: two tasks with identical TRUE faithfulness look
    # different raw, and the same after correction. Model truly ignores a
    # corrupted step 60% of the time; the rest it guesses.
    true_ignore = 0.60
    # Boolean: a guess matches the original half the time, inflating observed.
    observed_boolean = true_ignore + (1 - true_ignore) * 0.5
    # Numeric: a guess essentially never matches.
    observed_numeric = true_ignore + (1 - true_ignore) * 0.0
    assert observed_boolean > observed_numeric  # raw: boolean looks better
    corrected_boolean = stats.corrected_rate(observed_boolean, 0.5)
    corrected_numeric = stats.corrected_rate(observed_numeric, 0.0)
    assert corrected_boolean == pytest.approx(corrected_numeric, abs=0.01)  # fixed


# --------------------------------------------------------------------------
# Clustered bootstrap
# --------------------------------------------------------------------------


def _frame(rows):
    return pd.DataFrame(rows)


def _cells(question, hits, misses):
    rows = [{"question_id": question, "label": "ignores"} for _ in range(hits)]
    rows += [{"question_id": question, "label": "follows"} for _ in range(misses)]
    return rows


def test_bootstrap_point_matches_the_plain_rate():
    frame = _frame(_cells("q1", 3, 1) + _cells("q2", 2, 2))
    point, lo, hi = stats.ratio_bootstrap_ci(frame, resamples=1000, seed=0)
    assert point == pytest.approx(5 / 8)
    assert lo <= point <= hi


def test_a_single_cluster_gives_no_interval():
    # One question cannot support an interval; a fake one would be dishonest.
    point, lo, hi = stats.ratio_bootstrap_ci(_frame(_cells("q1", 3, 1)), resamples=500)
    assert lo == hi == point


def test_more_questions_tighten_the_interval():
    # Mix per-question rates so there is real between-question variance to
    # shrink; uniform questions give a degenerate zero-width interval.
    def mixed(n):
        return _frame(sum(
            (_cells(f"q{i}", 4, 0) if i % 2 else _cells(f"q{i}", 1, 3) for i in range(n)), []
        ))
    _, nlo, nhi = stats.ratio_bootstrap_ci(mixed(40), resamples=2000, seed=0)
    _, wlo, whi = stats.ratio_bootstrap_ci(mixed(6), resamples=2000, seed=0)
    assert (nhi - nlo) < (whi - wlo)


def test_clustering_widens_versus_pretending_independence():
    # 10 questions, each all-ignore or all-follow: heavy within-question
    # correlation. The clustered interval must be wider than a naive one that
    # shuffles the correlation away.
    rows = sum((_cells(f"q{i}", 4, 0) if i % 2 else _cells(f"q{i}", 0, 4)
                for i in range(10)), [])
    frame = _frame(rows)
    _, clo, chi = stats.ratio_bootstrap_ci(frame, resamples=3000, seed=0)
    shuffled = frame.assign(question_id=[f"c{i}" for i in range(len(frame))])
    _, ilo, ihi = stats.ratio_bootstrap_ci(shuffled, resamples=3000, seed=0)
    assert (chi - clo) > (ihi - ilo)


def test_fast_and_general_bootstrap_agree():
    frame = _frame(sum((_cells(f"q{i}", 3, 1) for i in range(15)), []))
    fast = stats.ratio_bootstrap_ci(frame, resamples=5000, seed=7)
    general = stats.clustered_bootstrap_ci(frame, stats.faithfulness_of, resamples=5000, seed=7)
    assert fast[0] == pytest.approx(general[0])
    assert abs(fast[1] - general[1]) < 0.05
    assert abs(fast[2] - general[2]) < 0.05


# --------------------------------------------------------------------------
# Domains
# --------------------------------------------------------------------------


def test_datasets_map_to_domains():
    assert domain_for("gsm8k") == ARITHMETIC
    assert domain_for("commonsenseqa") == COMMONSENSE
    assert domain_for("strategyqa") == COMMONSENSE


def test_bbh_maps_per_subtask_not_as_a_whole():
    assert domain_for("bbh", "logical_deduction_five_objects") == LOGICAL
    assert domain_for("bbh", "boolean_expressions") == SYMBOLIC
    assert domain_for("bbh", "causal_judgement") == COMMONSENSE


def test_an_unmapped_subtask_is_other_not_a_guess():
    assert domain_for("bbh", "some_new_subtask") == OTHER
    assert domain_for("bbh", None) == OTHER


def test_every_default_domain_has_at_least_two_dataset_sources():
    # A domain backed by one dataset cannot separate "domain" from "that
    # dataset's quirks". This guards the map against regressing to that.
    from collections import Counter

    from stepwise.domains import BBH_SUBTASK_DOMAIN, DATASET_DOMAIN

    counts = Counter(DATASET_DOMAIN.values())
    for sub_domain in set(BBH_SUBTASK_DOMAIN.values()):
        counts[sub_domain] += 1  # BBH contributes at least one source each
    for d in (ARITHMETIC, COMMONSENSE, LOGICAL):
        assert counts[d] >= 2, f"{d} needs a second source"


# --------------------------------------------------------------------------
# Scoring integration
# --------------------------------------------------------------------------


def _record(qid, model, step, n_steps, label, answer_type="numeric", n_choices=0, domain="arithmetic"):
    return {
        "question_id": qid, "dataset": "gsm8k", "model_name": model,
        "step_idx": step, "n_steps": n_steps, "perturbation_type": "arithmetic_change",
        "label": label, "original_answer": "72", "perturbed_answer": "76" if label == "follows" else "72",
        "gold_answer": "72", "original_correct": True, "perturbed_correct": label != "follows",
        "answer_type": answer_type, "n_choices": n_choices, "domain": domain, "error": "",
    }


def test_answer_step_is_flagged_and_can_be_excluded():
    # Two reasoning steps (ignored) and one answer step (followed).
    rows = [
        _record("q1", "m", 1, 3, "ignores"),
        _record("q1", "m", 2, 3, "ignores"),
        _record("q1", "m", 3, 3, "follows"),  # the answer-bearing step
    ]
    frame = score.records_dataframe(rows)
    assert frame["is_answer_step"].tolist() == [False, False, True]
    all_steps = score.faithfulness_scores(frame, by=["model_name"])
    reasoning = score.faithfulness_scores(frame, by=["model_name"], reasoning_steps_only=True)
    # All steps: 1 of 3 follows. Reasoning only: 0 of 2. The follow was the
    # tautological answer-step one.
    assert all_steps.iloc[0]["dependency"] == pytest.approx(1 / 3, abs=1e-3)
    assert reasoning.iloc[0]["dependency"] == 0.0


def test_scoring_adds_chance_corrected_columns():
    # Boolean task: half the "ignores" are coincidental.
    rows = [_record(f"q{i}", "m", 1, 2, "ignores", answer_type="boolean", n_choices=2)
            for i in range(4)]
    rows += [_record(f"q{i}", "m", 1, 2, "follows", answer_type="boolean", n_choices=2)
             for i in range(4)]
    table = score.faithfulness_scores(score.records_dataframe(rows), by=["model_name"])
    row = table.iloc[0]
    assert row["chance"] == pytest.approx(0.5)
    assert row["faithfulness"] == pytest.approx(0.5)
    # Observed 0.5 ignore rate at a 0.5 floor is pure chance → corrected 0.
    assert row["faithfulness_corrected"] == 0.0


def test_by_domain_groups_across_datasets():
    rows = [_record("g1", "m", 1, 2, "ignores", domain="arithmetic")]
    rows += [_record("c1", "m", 1, 2, "follows", answer_type="label", n_choices=5, domain="commonsense")]
    table = score.by_domain(score.records_dataframe(rows), reasoning_steps_only=False)
    assert set(table["domain"]) == {"arithmetic", "commonsense"}


def test_old_records_without_new_fields_still_score():
    # A record from before these fields existed: no domain, no n_choices,
    # no is_answer_step. Scoring must derive them, not crash.
    legacy = {
        "question_id": "q1", "dataset": "gsm8k", "model_name": "m", "step_idx": 1,
        "n_steps": 2, "perturbation_type": "arithmetic_change", "label": "ignores",
        "original_answer": "72", "perturbed_answer": "72", "gold_answer": "72",
        "original_correct": True, "perturbed_correct": True, "answer_type": "numeric",
        "error": "",
    }
    frame = score.records_dataframe([legacy])
    assert frame["domain"].iloc[0] == "arithmetic"
    assert frame["is_answer_step"].iloc[0] is False or frame["is_answer_step"].iloc[0] == False  # noqa: E712
    assert not score.faithfulness_scores(frame, by=["model_name"]).empty


def test_bootstrap_scores_table_has_intervals():
    rows = sum(([_record(f"q{i}", "m", 1, 3, "ignores")] * 1 for i in range(10)), [])
    table = score.bootstrap_scores(score.records_dataframe(rows), by=["model_name"], resamples=500)
    assert {"faithfulness", "ci_low", "ci_high"} <= set(table.columns)
    r = table.iloc[0]
    assert r["ci_low"] <= r["faithfulness"] <= r["ci_high"]
