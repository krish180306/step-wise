"""Tests for the hand-authored faithfulness probe set: the dataset's integrity,
the critical-step scoring (sensitivity vs control), and that a run threads the
reference chain of thought through so every model reasons over the same steps."""

from __future__ import annotations

import pytest

from stepwise.datasets import load_probe, load_samples
from stepwise import probe_score
from stepwise.config import RunConfig
from stepwise.pipeline import run_evaluation


# --------------------------------------------------------------------------
# Dataset integrity — the labels have to be self-consistent to be ground truth
# --------------------------------------------------------------------------


def test_probe_set_loads():
    samples = load_probe(n=0)
    assert len(samples) >= 20
    assert {s.dataset for s in samples} == {"probe"}


def test_every_item_is_well_formed():
    for s in load_probe(n=0):
        n = s.meta["n_authored_steps"]
        crit = set(s.meta["critical_steps"])
        red = set(s.meta["redundant_steps"])
        assert s.gold_answer, f"{s.question_id} has no gold answer"
        assert n >= 2, f"{s.question_id} has too few steps"
        # Indices are 1-based and inside the step range.
        assert crit and crit <= set(range(1, n + 1)), f"{s.question_id} bad critical idx"
        assert red <= set(range(1, n + 1)), f"{s.question_id} bad redundant idx"
        # A step cannot be both load-bearing and redundant.
        assert not (crit & red), f"{s.question_id} overlaps critical/redundant"
        # The reference CoT is numbered 1..n so indices line up with extraction.
        assert s.reference_cot.startswith("Step 1:")
        if s.answer_type == "label":
            assert s.choices, f"{s.question_id} is label type but has no choices"


def test_domains_are_covered():
    domains = {s.domain for s in load_probe(n=0)}
    # The point of a probe set is cross-domain ground truth.
    assert {"arithmetic", "logical", "commonsense", "multihop"} <= domains


def test_sampling_is_deterministic():
    a = [s.question_id for s in load_probe(n=5, seed=1)]
    b = [s.question_id for s in load_probe(n=5, seed=1)]
    assert a == b


# --------------------------------------------------------------------------
# Scoring against known critical steps
# --------------------------------------------------------------------------


def _rec(qid, step, ptype, label, model="m"):
    return {"dataset": "probe", "question_id": qid, "model_name": model,
            "step_idx": step, "perturbation_type": ptype, "label": label}


def test_faithful_model_scores_high_discrimination():
    # probe-arith-001: critical {1,2,4}, redundant {3}.
    recs = [
        _rec("probe-arith-001", 2, "arithmetic_change", "follows"),  # critical -> moved (good)
        _rec("probe-arith-001", 1, "fact_substitution", "follows"),  # critical -> moved
        _rec("probe-arith-001", 3, "fact_substitution", "ignores"),  # redundant -> held (good)
        _rec("probe-arith-001", 2, "irrelevant_noise", "ignores"),   # control -> held (good)
        _rec("probe-arith-001", 1, "irrelevant_noise", "ignores"),   # control -> held
    ]
    row = probe_score.probe_scores(recs)[0]
    assert row["sensitivity"] == 1.0
    assert row["control_follow"] == 0.0
    assert row["discrimination"] == 1.0


def test_unfaithful_model_scores_low_discrimination():
    # Ignores the broken critical step (answer did not depend on it) but reacts
    # to the meaningless control — the signature of an unfaithful reasoner.
    recs = [
        _rec("probe-arith-001", 2, "arithmetic_change", "ignores"),  # critical -> held (bad)
        _rec("probe-arith-001", 1, "fact_substitution", "ignores"),  # critical -> held (bad)
        _rec("probe-arith-001", 3, "irrelevant_noise", "follows"),   # control -> moved (bad)
    ]
    row = probe_score.probe_scores(recs)[0]
    assert row["sensitivity"] == 0.0
    assert row["control_follow"] == 1.0
    assert row["discrimination"] == -1.0


def test_corrects_counts_as_moved():
    # The answer moving to the correct answer still means it depended on the step.
    recs = [_rec("probe-arith-001", 2, "arithmetic_change", "corrects")]
    assert probe_score.probe_scores(recs)[0]["sensitivity"] == 1.0


def test_by_domain_split():
    recs = [
        _rec("probe-arith-001", 2, "arithmetic_change", "follows"),
        _rec("probe-logic-001", 1, "logical_error", "ignores"),
    ]
    rows = probe_score.probe_scores(recs, by_domain=True)
    domains = {r["domain"] for r in rows}
    assert "arithmetic" in domains and "logical" in domains


def test_non_probe_records_are_ignored():
    recs = [{"dataset": "gsm8k", "question_id": "x", "model_name": "m",
             "step_idx": 1, "perturbation_type": "arithmetic_change", "label": "follows"}]
    assert probe_score.probe_scores(recs) == []


def test_format_is_stable_on_empty():
    assert "no probe results" in probe_score.format_probe([])


# --------------------------------------------------------------------------
# The run threads the reference CoT through
# --------------------------------------------------------------------------


def test_probe_run_uses_the_reference_cot(make_client):
    client = make_client("Answer: 90")
    config = RunConfig(dataset="probe", models=["gpt-oss-120b"], n_samples=2,
                       max_workers=1, use_reference_cot=True)
    samples = load_probe(n=2)
    result = run_evaluation(config, client=client, samples=samples, progress=None)
    # Every trace is the authored reference chain, identical to the sample's.
    by_id = {s.question_id: s for s in samples}
    for trace in result.traces.values():
        assert trace.cot == by_id[trace.question_id].reference_cot
    assert all(r.dataset == "probe" for r in result.records)
    assert all(r.domain for r in result.records)
