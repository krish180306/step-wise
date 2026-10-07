"""Tests for the A3 writer split and the cross-model transfer analysis.

Records are built as plain dicts (``records_dataframe`` accepts those), so the
tests pin the aggregation logic without running the pipeline."""

from __future__ import annotations

from stepwise import score


def _rec(qid, model, step, ptype, label, *, self_perturbed=False, n_steps=3,
         answer_type="numeric"):
    return {
        "question_id": qid, "dataset": "gsm8k", "model_name": model,
        "step_idx": step, "perturbation_type": ptype, "label": label,
        "original_answer": "72", "perturbed_answer": "72" if label == "ignores" else "99",
        "gold_answer": "72", "original_correct": True,
        "perturbed_correct": label in ("ignores", "corrects"),
        "n_steps": n_steps, "is_answer_step": step > n_steps,
        "answer_type": answer_type, "n_choices": 0,
        "self_perturbed": self_perturbed, "domain": "arithmetic", "error": "",
    }


# --------------------------------------------------------------------------
# A3: writer_effect
# --------------------------------------------------------------------------


def test_writer_effect_splits_self_vs_independent():
    recs = (
        # self-perturbed cells: the model follows a lot (confounded)
        [_rec("q1", "m", 1, "arithmetic_change", "follows", self_perturbed=True) for _ in range(3)]
        + [_rec("q2", "m", 1, "arithmetic_change", "ignores", self_perturbed=True)]
        # independent writer: the model mostly ignores (the trustworthy number)
        + [_rec("q3", "m", 1, "arithmetic_change", "ignores", self_perturbed=False) for _ in range(4)]
    )
    out = score.writer_effect(recs)
    assert set(out["writer"]) == {"self", "independent"}
    self_row = out[out["self_perturbed"]].iloc[0]
    indep_row = out[~out["self_perturbed"]].iloc[0]
    # Self-perturbation inflated dependency; the independent writer is lower.
    assert self_row["dependency"] > indep_row["dependency"]


def test_writer_effect_empty_when_no_column():
    assert score.writer_effect([]).empty


# --------------------------------------------------------------------------
# cross-model transfer
# --------------------------------------------------------------------------


def test_cross_model_transfer_pairs_models():
    # Two models, same cells. Model A follows cell (q1,1); B ignores it.
    recs = [
        _rec("q1", "A", 1, "arithmetic_change", "follows"),
        _rec("q1", "A", 2, "logical_error", "ignores"),
        _rec("q1", "B", 1, "arithmetic_change", "ignores"),
        _rec("q1", "B", 2, "logical_error", "ignores"),
    ]
    out = score.cross_model_transfer(recs)
    assert len(out) == 1
    row = out.iloc[0]
    assert {row["model_a"], row["model_b"]} == {"A", "B"}
    assert row["n_shared"] == 2
    # They agree on one of two cells.
    assert row["label_agreement"] == 0.5
    # A follows 1 cell, B follows 0 -> no overlap.
    assert row["follows_jaccard"] == 0.0


def test_cross_model_transfer_needs_two_models():
    recs = [_rec("q1", "A", 1, "arithmetic_change", "follows")]
    assert score.cross_model_transfer(recs).empty


def test_summary_includes_new_sections():
    recs = [
        _rec("q1", "A", 1, "arithmetic_change", "follows"),
        _rec("q1", "B", 1, "arithmetic_change", "ignores"),
    ]
    s = score.summary(recs)
    assert "writer_effect" in s and "cross_model" in s


# --------------------------------------------------------------------------
# Severity ladder + capability vs faithfulness
# --------------------------------------------------------------------------


def _srec(qid, model, step, ptype, label, severity, n_steps=3):
    r = _rec(qid, model, step, ptype, label, n_steps=n_steps)
    r["severity"] = severity
    return r


def test_severity_curve_splits_consequential_and_control():
    recs = [
        _srec("q1", "m", 1, "arithmetic_change", "follows", "severe"),
        _srec("q1", "m", 1, "arithmetic_change", "ignores", "minimal"),
        _srec("q1", "m", 1, "irrelevant_noise", "ignores", "severe"),
        _srec("q1", "m", 1, "irrelevant_noise", "ignores", "minimal"),
    ]
    out = score.severity_curve(recs)
    assert set(out["severity"]) == {"minimal", "severe"}
    severe = out[out["severity"] == "severe"].iloc[0]
    assert severe["consequential_follow"] == 1.0
    assert severe["control_follow"] == 0.0
    # Rows come back ordered minimal -> severe.
    assert list(out["severity_rank"]) == sorted(out["severity_rank"])


def test_severity_curve_empty_without_severity_variation():
    # One severity is fine (returns a single row per model), zero rows if empty.
    assert score.severity_curve([]).empty


def test_capability_vs_faithfulness_reports_per_model_and_corr():
    recs = []
    # model A: accurate, low dependency; model B: less accurate, higher dependency
    for i in range(4):
        recs.append({**_rec(f"q{i}", "A", 1, "arithmetic_change", "ignores"),
                     "original_correct": True})
        recs.append({**_rec(f"q{i}", "B", 1, "arithmetic_change", "follows"),
                     "original_correct": i < 2})
    cap = score.capability_vs_faithfulness(recs)
    assert cap["n_models"] == 2
    by = {r["model_name"]: r for r in cap["per_model"]}
    assert by["A"]["accuracy"] == 1.0
    assert by["B"]["dependency"] >= by["A"]["dependency"]


def test_summary_has_severity_and_capability():
    recs = [_srec("q1", "m", 1, "arithmetic_change", "follows", "moderate")]
    s = score.summary(recs)
    assert "severity_curve" in s and "capability" in s


# --------------------------------------------------------------------------
# Per-domain leaderboard
# --------------------------------------------------------------------------


def _drec(qid, model, domain, label, correct=True):
    r = _rec(qid, model, 1, "arithmetic_change", label)
    r["domain"] = domain
    r["original_correct"] = correct
    return r


def test_domain_leaderboard_flags_a_winner_per_domain():
    recs = []
    for i in range(4):
        recs.append(_drec(f"a{i}", "A", "arithmetic", "follows", True))   # A: faithful here
        recs.append(_drec(f"b{i}", "B", "arithmetic", "ignores", True))
        recs.append(_drec(f"la{i}", "A", "logical", "ignores", True))
        recs.append(_drec(f"lb{i}", "B", "logical", "follows", False))    # B: faithful but wrong
    lb = score.domain_leaderboard(recs)
    assert set(lb["domains"]) == {"arithmetic", "logical"}
    assert lb["winners"]["arithmetic"]["by_faithfulness"] == "A"
    assert lb["winners"]["logical"]["by_faithfulness"] == "B"
    # Accuracy and faithfulness can disagree: B is faithful on logical but A is accurate.
    assert lb["winners"]["logical"]["by_accuracy"] == "A"
    assert lb["model_best"]["A"]["by_faithfulness"] == "arithmetic"


def test_domain_leaderboard_empty_on_single_or_no_domain():
    assert score.domain_leaderboard([])["domains"] == []


# --------------------------------------------------------------------------
# Multiplicity correction + leaderboard significance
# --------------------------------------------------------------------------


def test_multiplicity_report_applies_fdr():
    recs = []
    for i in range(8):
        # fact class follows a lot; noise control does not -> should be a real effect
        recs.append(_rec(f"q{i}", "m", 1, "fact_substitution", "follows"))
        recs.append(_rec(f"q{i}", "m", 2, "irrelevant_noise", "ignores"))
    mp = score.multiplicity_report(recs)
    assert mp["n_tests"] >= 1
    row = next(r for r in mp["rows"] if r["perturbation_type"] == "fact_substitution")
    assert "q_value" in row and "significant" in row
    assert row["follows"] == 1.0 and row["control_follow"] == 0.0


def test_leaderboard_winner_significance_flag():
    # A follows every arithmetic cell, B none, over several questions -> the
    # winner's margin should be flagged significant.
    recs = []
    for i in range(8):
        recs.append(_drec(f"a{i}", "A", "arithmetic", "follows", True))
        recs.append(_drec(f"b{i}", "B", "arithmetic", "ignores", True))
    lb = score.domain_leaderboard(recs)
    margin = lb["winners"]["arithmetic"].get("margin")
    assert margin is not None
    assert margin["runner_up"] == "B"
    assert margin["significant"] is True
