"""Tests for the faithfulness benchmark apparatus: structural validation, the
frozen manifest, tiered scoring, and the independent-annotator check."""

from __future__ import annotations

import copy

import pytest

from stepwise import benchmark as B
from stepwise import probe_score


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_the_shipped_benchmark_is_valid():
    assert B.validate_items() == []


def test_validation_catches_overlapping_labels():
    item = copy.deepcopy(B.load_items()[0])
    item["critical_steps"] = [1, 2]
    item["redundant_steps"] = [2, 3]  # 2 is in both
    issues = B.validate_items([item])
    assert any("both critical and redundant" in i for i in issues)


def test_validation_catches_distractor_not_in_redundant():
    item = copy.deepcopy(B.load_items()[0])
    item["distractor_steps"] = [1]  # 1 is critical, not redundant
    issues = B.validate_items([item])
    assert any("distractor" in i for i in issues)


def test_validation_requires_adversarial_distractor():
    item = copy.deepcopy(B.load_items()[0])
    item["tier"] = "adversarial"
    item["distractor_steps"] = []
    issues = B.validate_items([item])
    assert any("no distractor" in i for i in issues)


def test_validation_requires_full_step_coverage():
    item = copy.deepcopy(B.load_items()[0])
    # Drop a step from the labels so not every step is accounted for.
    item["critical_steps"] = [1]
    item["redundant_steps"] = []
    issues = B.validate_items([item])
    assert any("every step must be critical or redundant" in i for i in issues)


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def test_manifest_has_version_counts_and_checksum():
    m = B.build_manifest()
    assert m["version"] == B.BENCHMARK_VERSION
    assert m["n_items"] >= 50
    assert len(m["checksum_sha256"]) == 64
    assert m["valid"] is True
    assert set(m["by_tier"]) <= set(B.TIERS)


def test_checksum_is_deterministic():
    assert B.build_manifest()["checksum_sha256"] == B.build_manifest()["checksum_sha256"]


def test_fallback_coverage_is_reported():
    cov = B.fallback_coverage()
    assert 0 <= cov["coverage"] <= 1
    assert cov["fallback_perturbable"] <= cov["critical_steps"]


def test_write_manifest(tmp_path):
    p = B.write_manifest(tmp_path / "m.json")
    import json
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["name"] == B.NAME


# --------------------------------------------------------------------------
# Tiered scoring + distractor metric
# --------------------------------------------------------------------------


def _rec(qid, step, ptype, label):
    return {"dataset": "probe", "question_id": qid, "model_name": "m",
            "step_idx": step, "perturbation_type": ptype, "label": label}


def test_distractor_follow_is_measured():
    # probe-arith-010 is adversarial with distractor step 3.
    recs = [
        _rec("probe-arith-010", 2, "arithmetic_change", "follows"),  # critical -> moved
        _rec("probe-arith-010", 3, "logical_error", "follows"),      # distractor -> moved (bad)
    ]
    row = probe_score.probe_scores(recs)[0]
    assert row["sensitivity"] == 1.0
    assert row["distractor_follow"] == 1.0  # fooled by the look-alike step


def test_by_tier_breakdown():
    recs = [
        _rec("probe-arith-001", 2, "arithmetic_change", "follows"),  # easy
        _rec("probe-arith-010", 2, "arithmetic_change", "ignores"),  # adversarial
    ]
    rows = probe_score.probe_scores(recs, by_tier=True)
    tiers = {r["tier"] for r in rows}
    assert "easy" in tiers and "adversarial" in tiers


def test_benchmark_report_has_tiers():
    recs = [_rec("probe-arith-001", 2, "arithmetic_change", "follows")]
    report = B.benchmark_report(recs)
    assert report["version"] == B.BENCHMARK_VERSION
    assert "by_tier" in report and "by_domain" in report


# --------------------------------------------------------------------------
# Independent annotator check
# --------------------------------------------------------------------------


def test_llm_verify_agreement(make_client):
    # Model picks step 2; probe-arith-001's critical steps include 2 -> agreement.
    client = make_client("STEP: 2")
    v = B.llm_verify(B.load_items()[:1], client, model="gpt-oss-120b", progress=None)
    assert v["n"] == 1
    assert v["agreement"] == 1.0


def test_llm_verify_flags_distractor_pick(make_client):
    # Force the model to pick the distractor step of an adversarial item.
    advers = next(it for it in B.load_items() if it.get("distractor_steps"))
    pick = advers["distractor_steps"][0]
    client = make_client(f"STEP: {pick}")
    v = B.llm_verify([advers], client, model="gpt-oss-120b", progress=None)
    assert v["distractor_confusion"] == 1.0
    assert v["agreement"] == 0.0
