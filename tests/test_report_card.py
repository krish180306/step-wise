"""Tests for the HTML report card: it builds from a run's records and sidecars,
renders valid standalone HTML, and degrades gracefully when pieces are missing.
"""

from __future__ import annotations

from stepwise import report_card, score
from stepwise import demo


def _write_run(tmp_path, name="gsm8k"):
    result = demo.run_demo(dataset="gsm8k", n_samples=3, progress=None)
    path = tmp_path / f"{name}.jsonl"
    result.save(path)
    return tmp_path


def test_build_report_has_models_and_leaderboard(tmp_path):
    rd = _write_run(tmp_path)
    report = report_card.build_report("gsm8k", results_dir=rd)
    assert report["dataset"] == "gsm8k"
    assert report["per_model"]  # at least one model card
    assert "domain_leaderboard" in report
    # Simulated demo must be marked, so a card can never masquerade as real.
    assert report["simulated"] is True


def test_render_html_is_selfcontained(tmp_path):
    rd = _write_run(tmp_path)
    html = report_card.render_html(report_card.build_report("gsm8k", results_dir=rd))
    assert html.startswith("<!doctype html>")
    assert "<style>" in html and "faithfulness report card" in html
    # The simulated banner must show for a simulated run.
    assert "SIMULATED" in html


def test_write_report_creates_file(tmp_path):
    rd = _write_run(tmp_path)
    out = tmp_path / "card.html"
    # Build against the temp run dir, then write explicitly.
    report = report_card.build_report("gsm8k", results_dir=rd)
    out.write_text(report_card.render_html(report), encoding="utf-8")
    assert out.exists() and out.stat().st_size > 500


def test_missing_sidecars_do_not_crash(tmp_path):
    rd = _write_run(tmp_path)
    # No battery/introspect/probe sidecars exist -> card still builds.
    report = report_card.build_report("gsm8k", results_dir=rd)
    m = report["per_model"][0]
    assert m["battery"] is None and m["introspect"] is None


def test_empty_run_yields_empty_card(tmp_path):
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    report = report_card.build_report("empty", results_dir=tmp_path)
    assert report["per_model"] == []
    html = report_card.render_html(report)
    assert "No scored records" in html


def test_card_carries_the_tautology_block(tmp_path):
    rd = _write_run(tmp_path)
    report = report_card.build_report("gsm8k", results_dir=rd)
    assert "tautology" in report
    html = report_card.render_html(report)
    assert "How much of that was the answer being restated" in html
    assert "the honest number" in html


def test_card_shows_the_ablation_section_when_the_run_has_one(tmp_path):
    # The demo now includes the ablation arm, so the section should appear.
    rd = _write_run(tmp_path)
    html = report_card.render_html(report_card.build_report("gsm8k", results_dir=rd))
    assert "Was the step needed at all" in html


def test_card_hides_the_ablation_section_when_there_is_none(tmp_path):
    import json

    from stepwise.config import ABLATION_TYPES

    rd = _write_run(tmp_path)
    path = rd / "gsm8k.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [r for r in rows if r.get("perturbation_type") not in ABLATION_TYPES]
    assert len(kept) < len(rows), "the demo should have had ablation cells to strip"
    path.write_text("\n".join(json.dumps(r) for r in kept) + "\n", encoding="utf-8")

    html = report_card.render_html(report_card.build_report("gsm8k", results_dir=rd))
    assert "Was the step needed at all" not in html


def test_card_shows_the_ablation_section_when_present():
    html = report_card.render_html({
        "dataset": "x", "generated_at": "now", "coverage": {}, "per_model": [],
        "domain_leaderboard": {}, "tautology": {},
        "necessity": {
            "n_ablation_cells": 4, "necessity_rate_reasoning": 0.5,
            "cross_tab": {"necessary_and_followed": 1, "necessary_not_followed": 1,
                          "redundant_but_followed": 1, "redundant_and_ignored": 1},
        },
    })
    assert "Was the step needed at all" in html
    assert "without blindly trusting it" in html


def test_empty_blocks_render_nothing():
    assert report_card._tautology_html({}) == ""
    assert report_card._tautology_html({"n_scored": 0}) == ""
    assert report_card._necessity_html({}) == ""
    assert report_card._necessity_html({"n_ablation_cells": 0}) == ""
