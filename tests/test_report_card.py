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
