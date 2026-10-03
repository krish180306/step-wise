"""Unit tests for the dashboard API and the offline demo run.

The API is exercised through FastAPI's TestClient against a temporary results
directory, so nothing here depends on a real evaluation having been run.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from stepwise import demo
from stepwise.dashboard import app as dashboard_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A dashboard wired to a temp results dir holding one small demo run."""
    monkeypatch.setattr(dashboard_app, "RESULTS_DIR", tmp_path)
    dashboard_app._cached_payload.cache_clear()

    result = demo.run_demo(dataset="gsm8k", n_samples=2, progress=None)
    result.save(tmp_path / "demo.jsonl")
    return TestClient(dashboard_app.app)


# --------------------------------------------------------------------------
# The offline demo
# --------------------------------------------------------------------------


def test_the_demo_runs_without_a_key_or_network():
    result = demo.run_demo(dataset="gsm8k", n_samples=2, progress=None)
    assert result.records
    assert {r.model_name for r in result.records} == set(demo.PROFILES)


def test_the_demo_is_deterministic():
    first = demo.run_demo(dataset="gsm8k", n_samples=2, progress=None)
    second = demo.run_demo(dataset="gsm8k", n_samples=2, progress=None)
    key = lambda rs: sorted((r.question_id, r.model_name, r.step_idx,
                             r.perturbation_type, r.label) for r in rs)
    assert key(first.records) == key(second.records)


def test_the_demo_reproduces_its_model_profiles():
    """The simulated models must differ, or the demo grid teaches nothing."""
    from stepwise import score

    result = demo.run_demo(dataset="all", n_samples=3, progress=None)
    table = score.by_model(result.records).set_index("model_name")
    # sim-detached ignores its own reasoning; sim-faithful depends on it.
    assert table.loc["sim-detached", "faithfulness"] > table.loc["sim-faithful", "faithfulness"]


def test_the_demo_control_condition_is_mostly_ignored():
    from stepwise import score

    result = demo.run_demo(dataset="all", n_samples=3, progress=None)
    table = score.by_perturbation(result.records)
    noise = table[(table["perturbation_type"] == "irrelevant_noise")
                  & (table["model_name"] == "sim-faithful")]
    # A model whose reasoning is load-bearing should still shrug off an
    # irrelevant inserted sentence.
    assert noise.iloc[0]["faithfulness"] > 0.8


def test_a_pooled_demo_writes_a_demo_run():
    result = demo.run_demo(dataset="all", n_samples=2, progress=None)
    assert result.config.dataset == "demo"
    assert len({r.dataset for r in result.records}) > 1


def test_a_simulated_run_marks_itself_as_simulated(tmp_path):
    """Simulated output must never be mistakable for a measurement.

    The flag has to survive onto disk — into every record and into the
    sidecar — because that file outlives the terminal session that explained
    where it came from.
    """
    result = demo.run_demo(dataset="gsm8k", n_samples=2, progress=None)
    assert result.simulated is True

    path = result.save(tmp_path / "demo.jsonl")
    for line in path.read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["simulated"] is True

    sidecar = json.loads(path.with_suffix(".traces.json").read_text(encoding="utf-8"))
    assert sidecar["simulated"] is True
    assert "no model was queried" in sidecar["warning"]


def test_a_real_run_is_not_marked_simulated(make_client, tmp_path):
    """The marker must not fire on runs that did contact a provider."""
    from stepwise.config import RunConfig
    from stepwise.datasets import load_fixtures
    from stepwise.pipeline import run_evaluation

    client = make_client("Answer: 72")
    config = RunConfig(dataset="gsm8k", models=["gpt-oss-120b"], n_samples=1, max_workers=1)
    result = run_evaluation(
        config, client=client, samples=load_fixtures("gsm8k", n=1), progress=None
    )
    assert result.simulated is False

    path = result.save(tmp_path / "real.jsonl")
    sidecar = json.loads(path.with_suffix(".traces.json").read_text(encoding="utf-8"))
    assert sidecar["simulated"] is False
    assert sidecar["warning"] is None


def test_the_api_reports_a_run_as_simulated(client):
    payload = client.get("/api/summary", params={"run": "demo"}).json()
    assert payload["simulated"] is True
    assert "no model was queried" in payload["warning"]
    assert client.get("/api/runs").json()["runs"][0]["simulated"] is True


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


def test_health(client):
    assert client.get("/api/health").json()["ok"] is True


def test_runs_lists_the_demo(client):
    payload = client.get("/api/runs").json()
    assert [r["name"] for r in payload["runs"]] == ["demo"]
    assert payload["runs"][0]["records"] > 0
    assert payload["labels"] == ["follows", "ignores", "corrects"]


def test_summary_carries_everything_the_charts_need(client):
    payload = client.get("/api/summary", params={"run": "demo"}).json()
    assert payload["empty"] is False
    for key in ("coverage", "scores", "by_model", "by_perturbation", "outcomes", "heatmap"):
        assert key in payload
    assert set(payload["models"]) == set(demo.PROFILES)


def test_summary_heatmap_is_models_by_types(client):
    heatmap = client.get("/api/summary", params={"run": "demo"}).json()["heatmap"]
    assert len(heatmap["values"]) == len(heatmap["models"])
    assert all(len(row) == len(heatmap["types"]) for row in heatmap["values"])
    assert all(len(row) == len(heatmap["types"]) for row in heatmap["counts"])


def test_unknown_run_is_a_404(client):
    assert client.get("/api/summary", params={"run": "nope"}).status_code == 404


def test_a_run_name_cannot_escape_the_results_directory(client):
    response = client.get("/api/summary", params={"run": "../../etc/passwd"})
    assert response.status_code == 404


def test_grid_rows_are_question_by_step(client):
    payload = client.get("/api/grid", params={"run": "demo"}).json()
    assert payload["rows"]
    row = payload["rows"][0]
    assert {"question_id", "step_idx", "cells"} <= set(row)
    assert len(row["cells"]) == len(payload["types"])


def test_grid_filters_by_model(client):
    all_rows = client.get("/api/grid", params={"run": "demo"}).json()
    one = client.get("/api/grid", params={"run": "demo", "model": "sim-faithful"}).json()
    assert one["total_rows"] <= all_rows["total_rows"]
    models = {c["model"] for row in one["rows"] for c in row["cells"] if c["model"]}
    assert models == {"sim-faithful"}


def test_records_are_paged(client):
    first = client.get("/api/records", params={"run": "demo", "limit": 5}).json()
    second = client.get("/api/records", params={"run": "demo", "limit": 5, "offset": 5}).json()
    assert len(first["rows"]) == 5
    assert first["total"] == second["total"]
    assert first["rows"][0] != second["rows"][0]


def test_records_filter_by_label(client):
    payload = client.get("/api/records", params={"run": "demo", "label": "follows"}).json()
    assert all(r["label"] == "follows" for r in payload["rows"])


def test_records_limit_is_capped(client):
    payload = client.get("/api/records", params={"run": "demo", "limit": 9999}).json()
    assert payload["limit"] <= 200


def test_example_returns_both_traces_with_one_changed_step(client):
    row = client.get("/api/records", params={"run": "demo", "limit": 1}).json()["rows"][0]
    payload = client.get("/api/example", params={
        "run": "demo",
        "question_id": row["question_id"],
        "model": row["model_name"],
        "step_idx": row["step_idx"],
        "perturbation_type": row["perturbation_type"],
    }).json()

    changed = [s for s in payload["perturbed"]["steps"] if s["changed"]]
    assert len(changed) == 1
    assert changed[0]["was"] != changed[0]["text"]
    # Every other step must be identical to the original.
    originals = {s["index"]: s["text"] for s in payload["original"]["steps"]}
    for step in payload["perturbed"]["steps"]:
        if not step["changed"]:
            assert step["text"] == originals[step["index"]]


def test_example_for_a_missing_cell_is_a_404(client):
    response = client.get("/api/example", params={
        "run": "demo", "question_id": "nope", "model": "sim-mixed",
        "step_idx": 1, "perturbation_type": "logical_error",
    })
    assert response.status_code == 404


def test_analysis_reports_absence_without_erroring(client):
    payload = client.get("/api/analysis", params={"run": "demo"}).json()
    assert payload["exists"] is False


def test_analysis_is_served_once_written(client, tmp_path):
    (tmp_path / "analysis_demo.md").write_text("## Verdict\nAll good.", encoding="utf-8")
    (tmp_path / "analysis_demo.json").write_text(
        json.dumps({"model": "openai/gpt-oss-120b", "turns": 4,
                    "tool_calls": [{"tool": "describe_run"}]}),
        encoding="utf-8",
    )
    payload = client.get("/api/analysis", params={"run": "demo"}).json()
    assert payload["exists"] is True
    assert payload["report"].startswith("## Verdict")
    assert payload["turns"] == 4


def test_the_front_end_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Does the reasoning do any work?" in response.text


def test_summary_survives_a_group_with_no_data(client, tmp_path):
    # A model wrong on every baseline question has nothing to compute
    # "accuracy retained" over, so pandas reports NaN, which is not valid
    # JSON. Found on the first real run: /api/summary returned a 500 and the
    # dashboard rendered nothing at all.
    path = tmp_path / "demo.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["model_name"] == "sim-mixed":
            row["original_correct"] = False
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    dashboard_app._cached_payload.cache_clear()

    response = client.get("/api/summary", params={"run": "demo"})
    assert response.status_code == 200
    mixed = [r for r in response.json()["by_model"] if r["model_name"] == "sim-mixed"][0]
    assert mixed["accuracy_retained"] is None


def test_real_runs_are_listed_before_simulated_ones(client, make_client, tmp_path):
    # The dashboard opens on the first run listed. A simulated demo must never
    # be that run while real results exist, even when the demo is newer.
    import os
    import time

    from stepwise.config import RunConfig
    from stepwise.datasets import load_fixtures
    from stepwise.pipeline import run_evaluation

    config = RunConfig(dataset="gsm8k", models=["gpt-oss-120b"], n_samples=1, max_workers=1)
    real = run_evaluation(
        config, client=make_client("Answer: 72"), samples=load_fixtures("gsm8k", n=1), progress=None
    )
    real.save(tmp_path / "real.jsonl")
    later = time.time() + 60
    os.utime(tmp_path / "demo.jsonl", (later, later))  # the demo is now the newest file

    runs = client.get("/api/runs").json()["runs"]
    assert [r["name"] for r in runs] == ["real", "demo"]
    assert runs[0]["simulated"] is False
    assert runs[1]["simulated"] is True
