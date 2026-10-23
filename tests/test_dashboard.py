"""Unit tests for the dashboard API and the offline demo run.

The API is exercised through FastAPI's TestClient against a temporary results
directory, so nothing here depends on a real evaluation having been run.
"""

from __future__ import annotations

import json
import re

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


# --------------------------------------------------------------------------
# Human annotation endpoints (measurement validation)
# --------------------------------------------------------------------------


def test_annotation_save_and_reload(client, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_app, "RESULTS_DIR", tmp_path)
    # Grab a real cell key from the demo run.
    recs = client.get("/api/records", params={"run": "demo", "limit": 1}).json()["rows"][0]
    key = f"{recs['question_id']}|{recs['model_name']}|{recs['step_idx']}|{recs['perturbation_type']}"

    save = client.post("/api/annotate", json={"run": "demo", "key": key, "label": "follows"})
    assert save.status_code == 200 and save.json()["saved"] is True
    assert save.json()["count"] == 1

    got = client.get("/api/annotations", params={"run": "demo"}).json()
    assert got["labels"][key] == "follows"
    assert "kappa" in got["agreement"]


def test_annotation_clear(client, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_app, "RESULTS_DIR", tmp_path)
    client.post("/api/annotate", json={"run": "demo", "key": "k|m|1|x", "label": "ignores"})
    client.post("/api/annotate", json={"run": "demo", "key": "k|m|1|x", "label": ""})
    assert client.get("/api/annotations", params={"run": "demo"}).json()["count"] == 0


def test_a_bad_label_is_rejected(client, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_app, "RESULTS_DIR", tmp_path)
    r = client.post("/api/annotate", json={"run": "demo", "key": "k|m|1|x", "label": "banana"})
    assert r.status_code == 400


# --------------------------------------------------------------------------
# The analyses added on top of the original probe
# --------------------------------------------------------------------------


def test_tautology_endpoint_reports_both_filters(client):
    payload = client.get("/api/tautology?run=demo").json()
    assert payload["n_scored"] > 0
    # Both proxies are reported, so the gap between them is readable.
    assert payload["dependency_positional_filter"] is not None
    assert payload["dependency_content_filter"] is not None
    assert payload["per_model"]


def test_tautology_endpoint_finds_the_disagreements(client):
    payload = client.get("/api/tautology?run=demo").json()
    # The demo chains state their answer in a step that is not always last,
    # so the two proxies must disagree somewhere.
    assert (payload["leaks_outside_final_step"]
            + payload["final_steps_without_leak"]) > 0


def test_necessity_endpoint_reads_the_ablation_arm(client):
    # The demo includes step_deletion cells, so this panel has data offline.
    payload = client.get("/api/necessity?run=demo").json()
    assert payload["n_ablation_cells"] > 0
    assert payload["per_model"]
    assert payload["cross_tab"]


def test_necessity_endpoint_is_empty_without_the_ablation_arm(client, tmp_path):
    import json

    from stepwise.config import ABLATION_TYPES

    path = tmp_path / "demo.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [r for r in rows if r.get("perturbation_type") not in ABLATION_TYPES]
    (tmp_path / "noabl.jsonl").write_text(
        "\n".join(json.dumps(r) for r in kept) + "\n", encoding="utf-8")
    assert client.get("/api/necessity?run=noabl").json()["n_ablation_cells"] == 0


def test_power_endpoint_measures_the_design(client):
    payload = client.get("/api/power?run=demo").json()
    assert payload["measured"]["questions"] == 2
    assert payload["target"]["questions"] > 0
    assert payload["current_precision"]["half_width"] is not None


def test_power_endpoint_takes_the_effect_size(client):
    tight = client.get("/api/power?run=demo&p1=0.30&p2=0.25").json()
    loose = client.get("/api/power?run=demo&p1=0.60&p2=0.25").json()
    assert tight["target"]["questions"] > loose["target"]["questions"]


def test_power_endpoint_rejects_an_untabulated_level(client):
    # power_plan raises ValueError, which the route turns into a 400 rather
    # than a 500 traceback.
    import stepwise.stats as stats

    original = stats.power_plan
    try:
        stats.power_plan = lambda *a, **k: (_ for _ in ()).throw(ValueError("alpha=0.037"))
        assert client.get("/api/power?run=demo").status_code == 400
    finally:
        stats.power_plan = original


def test_runs_endpoint_lists_the_ablation_class(client):
    types = client.get("/api/runs").json()["perturbation_types"]
    assert "step_deletion" in types


def test_summary_carries_the_new_blocks(client):
    payload = client.get("/api/summary?run=demo").json()
    assert "tautology" in payload
    assert "necessity" in payload


def test_the_new_endpoints_404_on_an_unknown_run(client):
    for endpoint in ("tautology", "necessity", "power"):
        assert client.get(f"/api/{endpoint}?run=nope").status_code == 404


# --------------------------------------------------------------------------
# The sizing panel's markup and the contract its renderer depends on
# --------------------------------------------------------------------------


def _page() -> str:
    from pathlib import Path

    from stepwise.dashboard import app as mod

    return (Path(mod.__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


def test_the_page_has_a_sizing_panel():
    page = _page()
    assert 'id="power-block"' in page
    assert 'id="power-stats"' in page
    assert 'id="power-chart"' in page
    assert 'id="power-toggle"' in page
    assert "How big does the run need to be?" in page


def test_the_sizing_panel_is_rendered_on_refresh():
    # A panel nothing calls is an empty div; this is the wiring that fills it.
    page = _page()
    assert "renderPower();" in page
    assert "async function renderPower()" in page


def test_the_effect_size_chips_cover_three_sizes():
    page = _page()
    for p1 in ("0.60", "0.40", "0.30"):
        assert f'data-p1="{p1}"' in page
    # Exactly one starts pressed, so the panel has a defined initial state.
    assert page.count('data-p1="0.40" data-p2="0.25" aria-pressed="true"') == 1


def test_the_nav_points_at_the_probe_picker():
    page = _page()
    assert "choose your probes" in page             # Findings
    assert "all probes" in page                     # Diagnostics


def test_every_probe_in_the_registry_has_a_block_to_show():
    """A registry entry with no block silently renders an empty picker chip."""
    page = _page()
    ids = re.findall(r'\{ id:"(\w+)",\s+group:"(findings|methods)"', page)
    assert len(ids) >= 16, f"expected the full probe registry, found {len(ids)}"
    for probe_id, _group in ids:
        block = "probe-block" if probe_id == "probeset" else f"{probe_id}-block"
        assert f'id="{block}"' in page, f"{probe_id} has no block"
        assert f'data-probe="{probe_id}"' in page, f"{probe_id} is not tagged"


def test_the_picker_can_select_filter_and_persist():
    page = _page()
    for hook in ('data-act="all"', 'data-act="ran"', 'data-act="none"',
                 "stepwise.probes.v1", "function applyProbeChoice()",
                 "function probeRan(", 'id="triangulation"'):
        assert hook in page, hook


def test_availability_is_read_after_the_async_panels_land():
    """The picker decides "ran" from what each block drew, so every panel that
    fetches a sidecar has to be awaited before applyProbeChoice runs."""
    page = _page()
    body = page[page.index("async function refresh()"):]
    body = body[: body.index("\n}")]
    awaited = body[body.index("await Promise.all(["):]
    for renderer in ("renderNecessity()", "renderInteractions()", "renderBattery()",
                     "renderMech()", "renderPatch()", "renderIntrospect()",
                     "renderProbe()", "renderPower()"):
        assert renderer in awaited, f"{renderer} must be awaited before the picker syncs"
    assert body.index("await Promise.all([") < body.index("applyProbeChoice()")


def test_power_endpoint_supplies_every_field_the_panel_reads(client):
    """The renderer indexes these directly; a rename would blank the panel."""
    payload = client.get("/api/power?run=demo").json()
    assert {"measured", "current_precision", "target"} <= set(payload)
    target = payload["target"]
    for key in ("detectable", "p1", "p2", "alpha", "power", "icc", "design_effect",
                "cells_per_question", "cells_per_arm", "cells_per_arm_independent",
                "questions", "api_calls_per_model"):
        assert key in target, key
    for key in ("questions", "cells", "cells_per_question", "icc"):
        assert key in payload["measured"], key
    assert "half_width" in payload["current_precision"]


def test_power_endpoint_handles_an_undetectable_effect(client):
    payload = client.get("/api/power?run=demo&p1=0.3&p2=0.3").json()
    assert payload["target"]["detectable"] is False
    assert payload["target"]["questions"] is None


# --------------------------------------------------------------------------
# The pair-interaction arm's surface
# --------------------------------------------------------------------------


def test_interactions_endpoint_is_empty_without_a_sidecar(client):
    payload = client.get("/api/interactions?run=demo").json()
    assert payload["available"] is False


def test_interactions_endpoint_reads_its_sidecar(client, tmp_path):
    from stepwise import interactions as inter

    inter.save_interactions(
        [inter.InteractionRecord(
            question_id="q1", dataset="gsm8k", model_name="m", step_i=1, step_j=2,
            perturbation_type="arithmetic_change", original_answer="72",
            perturbed_answer="99", label="follows", single_label_i="ignores",
            single_label_j="ignores", interaction=inter.SYNERGISTIC)],
        tmp_path / "demo.interactions.jsonl",
    )
    payload = client.get("/api/interactions?run=demo").json()
    assert payload["available"] is True
    assert payload["classes"]["synergistic"] == 1
    assert payload["synergy_rate"] == 1.0
    assert payload["per_model"][0]["model_name"] == "m"


def test_the_page_has_an_interactions_panel():
    page = _page()
    assert 'id="interactions-table"' in page
    assert 'id="interactions-note"' in page
    assert "renderInteractions()" in page
    assert "async function renderInteractions()" in page


def test_the_interactions_hint_targets_this_run(_=None):
    # `interactions` reads the run file to find the single-step labels it
    # targets from, so its hint must point --output at that run.
    page = _page()
    assert 'cliCommand("interactions", { base: true })' in page
    assert "base = false" in page


def test_the_pair_sidecar_is_not_listed_as_a_run(client, tmp_path):
    """A sidecar matching *.jsonl must not appear as a phantom run."""
    (tmp_path / "demo.interactions.jsonl").write_text("{}\n", encoding="utf-8")
    names = [r["name"] for r in client.get("/api/runs").json()["runs"]]
    assert "demo.interactions" not in names


# --------------------------------------------------------------------------
# The offline demo now drives the battery and the self-knowledge probe too
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_run():
    return demo.run_demo(dataset="all", n_samples=3, progress=None)


def test_the_demo_drives_the_whole_battery(demo_run):
    """Every panel must be fillable with no key and no network."""
    from stepwise.battery_score import battery_summary

    records = demo.run_demo_battery(demo_run, progress=None)
    assert records
    tests_present = {r.test_type for r in records}
    assert {"early_answering", "paraphrase", "filler", "biased_context"} <= tests_present
    assert battery_summary(records)["per_model"]


def test_the_battery_separates_the_simulated_models(demo_run):
    """A model that decided before reasoning should answer from zero steps far
    more often than one whose reasoning is load-bearing."""
    records = demo.run_demo_battery(demo_run, progress=None)
    early = [r for r in records if r.test_type == "early_answering"
             and r.detail.get("k") == 0]
    rate = {}
    for model in {r.model_name for r in early}:
        rows = [r for r in early if r.model_name == model]
        rate[model] = sum(1 for r in rows if r.matches_baseline) / len(rows)
    assert rate["sim-detached"] > rate["sim-faithful"]


def test_a_paraphrase_mostly_leaves_the_answer_alone(demo_run):
    """The paraphrase control: rewording changes nothing, so the answer should
    not move except for a surface-reactive model."""
    records = demo.run_demo_battery(demo_run, progress=None)
    para = [r for r in records if r.test_type == "paraphrase"]
    stable = {}
    for model in {r.model_name for r in para}:
        rows = [r for r in para if r.model_name == model]
        stable[model] = sum(1 for r in rows if r.matches_baseline) / len(rows)
    assert stable["sim-faithful"] > 0.6
    # sim-jumpy is the surface-reactive one, by construction.
    assert stable["sim-jumpy"] < stable["sim-faithful"]


def test_the_demo_drives_the_introspection_probe(demo_run):
    from stepwise.introspect import introspect_summary

    rows = demo.run_demo_introspect(demo_run, limit=80, progress=None)
    assert rows
    summary = introspect_summary(rows)
    # Every model is represented: the sample is stratified, not the first N.
    assert len(summary) == 4
    by_model = {r["model_name"]: r for r in summary}
    # The model built to notice errors should detect more than the one built
    # to ignore its own reasoning.
    assert by_model["sim-faithful"]["detection_rate"] > by_model["sim-detached"]["detection_rate"]


def test_noise_is_treated_as_the_introspection_control(demo_run):
    """An irrelevant-noise edit adds a sentence but breaks nothing, so flagging
    it is a false alarm. If the provider scored it as a real error the
    false-alarm rate would collapse into the detection rate."""
    rows = demo.run_demo_introspect(demo_run, limit=80, progress=None)
    controls = [r for r in rows if r["is_control"]]
    assert controls
    flagged = sum(1 for r in controls if r["flagged"])
    assert flagged / len(controls) < 0.5


def test_the_demo_can_run_the_in_repo_benchmark(demo_run):
    """The probe set ships with the repo, so the benchmark panel is fillable
    offline — every model reasoning over the same authored chain."""
    from stepwise import probe_score

    result = demo.run_demo(dataset="probe", n_samples=12, progress=None)
    assert result.config.use_reference_cot is True
    rows = probe_score.probe_scores(result.records)
    assert rows
    by_model = {r["model_name"]: r for r in rows}
    # The model whose reasoning is load-bearing should discriminate a corrupted
    # critical step from an irrelevant-noise control better than the one that
    # ignores its own chain.
    assert by_model["sim-faithful"]["discrimination"] > by_model["sim-detached"]["discrimination"]


def test_every_model_shares_one_chain_on_the_probe_set():
    """The critical-step labels only apply if the chain is the authored one."""
    result = demo.run_demo(dataset="probe", n_samples=6, progress=None)
    by_question = {}
    for trace in result.traces.values():
        by_question.setdefault(trace.question_id, set()).add(trace.cot)
    assert by_question
    assert all(len(cots) == 1 for cots in by_question.values())
