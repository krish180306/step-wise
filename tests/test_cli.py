"""Tests for the CLI surface added alongside the new analyses.

The `power` command and the `--perturbations` validation are the two places a
user can be given a wrong answer or a confusing crash, so both are pinned here.
Nothing in this file touches the network or needs a key.
"""

from __future__ import annotations

import pytest

import main as cli
from stepwise import demo


@pytest.fixture
def run_file(tmp_path):
    """A small offline demo run on disk, for the commands that read one."""
    result = demo.run_demo(dataset="gsm8k", n_samples=2, progress=None)
    path = tmp_path / "demo.jsonl"
    result.save(path)
    return path


def _args(**overrides):
    args = cli.build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# --------------------------------------------------------------------------
# power
# --------------------------------------------------------------------------


def test_power_is_a_registered_command():
    assert "power" in cli.COMMANDS
    assert cli.build_parser().parse_args(["power"]).command == "power"


def test_power_reads_the_design_off_a_run(run_file, capsys):
    assert cli.cmd_power(_args(output=str(run_file))) == 0
    out = capsys.readouterr().out
    assert "Measuring the design from" in out
    assert "intra-cluster correlation" in out
    assert "QUESTIONS NEEDED" in out


def test_power_falls_back_to_defaults_without_a_run(tmp_path, capsys):
    assert cli.cmd_power(_args(output=str(tmp_path / "missing.jsonl"))) == 0
    out = capsys.readouterr().out
    assert "using defaults" in out
    assert "QUESTIONS NEEDED" in out


def test_power_reports_the_design_effect_rather_than_hiding_it(run_file, capsys):
    cli.cmd_power(_args(output=str(run_file)))
    out = capsys.readouterr().out
    assert "design effect" in out.lower()
    assert "not" in out and "independent" in out


def test_power_honours_the_effect_size(tmp_path, capsys):
    def questions(effect):
        cli.cmd_power(_args(output=str(tmp_path / "missing.jsonl"), effect=effect))
        line = [l for l in capsys.readouterr().out.splitlines()
                if "QUESTIONS NEEDED" in l][0]
        return int(line.split()[-1])

    # A smaller gap to detect needs a bigger run.
    assert questions("0.30,0.25") > questions("0.60,0.25")


def test_power_rejects_a_malformed_effect(tmp_path):
    with pytest.raises(SystemExit, match="two numbers"):
        cli.cmd_power(_args(output=str(tmp_path / "missing.jsonl"), effect="0.4"))


def test_power_reports_an_untabulated_level_legibly(tmp_path):
    with pytest.raises(SystemExit, match="alpha"):
        cli.cmd_power(_args(output=str(tmp_path / "missing.jsonl"), alpha=0.037))


# --------------------------------------------------------------------------
# --perturbations validation
# --------------------------------------------------------------------------


def test_unknown_intervention_class_is_rejected_before_any_call():
    with pytest.raises(SystemExit, match="Unknown intervention class"):
        cli.cmd_run(_args(perturbations="fact_substitution,not_a_class"))


def test_the_error_names_every_valid_class():
    with pytest.raises(SystemExit) as caught:
        cli.cmd_run(_args(perturbations="bogus"))
    message = str(caught.value)
    assert "step_deletion" in message
    assert "irrelevant_noise" in message


def test_step_deletion_is_accepted(monkeypatch):
    """The ablation class must pass validation and reach the config."""
    captured = {}

    def fake_exit(message):
        raise AssertionError(f"should not have exited: {message}")

    monkeypatch.setattr(cli.sys, "exit", fake_exit)
    monkeypatch.setattr(cli, "_require_key", lambda models: None)

    import stepwise.pipeline as pipeline

    def fake_run(config, **kwargs):
        captured["types"] = config.perturbation_types
        raise KeyboardInterrupt  # stop before anything is written

    monkeypatch.setattr(pipeline, "run_evaluation", fake_run)
    with pytest.raises(KeyboardInterrupt):
        cli.cmd_run(_args(perturbations="arithmetic_change,step_deletion"))
    assert captured["types"] == ["arithmetic_change", "step_deletion"]


# --------------------------------------------------------------------------
# The report printers
# --------------------------------------------------------------------------


def test_tautology_printer_labels_the_honest_number(capsys):
    from stepwise import score

    records = [{
        "question_id": "q1", "dataset": "gsm8k", "model_name": "m",
        "step_idx": 2, "perturbation_type": "arithmetic_change", "label": "follows",
        "original_answer": "72", "perturbed_answer": "99", "gold_answer": "72",
        "original_correct": True, "perturbed_correct": False, "n_steps": 3,
        "is_answer_step": False, "original_step": "So 48 + 24 = 72.",
        "perturbed_step": "So 48 + 24 = 99.", "answer_type": "numeric",
        "n_choices": 0, "error": "",
    }]
    cli._print_tautology(score.tautology_report(records))
    out = capsys.readouterr().out
    assert "the honest number" in out
    assert "NOT the final step" in out


def test_tautology_printer_is_silent_on_an_empty_run(capsys):
    cli._print_tautology({"n_scored": 0})
    assert capsys.readouterr().out == ""


def test_necessity_printer_is_silent_without_ablation_cells(capsys):
    cli._print_necessity({"n_ablation_cells": 0})
    assert capsys.readouterr().out == ""


def test_necessity_printer_explains_the_cross_tab(capsys):
    cli._print_necessity({
        "n_ablation_cells": 4, "n_reasoning_cells": 4,
        "necessity_rate_reasoning": 0.5, "n_paired": 4,
        "cross_tab": {
            "necessary_and_followed": 1, "necessary_not_followed": 1,
            "redundant_but_followed": 1, "redundant_and_ignored": 1,
            "agreement": 0.5,
        },
    })
    out = capsys.readouterr().out
    assert "was the step NEEDED?" in out
    assert "without trusting it" in out
    assert "surface reactivity" in out


# --------------------------------------------------------------------------
# score still works end to end
# --------------------------------------------------------------------------


def test_score_command_prints_the_new_sections(run_file, capsys):
    assert cli.cmd_score(_args(output=str(run_file))) == 0
    out = capsys.readouterr().out
    assert "Per model" in out
    assert "Tautology check" in out


def test_score_command_exits_cleanly_on_a_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="No results"):
        cli.cmd_score(_args(output=str(tmp_path / "nope.jsonl")))


# --------------------------------------------------------------------------
# What --dataset means, per command
# --------------------------------------------------------------------------
#
# The dashboard builds the commands its empty states suggest, and it has to
# know which of these take a task family and which take a saved run. Getting
# it wrong produces a command that exits with "Unknown dataset" (or "No
# results for ..."), which is what the hints used to do for every run not
# named after a dataset - including `demo`, the one the dashboard opens on.


def test_generating_commands_validate_against_the_dataset_list():
    """run / battery / mech take a task family and reject anything else."""
    from stepwise.config import DATASETS

    assert "demo" not in DATASETS
    for command in (cli.cmd_run, cli.cmd_battery, cli.cmd_mech):
        with pytest.raises(SystemExit, match="Unknown dataset"):
            command(_args(dataset="demo"))


def test_reading_commands_take_a_run_name(run_file, monkeypatch, tmp_path):
    """introspect / analyse read an existing results file instead."""
    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path)
    import stepwise.config as config_mod

    monkeypatch.setattr(config_mod, "RESULTS_DIR", tmp_path)
    # The run is called "demo", which is not a dataset; these must get past
    # the name and fail on the missing key, not on the name.
    with pytest.raises(SystemExit, match="Missing API key"):
        cli.cmd_analyse(_args(dataset="demo", output=str(run_file)))


def test_the_page_builds_commands_through_one_helper():
    from pathlib import Path

    from stepwise.dashboard import app as mod

    page = (Path(mod.__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    assert "function cliCommand(" in page
    # The generating commands go through the helper...
    for verb in ("battery", "mech", "run"):
        assert f'cliCommand("{verb}"' in page
    # ...and introspect deliberately does not, because its --dataset is a run.
    assert "python main.py introspect --dataset ${esc(state.run)}" in page


def test_mech_sidecars_follow_the_output_stem():
    """--validate and --patch used to write beside --dataset, scattering the
    three outputs across two runs so the dashboard found none of them."""
    source = (cli.Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    assert 'stem = path.name[: -len(".mech.jsonl")]' in source
    assert 'out_dir / f"{stem}.mech_behav.jsonl"' in source
    assert 'out_dir / f"{stem}.patch.jsonl"' in source


# --------------------------------------------------------------------------
# interactions
# --------------------------------------------------------------------------


def test_interactions_is_a_registered_command():
    assert "interactions" in cli.COMMANDS
    assert cli.build_parser().parse_args(["interactions"]).command == "interactions"


def test_interactions_rejects_a_run_name_as_dataset():
    with pytest.raises(SystemExit, match="Unknown dataset"):
        cli.cmd_interactions(_args(dataset="demo"))


def test_interactions_rejects_an_unknown_pair_class():
    with pytest.raises(SystemExit, match="Unknown intervention class"):
        cli.cmd_interactions(_args(pair_perturbations="arithmetic_change,bogus"))


def test_interactions_accepts_the_ablation_class():
    """Deleting two steps at once is a legitimate pair intervention."""
    with pytest.raises(SystemExit, match="Missing API key"):
        cli.cmd_interactions(_args(pair_perturbations="step_deletion"))


def test_interactions_defaults_are_sane():
    args = cli.build_parser().parse_args([])
    assert args.max_pairs == 6          # pairs are quadratic; the grid is capped
    assert args.untargeted is False     # targeted by default
    assert args.pair_perturbations == "arithmetic_change"
