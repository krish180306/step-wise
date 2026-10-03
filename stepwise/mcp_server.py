"""Stepwise as an MCP server: query a run from Claude Desktop or Claude Code.

The analysis tools already exist as plain functions over a run's records
(:mod:`stepwise.agent.tools`). This wraps a curated subset of them, plus the
battery and verifier summaries, as a Model Context Protocol server, so an
external Claude client can ask about your results directly — "which model's
reasoning is most load-bearing on arithmetic?" — and get exact numbers back,
not a guess.

The server is read-only: every tool reads a results file under ``results/``
and returns a computed answer. It never runs an evaluation, never calls Groq,
and never writes. The run to query defaults to ``gsm8k`` and can be set with
the ``STEPWISE_RUN`` environment variable or passed per call.

Run it with::

    python -m stepwise.mcp_server            # stdio, for a desktop client
    python main.py mcp                        # same thing via the CLI

and point your MCP client at that command. Needs ``pip install mcp`` (2.x).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import RESULTS_DIR
from .pipeline import load_records

DEFAULT_RUN = os.environ.get("STEPWISE_RUN", "gsm8k")


def _run_path(run: str) -> Path:
    name = Path(run).name
    if not name.endswith(".jsonl"):
        name = f"{name}.jsonl"
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"no run named {run!r}; files in results/: "
                                f"{[p.stem for p in RESULTS_DIR.glob('*.jsonl')]}")
    return path


def _ctx(run: str):
    from .agent.tools import AnalysisContext

    return AnalysisContext.from_files(_run_path(run))


def build_server():
    """Construct the MCP server. Imported lazily so the module loads without mcp."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("stepwise")

    @server.tool()
    def list_runs() -> str:
        """List the evaluation runs available to query (files in results/)."""
        runs = []
        for p in sorted(RESULTS_DIR.glob("*.jsonl")):
            if p.name.endswith((".traces.json",)) or ".battery" in p.name or ".mech" in p.name:
                continue
            n = sum(1 for _ in p.open(encoding="utf-8"))
            runs.append({"run": p.stem, "cells": n})
        return json.dumps({"runs": runs, "default": DEFAULT_RUN}, indent=2)

    @server.tool()
    def describe_run(run: str = DEFAULT_RUN) -> str:
        """Overview of a run: models, domains, perturbation classes, coverage."""
        from .agent.tools import tool_describe_run

        return json.dumps(tool_describe_run(_ctx(run)), indent=2, default=str)

    @server.tool()
    def faithfulness_scores(run: str = DEFAULT_RUN, group_by: str = "model_name,perturbation_type") -> str:
        """Faithfulness and dependency rates, grouped by any columns.

        group_by is a comma-separated list of: model_name, dataset, domain,
        perturbation_type. Rates are chance-corrected and carry their n.
        """
        from .agent.tools import tool_compute_scores

        cols = [c.strip() for c in group_by.split(",") if c.strip()]
        return json.dumps(tool_compute_scores(_ctx(run), group_by=cols), indent=2, default=str)

    @server.tool()
    def by_domain(run: str = DEFAULT_RUN) -> str:
        """Faithfulness by reasoning domain (reasoning steps only, chance-corrected,
        with question-clustered 95% confidence intervals). The cross-domain comparison.
        """
        from . import score

        return json.dumps(score.bootstrap_scores(load_records(_run_path(run))).to_dict(orient="records"),
                          indent=2, default=str)

    @server.tool()
    def answer_step_gap(run: str = DEFAULT_RUN) -> str:
        """How much of each model's apparent dependency was the trivial
        answer-bearing final step (dependency all-steps vs reasoning-only).
        """
        from . import score

        frame = score.records_dataframe(load_records(_run_path(run)))
        allm = {r["model_name"]: r for r in score.by_model(frame).to_dict(orient="records")}
        res = {r["model_name"]: r for r in
               score._aggregate(frame, ["model_name"], reasoning_steps_only=True).to_dict(orient="records")}
        out = []
        for m, a in allm.items():
            r = res.get(m, {}).get("dependency", 0.0)
            out.append({"model": m, "dependency_all_steps": a["dependency"],
                        "dependency_reasoning_only": r,
                        "share_was_answer_step": round((a["dependency"] - r) / a["dependency"], 4)
                        if a["dependency"] else None})
        return json.dumps(out, indent=2, default=str)

    @server.tool()
    def get_example(run: str, question_id: str, model_name: str, step_idx: int,
                    perturbation_type: str) -> str:
        """The full drill-down for one cell: both traces and the label."""
        from .agent.tools import tool_get_example

        return json.dumps(tool_get_example(_ctx(run), question_id=question_id,
                                           model_name=model_name, step_idx=step_idx,
                                           perturbation_type=perturbation_type), indent=2, default=str)

    @server.tool()
    def compare_models(run: str, model_a: str, model_b: str,
                       perturbation_type: str = "") -> str:
        """Head-to-head faithfulness for two models with a significance test."""
        from .agent.tools import tool_compare_models

        return json.dumps(tool_compare_models(_ctx(run), model_a=model_a, model_b=model_b,
                                              perturbation_type=perturbation_type or None),
                          indent=2, default=str)

    @server.tool()
    def battery_summary(run: str = DEFAULT_RUN) -> str:
        """The faithfulness battery results for a run: early-answering, paraphrase,
        filler and biased-context metrics per model (needs a <run>.battery.jsonl file).
        """
        from .battery_score import battery_summary as _bs

        path = _run_path(run).with_suffix("").with_suffix(".battery.jsonl")
        if not path.exists():
            return json.dumps({"error": f"no battery run at {path.name}; "
                               f"run: python main.py battery --dataset {run}"})
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return json.dumps(_bs(rows), indent=2, default=str)

    @server.tool()
    def verify_claim(run: str, claim: str) -> str:
        """Fact-check a sentence of claims about a run against the real numbers.

        Paste a sentence like "allam-2-7b was less faithful than qwen3.8 on
        arithmetic_change (p=0.022)"; every number is re-derived and marked
        SUPPORTED, UNSUPPORTED or UNCHECKABLE.
        """
        from .agent.verifier import verify_report

        return json.dumps(verify_report(claim, load_records(_run_path(run))), indent=2, default=str)

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
