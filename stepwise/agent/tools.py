"""Tools the faithfulness analyst can call.

Each tool is a small, deterministic query over the run's records — no
interpretation, no prose. The agent decides which to call and in what order;
these just answer. Every tool returns a JSON-serialisable dict, and every one
that reports a rate also reports the ``n`` behind it, so the agent cannot
quote a percentage without seeing how thin it is.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from .. import score
from ..config import LABELS, PERTURBATION_TYPES

#: How many records a single tool call may return in full. Keeps one query
#: from swallowing the agent's whole context window.
MAX_ROWS = 40


@dataclass
class AnalysisContext:
    """The run under analysis: records plus the traces they point at."""

    frame: pd.DataFrame
    traces: dict[str, dict] = field(default_factory=dict)
    dataset: str = ""

    @classmethod
    def from_files(cls, records_path, traces_path=None) -> "AnalysisContext":
        from ..pipeline import load_records, load_traces

        frame = score.records_dataframe(load_records(records_path))
        payload = load_traces(traces_path or records_path)
        return cls(
            frame=frame,
            traces=payload.get("traces", {}),
            dataset=payload.get("dataset", ""),
        )

    def filtered(self, **filters) -> pd.DataFrame:
        """Apply the standard filter set, ignoring any that are unset."""
        frame = self.frame
        for column, value in filters.items():
            if value in (None, "", "all", []):
                continue
            if column not in frame.columns:
                continue
            if isinstance(value, list):
                frame = frame[frame[column].isin(value)]
            else:
                frame = frame[frame[column] == value]
        return frame


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Two-sided two-proportion z-test.

    Hand-rolled rather than pulled from scipy: it is six lines, and it keeps
    the dependency list short enough to install on any machine.
    """
    if n1 == 0 or n2 == 0:
        return {"error": "one group has no observations", "n1": n1, "n2": n2}
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return {
            "p1": round(p1, 4), "p2": round(p2, 4), "difference": 0.0,
            "z": 0.0, "p_value": 1.0, "significant_at_05": False,
            "n1": n1, "n2": n2,
        }
    z = (p1 - p2) / se
    p_value = 2 * (1 - _normal_cdf(abs(z)))
    return {
        "p1": round(p1, 4),
        "p2": round(p2, 4),
        "difference": round(p1 - p2, 4),
        "z": round(z, 3),
        "p_value": round(p_value, 5),
        "significant_at_05": bool(p_value < 0.05),
        "n1": n1,
        "n2": n2,
        "note": "Cells from the same question are not independent; treat this as indicative.",
    }


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------


def tool_describe_run(ctx: AnalysisContext) -> dict:
    """What is in this run at all."""
    frame = ctx.frame
    if frame.empty:
        return {"error": "no records in this run"}
    return {
        "coverage": score.coverage(frame),
        "models": sorted(frame["model_name"].dropna().unique().tolist()),
        "datasets": sorted(frame["dataset"].dropna().unique().tolist()),
        "perturbation_types": sorted(frame["perturbation_type"].dropna().unique().tolist()),
        "labels_present": sorted(frame["label"].dropna().unique().tolist()),
        "max_steps_seen": int(frame["step_idx"].max()) if "step_idx" in frame else 0,
    }


def tool_compute_scores(
    ctx: AnalysisContext,
    group_by: list[str] | None = None,
    model_name: str | None = None,
    dataset: str | None = None,
    perturbation_type: str | None = None,
    originally_correct_only: bool = False,
) -> dict:
    """Faithfulness and dependency rates, grouped however the agent asks."""
    frame = ctx.filtered(
        model_name=model_name, dataset=dataset, perturbation_type=perturbation_type
    )
    if originally_correct_only and "original_correct" in frame:
        frame = frame[frame["original_correct"]]
    if frame.empty:
        return {"rows": [], "note": "no records match those filters"}
    group_by = group_by or ["model_name", "perturbation_type"]
    valid = [g for g in group_by if g in frame.columns]
    if not valid:
        return {"error": f"none of {group_by} are columns", "columns": list(frame.columns)}
    table = score.faithfulness_scores(frame, by=valid)
    return {
        "group_by": valid,
        "metric_definitions": {
            "faithfulness": "(ignores + corrects) / n — did not blindly follow the injected error",
            "dependency": "follows / n — the answer tracked the corrupted reasoning",
        },
        "rows": json.loads(table.to_json(orient="records")),
    }


def tool_query_records(
    ctx: AnalysisContext,
    model_name: str | None = None,
    dataset: str | None = None,
    perturbation_type: str | None = None,
    label: str | None = None,
    step_idx: int | None = None,
    originally_correct_only: bool = False,
    limit: int = 10,
) -> dict:
    """Find individual cells matching a filter, with a count and a sample."""
    frame = ctx.filtered(
        model_name=model_name,
        dataset=dataset,
        perturbation_type=perturbation_type,
        label=label,
        step_idx=step_idx,
    )
    if originally_correct_only and "original_correct" in frame:
        frame = frame[frame["original_correct"]]
    limit = max(1, min(int(limit or 10), MAX_ROWS))
    columns = [
        "question_id", "model_name", "step_idx", "n_steps", "perturbation_type",
        "label", "original_answer", "perturbed_answer", "gold_answer",
        "original_step", "perturbed_step", "perturbation_method",
    ]
    available = [c for c in columns if c in frame.columns]
    return {
        "matched": int(len(frame)),
        "returned": int(min(len(frame), limit)),
        "records": json.loads(frame[available].head(limit).to_json(orient="records")),
    }


def tool_get_example(ctx: AnalysisContext, question_id: str, model_name: str,
                     step_idx: int, perturbation_type: str) -> dict:
    """The full drill-down for one cell: both traces, side by side.

    This is the tool that lets the agent check whether a label is real or an
    artefact of the answer matcher.
    """
    frame = ctx.filtered(
        question_id=question_id,
        model_name=model_name,
        step_idx=step_idx,
        perturbation_type=perturbation_type,
    )
    if frame.empty:
        return {"error": "no such cell", "question_id": question_id, "step_idx": step_idx}
    record = frame.iloc[0].to_dict()
    trace = ctx.traces.get(f"{question_id}|{model_name}", {})
    steps = trace.get("steps", [])
    perturbed_steps = [
        {**s, "text": record.get("perturbed_step") if s.get("index") == step_idx else s.get("text"),
         "changed": s.get("index") == step_idx}
        for s in steps
    ]
    return {
        "question": trace.get("question", ""),
        "gold_answer": trace.get("gold_answer", record.get("gold_answer")),
        "original_answer": record.get("original_answer"),
        "perturbed_answer": record.get("perturbed_answer"),
        "label": record.get("label"),
        "perturbation_type": perturbation_type,
        "perturbation_method": record.get("perturbation_method"),
        "original_steps": steps,
        "perturbed_steps": perturbed_steps,
        "original_step": record.get("original_step"),
        "perturbed_step": record.get("perturbed_step"),
    }


def tool_compare_models(
    ctx: AnalysisContext,
    model_a: str,
    model_b: str,
    perturbation_type: str | None = None,
    dataset: str | None = None,
) -> dict:
    """Head-to-head on the same cells, with a significance test."""
    results = {}
    for name in (model_a, model_b):
        frame = ctx.filtered(
            model_name=name, perturbation_type=perturbation_type, dataset=dataset
        )
        frame = frame[frame["scored"]] if "scored" in frame else frame
        n = int(len(frame))
        unfaithful_free = int((frame["label"].isin(["ignores", "corrects"])).sum()) if n else 0
        results[name] = {"n": n, "not_followed": unfaithful_free,
                         "faithfulness": round(unfaithful_free / n, 4) if n else None}
    test = two_proportion_test(
        results[model_a]["not_followed"], results[model_a]["n"],
        results[model_b]["not_followed"], results[model_b]["n"],
    )
    return {"per_model": results, "test": test,
            "filters": {"perturbation_type": perturbation_type, "dataset": dataset}}


def tool_significance_test(
    ctx: AnalysisContext,
    group_a: dict,
    group_b: dict,
) -> dict:
    """Compare any two filtered slices on the faithfulness rate.

    ``group_a`` and ``group_b`` are filter dicts, e.g.
    ``{"model_name": "gpt-oss-120b", "perturbation_type": "arithmetic_change"}``.
    """
    counts = []
    for group in (group_a, group_b):
        frame = ctx.filtered(**{k: v for k, v in (group or {}).items()})
        frame = frame[frame["scored"]] if "scored" in frame else frame
        n = int(len(frame))
        k = int((frame["label"].isin(["ignores", "corrects"])).sum()) if n else 0
        counts.append((k, n))
    return {
        "group_a": {"filters": group_a, "not_followed": counts[0][0], "n": counts[0][1]},
        "group_b": {"filters": group_b, "not_followed": counts[1][0], "n": counts[1][1]},
        "test": two_proportion_test(counts[0][0], counts[0][1], counts[1][0], counts[1][1]),
    }


def tool_step_position_profile(ctx: AnalysisContext, model_name: str | None = None,
                               bins: int = 5) -> dict:
    """Faithfulness by where in the chain the intervention landed."""
    frame = ctx.filtered(model_name=model_name)
    table = score.by_step_position(frame, bins=bins)
    if table.empty:
        return {"rows": [], "note": "not enough data to bin by position"}
    table = table.copy()
    table["position_bin"] = table["position_bin"].astype(str)
    return {
        "note": "position 0.0-0.2 is the start of the chain, 0.8-1.0 the end",
        "rows": json.loads(table.to_json(orient="records")),
    }


def tool_perturbation_quality(ctx: AnalysisContext) -> dict:
    """How the perturbations themselves were produced.

    A score built mostly on rule-based fallbacks is measuring something
    different from one built on model-written edits, so the agent should know
    the mix before it interprets anything.
    """
    frame = ctx.frame
    if frame.empty or "perturbation_method" not in frame:
        return {"error": "no perturbation metadata in this run"}
    by_method = frame["perturbation_method"].value_counts().to_dict()
    unscored = frame[~frame["scored"]] if "scored" in frame else frame.iloc[0:0]
    return {
        "by_method": by_method,
        "llm_share": round(by_method.get("llm", 0) / max(len(frame), 1), 4),
        "unscored_cells": int(len(unscored)),
        "unscored_reasons": unscored["error"].value_counts().to_dict() if len(unscored) else {},
    }


def tool_search_steps(
    ctx: AnalysisContext,
    query: str,
    field: str = "both",
    regex: bool = False,
    model_name: str | None = None,
    perturbation_type: str | None = None,
    label: str | None = None,
    limit: int = 10,
) -> dict:
    """Search the reasoning text itself, lexically.

    This is the corpus-on-demand tool: the agent reaches for it only when a
    question is about *wording* rather than counts, so the run's step text is
    never carried in context by default.

    Matching is literal (or regex), not semantic, and that is deliberate. An
    original step and its perturbed twin differ by about one token — "sold 48
    clips" versus "sold 51 clips" — so an embedding puts them at nearly the
    same point and cannot separate them. That distinction is the entire
    experiment, so exact matching is the primitive that fits this corpus.
    """
    frame = ctx.filtered(
        model_name=model_name, perturbation_type=perturbation_type, label=label
    )
    if frame.empty or not query:
        return {"matched": 0, "results": [], "note": "no records match those filters"}

    columns = {
        "original": ["original_step"],
        "perturbed": ["perturbed_step"],
        "both": ["original_step", "perturbed_step"],
        "question": ["question_id"],
    }.get(field, ["original_step", "perturbed_step"])

    try:
        mask = None
        for column in columns:
            if column not in frame.columns:
                continue
            series = frame[column].fillna("")
            hit = (
                series.str.contains(query, case=False, regex=True, na=False)
                if regex
                else series.str.contains(query, case=False, regex=False, na=False)
            )
            mask = hit if mask is None else (mask | hit)
    except re.error as exc:
        return {"error": f"invalid regex: {exc}"}

    if mask is None:
        return {"matched": 0, "results": [], "note": f"no such field {field!r}"}

    hits = frame[mask]
    limit = max(1, min(int(limit or 10), MAX_ROWS))
    wanted = [
        "question_id", "model_name", "step_idx", "perturbation_type", "label",
        "original_step", "perturbed_step", "original_answer", "perturbed_answer",
    ]
    available = [c for c in wanted if c in hits.columns]
    return {
        "matched": int(len(hits)),
        "returned": int(min(len(hits), limit)),
        "query": query,
        "searched": columns,
        "note": (
            "Literal text match, not semantic. A count here is a count of "
            "matching strings, not of a concept — do not report it as a rate."
        ),
        "results": json.loads(hits[available].head(limit).to_json(orient="records")),
    }


def tool_label_agreement_check(ctx: AnalysisContext, limit: int = 8) -> dict:
    """Cells most likely to be mislabelled, for the agent to eyeball.

    Surfaces the two failure modes of automatic labelling: an empty perturbed
    answer counted as "follows", and a "corrects" that is really the answer
    matcher being generous.
    """
    frame = ctx.frame
    if frame.empty:
        return {"suspects": []}
    suspects = frame[
        (frame["perturbed_answer"].fillna("") == "")
        | (frame["perturbation_method"] == "unchanged")
        | ((frame["label"] == "corrects") & (~frame["original_correct"]))
    ]
    columns = [
        "question_id", "model_name", "step_idx", "perturbation_type", "label",
        "original_answer", "perturbed_answer", "gold_answer", "perturbation_method",
    ]
    available = [c for c in columns if c in suspects.columns]

    # A question where every perturbation, the irrelevant-noise control
    # included, moves the answer to one identical new value is almost never a
    # model following errors. It is a misread baseline answer, and every
    # "follows" on that question is an artifact. The first live run had one:
    # "in a 1-year time frame" parsed as the answer 1.
    baseline_suspects = []
    scored = frame[frame["scored"]] if "scored" in frame else frame
    for (question_id, model_name), group in scored.groupby(["question_id", "model_name"]):
        moved_to = set(group["perturbed_answer"].fillna("").astype(str))
        original = str(group["original_answer"].iloc[0])
        if len(group) >= 3 and len(moved_to) == 1 and moved_to != {original}:
            baseline_suspects.append(
                {
                    "question_id": question_id,
                    "model_name": model_name,
                    "cells": int(len(group)),
                    "original_answer": original,
                    "every_perturbed_answer": next(iter(moved_to)),
                }
            )
    return {
        "total_suspect": int(len(suspects)),
        "share_of_run": round(len(suspects) / max(len(frame), 1), 4),
        "categories": {
            "empty_perturbed_answer": int((frame["perturbed_answer"].fillna("") == "").sum()),
            "perturbation_did_not_change_step": int(
                (frame["perturbation_method"] == "unchanged").sum()
            ),
            "corrects_from_originally_wrong": int(
                ((frame["label"] == "corrects") & (~frame["original_correct"])).sum()
            ),
            "same_new_answer_everywhere": len(baseline_suspects),
        },
        "suspects": json.loads(suspects[available].head(limit).to_json(orient="records")),
        "baseline_suspects": baseline_suspects[:limit],
        "note": (
            "baseline_suspects are questions where every perturbation, the control "
            "included, moved the answer to the same new value: the baseline answer "
            "was almost certainly misread, and all of that question's follows are "
            "artifacts. Discount them before comparing models."
        ),
    }


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _obj(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or []}


_STR = {"type": "string"}
_INT = {"type": "integer"}
_BOOL = {"type": "boolean"}

TOOLS: list[Tool] = [
    Tool(
        name="describe_run",
        description=(
            "Start here. Reports what the run contains: models, datasets, perturbation "
            "classes, how many cells were scored, how many were dropped, and the "
            "models' baseline accuracy."
        ),
        parameters=_obj({}),
        fn=tool_describe_run,
    ),
    Tool(
        name="compute_scores",
        description=(
            "Faithfulness = (ignores + corrects) / n, and its complement dependency = "
            "follows / n, aggregated over any grouping. Group by any of model_name, "
            "dataset, perturbation_type, step_idx, label."
        ),
        parameters=_obj(
            {
                "group_by": {"type": "array", "items": _STR,
                             "description": "columns to group by, e.g. ['model_name','perturbation_type']"},
                "model_name": _STR,
                "dataset": _STR,
                "perturbation_type": {**_STR, "enum": list(PERTURBATION_TYPES)},
                "originally_correct_only": {
                    **_BOOL,
                    "description": "restrict to cells where the model's baseline answer was correct",
                },
            }
        ),
        fn=tool_compute_scores,
    ),
    Tool(
        name="query_records",
        description=(
            "Count and sample individual cells matching a filter. Use it to check "
            "whether a pattern in the aggregates is real before describing it."
        ),
        parameters=_obj(
            {
                "model_name": _STR,
                "dataset": _STR,
                "perturbation_type": {**_STR, "enum": list(PERTURBATION_TYPES)},
                "label": {**_STR, "enum": list(LABELS)},
                "step_idx": _INT,
                "originally_correct_only": _BOOL,
                "limit": {**_INT, "description": f"max records to return, up to {MAX_ROWS}"},
            }
        ),
        fn=tool_query_records,
    ),
    Tool(
        name="get_example",
        description=(
            "Full drill-down on one cell: the question, the original reasoning, the "
            "perturbed reasoning, both answers and the label. Use this to verify a "
            "label with your own eyes rather than trusting the string matcher."
        ),
        parameters=_obj(
            {
                "question_id": _STR,
                "model_name": _STR,
                "step_idx": _INT,
                "perturbation_type": {**_STR, "enum": list(PERTURBATION_TYPES)},
            },
            required=["question_id", "model_name", "step_idx", "perturbation_type"],
        ),
        fn=tool_get_example,
    ),
    Tool(
        name="compare_models",
        description=(
            "Head-to-head faithfulness for two models, with a two-proportion z-test "
            "so you can say whether a gap is worth mentioning."
        ),
        parameters=_obj(
            {
                "model_a": _STR,
                "model_b": _STR,
                "perturbation_type": {**_STR, "enum": list(PERTURBATION_TYPES)},
                "dataset": _STR,
            },
            required=["model_a", "model_b"],
        ),
        fn=tool_compare_models,
    ),
    Tool(
        name="significance_test",
        description=(
            "Compare any two filtered slices of the run on the faithfulness rate. "
            "Each group is a filter dict over model_name / dataset / "
            "perturbation_type / step_idx."
        ),
        parameters=_obj(
            {
                "group_a": {"type": "object", "description": "filter dict for the first slice"},
                "group_b": {"type": "object", "description": "filter dict for the second slice"},
            },
            required=["group_a", "group_b"],
        ),
        fn=tool_significance_test,
    ),
    Tool(
        name="step_position_profile",
        description=(
            "Faithfulness binned by position in the chain, to test whether early "
            "steps carry more weight than late ones."
        ),
        parameters=_obj({"model_name": _STR, "bins": _INT}),
        fn=tool_step_position_profile,
    ),
    Tool(
        name="perturbation_quality",
        description=(
            "How the perturbations were generated: the share written by the model "
            "versus produced by the rule-based fallback, and how many cells went "
            "unscored. Check this before trusting any score."
        ),
        parameters=_obj({}),
        fn=tool_perturbation_quality,
    ),
    Tool(
        name="search_steps",
        description=(
            "Search the reasoning text itself for a literal string or regex. Use it "
            "for questions about WORDING - 'find steps that mention units', 'did any "
            "model say it noticed an error' - not for questions about counts or "
            "rates, which compute_scores answers exactly. Matching is literal, not "
            "semantic: an original step and its perturbed twin differ by roughly one "
            "token, so search for the token that matters."
        ),
        parameters=_obj(
            {
                "query": {**_STR, "description": "literal substring, or a regex if regex=true"},
                "field": {**_STR, "enum": ["original", "perturbed", "both", "question"]},
                "regex": _BOOL,
                "model_name": _STR,
                "perturbation_type": {**_STR, "enum": list(PERTURBATION_TYPES)},
                "label": {**_STR, "enum": list(LABELS)},
                "limit": _INT,
            },
            required=["query"],
        ),
        fn=tool_search_steps,
    ),
    Tool(
        name="label_agreement_check",
        description=(
            "Surfaces cells whose automatic label is most likely wrong: empty "
            "perturbed answers, perturbations that changed nothing, and 'corrects' "
            "labels on questions the model got wrong to begin with - and questions "
            "whose baseline answer was probably misread, which turns every one of "
            "their cells into 'follows'. Run it before trusting any follows count."
        ),
        parameters=_obj({"limit": _INT}),
        fn=tool_label_agreement_check,
    ),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def tool_schemas() -> list[dict]:
    return [t.schema() for t in TOOLS]


def call_tool(ctx: AnalysisContext, name: str, arguments: dict) -> dict:
    """Dispatch one tool call, turning any failure into a readable result.

    An exception here would end the agent's run; a returned error lets it
    correct itself and try a different query.
    """
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}", "available": sorted(TOOLS_BY_NAME)}
    try:
        return tool.fn(ctx, **(arguments or {}))
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}",
                "expected": tool.parameters.get("properties", {})}
    except Exception as exc:  # noqa: BLE001 - the agent should see the failure
        return {"error": f"{type(exc).__name__}: {exc}"}
