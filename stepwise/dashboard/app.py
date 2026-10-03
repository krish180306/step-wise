"""FastAPI backend for the Stepwise dashboard.

Serves the single-page front end from ``static/`` and a small read-only JSON
API over whatever result files are in ``results/``. The front end is the
Stepwise design wired to this API — it does no computation of its own, so
every number on screen comes from :mod:`stepwise.score`.

Run it with::

    python main.py dashboard
    # or: uvicorn stepwise.dashboard.app:app --reload
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import score
from ..agent.chat import SESSIONS, stream_reply
from ..agent.tools import AnalysisContext
from ..config import ANALYST_MODEL, LABELS, PERTURBATION_TYPES, RESULTS_DIR, api_key
from ..llm import LLMError
from ..pipeline import load_records, load_traces

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Stepwise", description="CoT faithfulness dashboard", version="0.1.0")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _run_path(run: str) -> Path:
    """Resolve a run name to a results file, refusing paths outside results/."""
    name = Path(run).name  # strip any directory component
    if not name.endswith(".jsonl"):
        name = f"{name}.jsonl"
    path = RESULTS_DIR / name
    if not path.exists():
        raise HTTPException(404, f"no run named {run!r}; try /api/runs")
    return path


@lru_cache(maxsize=8)
def _cached_payload(path_str: str, mtime: float) -> dict:
    """Parse a results file once per modification.

    ``mtime`` is part of the key rather than something checked inside, so a
    rerun of the pipeline invalidates the cache for free.
    """
    path = Path(path_str)
    records = load_records(path)
    frame = score.records_dataframe(records)
    traces = load_traces(path).get("traces", {})
    return {"frame": frame, "traces": traces, "records": records}


def _load(run: str) -> dict:
    path = _run_path(run)
    return _cached_payload(str(path), path.stat().st_mtime)


def _filtered(frame, **filters):
    for column, value in filters.items():
        if value in (None, "", "all") or column not in frame.columns:
            continue
        frame = frame[frame[column] == value]
    return frame


def _json_safe(value):
    """Replace NaN and infinity with None, recursively.

    pandas reports "no data" as NaN. Accuracy retained, for one, is NaN for a
    group where the model got every question wrong at baseline, since there
    is nothing to retain. NaN is not valid JSON, so the encoder rejects the
    whole response. The simulated demo never produced one; the first real run
    did, and /api/summary returned a 500 that left the dashboard blank.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@app.get("/api/runs")
def list_runs() -> dict:
    """Every results file on disk: real runs first, newest first within each."""
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.jsonl"), key=lambda p: -p.stat().st_mtime):
        sidecar = path.with_suffix(".traces.json")
        meta = {}
        if sidecar.exists():
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                meta = {
                    "models": payload.get("models", []),
                    "duration_s": payload.get("duration_s"),
                    "simulated": bool(payload.get("simulated")),
                }
            except json.JSONDecodeError:
                pass
        runs.append(
            {
                "name": path.stem,
                "records": sum(1 for _ in path.open(encoding="utf-8")),
                "modified": path.stat().st_mtime,
                "has_analysis": (RESULTS_DIR / f"analysis_{path.stem}.md").exists(),
                **meta,
            }
        )
    # Real runs first, newest first within each group. A simulated demo must
    # never be what the dashboard opens on while real results exist, even if
    # the demo was generated more recently.
    runs.sort(key=lambda r: (bool(r.get("simulated")), -r["modified"]))
    return {"runs": runs, "labels": list(LABELS), "perturbation_types": list(PERTURBATION_TYPES)}


@app.get("/api/summary")
def summary(run: str = Query(..., description="run name, e.g. 'gsm8k'")) -> dict:
    """Everything the dashboard's charts need, in one payload."""
    data = _load(run)
    frame = data["frame"]
    if frame.empty:
        return {"run": run, "empty": True}
    sidecar = load_traces(_run_path(run))
    payload = score.summary(frame)
    payload.update(
        {
            "run": run,
            "empty": False,
            "simulated": bool(sidecar.get("simulated")),
            "warning": sidecar.get("warning"),
            "models": sorted(frame["model_name"].dropna().unique().tolist()),
            "datasets": sorted(frame["dataset"].dropna().unique().tolist()),
            "perturbation_types": [
                t for t in PERTURBATION_TYPES
                if t in set(frame["perturbation_type"].dropna())
            ],
            "labels": list(LABELS),
        }
    )
    return JSONResponse(_json_safe(json.loads(json.dumps(payload, default=str))))


@app.get("/api/grid")
def grid(
    run: str,
    model: str = "all",
    dataset: str = "all",
    metric: str = "faithfulness",
) -> dict:
    """The per-step dependency grid: rows are question x step, columns class.

    This is the drill-down heatmap from the design — one cell per
    intervention, coloured by its label, rather than an aggregate.
    """
    frame = _filtered(_load(run)["frame"], model_name=model, dataset=dataset)
    if frame.empty:
        return {"rows": [], "types": list(PERTURBATION_TYPES)}

    types = [t for t in PERTURBATION_TYPES if t in set(frame["perturbation_type"])]
    rows = []
    for (question_id, step_idx), group in frame.groupby(["question_id", "step_idx"]):
        by_type = {r["perturbation_type"]: r for r in group.to_dict("records")}
        rows.append(
            {
                "question_id": question_id,
                "step_idx": int(step_idx),
                "n_steps": int(group["n_steps"].max()),
                "label": f"{question_id.split('-')[-1]} · s{int(step_idx)}",
                "cells": [
                    {
                        "type": t,
                        "label": (by_type.get(t) or {}).get("label") or None,
                        "model": (by_type.get(t) or {}).get("model_name"),
                        "title": _cell_title(by_type.get(t), t),
                    }
                    for t in types
                ],
            }
        )
    rows.sort(key=lambda r: (r["question_id"], r["step_idx"]))
    return {"rows": rows[:400], "types": types, "total_rows": len(rows)}


def _cell_title(record: dict | None, ptype: str) -> str:
    if not record:
        return f"{ptype}: not run"
    label = record.get("label") or "unscored"
    return (
        f"{ptype} · {label}\n"
        f"{record.get('original_answer')} -> {record.get('perturbed_answer')}"
    )


@app.get("/api/records")
def records(
    run: str,
    model: str = "all",
    dataset: str = "all",
    perturbation_type: str = "all",
    label: str = "all",
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Paged drill-down rows for the examples table."""
    frame = _filtered(
        _load(run)["frame"],
        model_name=model,
        dataset=dataset,
        perturbation_type=perturbation_type,
        label=label,
    )
    columns = [
        "question_id", "model_name", "dataset", "step_idx", "n_steps",
        "perturbation_type", "label", "original_answer", "perturbed_answer",
        "gold_answer", "original_step", "perturbed_step", "perturbation_method",
        "original_correct",
    ]
    available = [c for c in columns if c in frame.columns]
    limit = max(1, min(int(limit), 200))
    page = frame[available].iloc[offset : offset + limit]
    return {
        "total": int(len(frame)),
        "offset": offset,
        "limit": limit,
        "rows": json.loads(page.to_json(orient="records")),
    }


@app.get("/api/example")
def example(
    run: str,
    question_id: str,
    model: str,
    step_idx: int,
    perturbation_type: str,
) -> dict:
    """One cell in full: the original and perturbed traces, side by side."""
    data = _load(run)
    frame = _filtered(
        data["frame"],
        question_id=question_id,
        model_name=model,
        step_idx=step_idx,
        perturbation_type=perturbation_type,
    )
    if frame.empty:
        raise HTTPException(404, "no such cell")
    record = json.loads(frame.iloc[[0]].to_json(orient="records"))[0]
    trace = data["traces"].get(f"{question_id}|{model}", {})
    steps = trace.get("steps", [])
    return {
        "question": trace.get("question", ""),
        "gold_answer": trace.get("gold_answer", record.get("gold_answer")),
        "answer_type": trace.get("answer_type", record.get("answer_type")),
        "original": {
            "answer": record.get("original_answer"),
            "correct": bool(record.get("original_correct")),
            "steps": steps,
        },
        "perturbed": {
            "answer": record.get("perturbed_answer"),
            "correct": bool(record.get("perturbed_correct")),
            "steps": [
                {
                    "index": s.get("index"),
                    "marker": s.get("marker", ""),
                    "text": (
                        record.get("perturbed_step")
                        if s.get("index") == step_idx
                        else s.get("text")
                    ),
                    "was": s.get("text") if s.get("index") == step_idx else None,
                    "changed": s.get("index") == step_idx,
                }
                for s in steps
            ],
        },
        "label": record.get("label"),
        "perturbation_type": perturbation_type,
        "perturbation_method": record.get("perturbation_method"),
        "step_idx": step_idx,
        "model": model,
    }


@app.get("/api/analysis")
def analysis(run: str) -> dict:
    """The agent's written analysis for a run, if one has been produced."""
    path = RESULTS_DIR / f"analysis_{run}.md"
    if not path.exists():
        return {"exists": False, "run": run}
    transcript_path = path.with_suffix(".json")
    transcript = {}
    if transcript_path.exists():
        try:
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            transcript = {}
    return {
        "exists": True,
        "run": run,
        "report": path.read_text(encoding="utf-8"),
        "model": transcript.get("model", ""),
        "turns": transcript.get("turns", 0),
        "tool_calls": transcript.get("tool_calls", []),
    }


# --------------------------------------------------------------------------
# Agentic chat
# --------------------------------------------------------------------------


def _context_for(run: str) -> AnalysisContext:
    data = _load(run)
    return AnalysisContext(
        frame=data["frame"], traces=data["traces"], dataset=run
    )


@app.get("/api/chat/status")
def chat_status() -> dict:
    """Whether chat can run at all, so the UI can explain itself."""
    model = ANALYST_MODEL
    from ..config import resolve_model

    spec = resolve_model(model)
    return {
        "available": bool(api_key(spec.provider)),
        "model": spec.display,
        "provider": spec.provider,
        "requires": {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }.get(spec.provider, spec.provider.upper()),
        "sessions": len(SESSIONS),
    }


@app.post("/api/chat")
def chat(
    run: str = Body(...),
    message: str = Body(...),
    session_id: str | None = Body(None),
    model: str | None = Body(None),
) -> StreamingResponse:
    """Answer a question about a run, streaming the agent's work as SSE.

    Each event is one line of ``data: {json}``. The agent decides its own
    tool calls; the client renders them as they arrive, which is what makes
    the loop legible rather than a spinner.
    """
    message = (message or "").strip()
    if not message:
        raise HTTPException(400, "message is empty")
    if len(message) > 4000:
        raise HTTPException(400, "message too long")

    ctx = _context_for(run)  # raises 404 for an unknown run before streaming
    if ctx.frame.empty:
        raise HTTPException(400, f"run {run!r} has no records to talk about")

    session = SESSIONS.get_or_create(session_id, run, ctx, model or ANALYST_MODEL)

    def sse(payload: dict) -> str:
        """One SSE frame. The blank line after the data is what flushes it."""
        return f"data: {json.dumps(payload, default=str)}\n\n"

    def events():
        try:
            for event in stream_reply(session, message):
                yield sse(event)
        except LLMError as exc:
            yield sse({"type": "error", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 - the stream must always close
            yield sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/reset")
def chat_reset(session_id: str = Body(..., embed=True)) -> dict:
    """Forget a conversation."""
    return {"reset": SESSIONS.reset(session_id)}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "results_dir": str(RESULTS_DIR)}


# --------------------------------------------------------------------------
# Static front end
# --------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Start the dashboard. Called by ``python main.py dashboard``."""
    import uvicorn

    uvicorn.run(
        "stepwise.dashboard.app:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
