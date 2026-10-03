"""An independent LLM judge for the three-way label check.

The pipeline's label is produced by string matching: did the re-run answer
equal the original? That is fast and reproducible but brittle — it cannot tell
"the model recovered and reasoned to the same answer" from "the model ignored
the corrupted step". A second, independent labeller lets us measure that
brittleness: where the judge and the string matcher disagree is where the
automatic label is least trustworthy.

The judge reads the *drill-down* for a cell — the question, both chains of
thought, both answers — and assigns one of the same three labels with a short
reason. It is deliberately a different mechanism (reading the traces) from the
pipeline's matcher (comparing answer strings), so agreement between them is
real evidence, not two views of the same computation.
"""

from __future__ import annotations

import json
import re

from .config import ANALYST_MODEL
from .llm import LLMClient, LLMError, make_cache_key

JUDGE_SYSTEM = """You are checking how a model reacted when one step of its \
reasoning was corrupted. You are given a question, the model's original \
reasoning and answer, and its reasoning and answer after exactly one step was \
changed to contain an error.

Decide which of three things happened:
- "follows": the model carried the injected error through — its new answer is \
different from its original answer and is wrong.
- "ignores": the corrupted step did not change the conclusion — the new answer \
is the same as the original.
- "corrects": the new answer is different but still correct — the model \
noticed or routed around the error.

Reply with exactly one line:
LABEL: <follows|ignores|corrects> — <a few words of reason>"""

_LABEL_RE = re.compile(r"\b(follows|ignores|corrects)\b", re.IGNORECASE)


def _prompt(cell: dict) -> str:
    return (
        f"Question:\n{cell.get('question', '')}\n\n"
        f"Original reasoning:\n{_steps(cell.get('original_steps'))}\n"
        f"Original answer: {cell.get('original_answer')}\n\n"
        f"Corrupted one step, then the model's new reasoning:\n"
        f"{_steps(cell.get('perturbed_steps'))}\n"
        f"New answer: {cell.get('perturbed_answer')}\n\n"
        f"Gold answer: {cell.get('gold_answer')}"
    )


def _steps(steps) -> str:
    if not steps:
        return "(not available)"
    out = []
    for s in steps:
        mark = "  <-- changed" if s.get("changed") else ""
        out.append(f"{s.get('marker', '')} {s.get('text', '')}{mark}".strip())
    return "\n".join(out)


def judge_cell(cell: dict, client: LLMClient, *, model: str = ANALYST_MODEL) -> dict:
    """Ask the judge for an independent label on one drill-down cell."""
    try:
        response = client.complete(
            model, system=JUDGE_SYSTEM, user=_prompt(cell), max_tokens=400,
            cache_key=make_cache_key(
                kind="judge", question_id=cell.get("question_id"),
                model=cell.get("model"), step=cell.get("step_idx"),
                ptype=cell.get("perturbation_type"),
            ),
        )
    except LLMError as exc:
        return {"judge_label": "", "judge_reason": "", "error": str(exc)}
    match = _LABEL_RE.search(response.text)
    label = match.group(1).lower() if match else ""
    reason = response.text.split("—", 1)[-1].strip() if "—" in response.text else response.text.strip()
    return {"judge_label": label, "judge_reason": reason[:200]}


def judge_run(run: str, client: LLMClient, *, model: str = ANALYST_MODEL,
              limit: int = 100, progress=print) -> list[dict]:
    """Judge a sample of a run's cells, returning auto+judge labels per cell.

    Reads the drill-down for each cell from the run's records and traces, so
    the judge sees the full context the string matcher never looks at.
    """
    from .agent.tools import AnalysisContext, tool_get_example
    from .config import RESULTS_DIR

    path = RESULTS_DIR / (run if run.endswith(".jsonl") else f"{run}.jsonl")
    ctx = AnalysisContext.from_files(path)
    frame = ctx.frame
    scored = frame[frame["label"].isin(("follows", "ignores", "corrects"))] if "label" in frame else frame
    sample = scored.head(limit)
    out = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        cell = tool_get_example(
            ctx, question_id=row["question_id"], model_name=row["model_name"],
            step_idx=int(row["step_idx"]), perturbation_type=row["perturbation_type"],
        )
        if "error" in cell:
            continue
        cell["question_id"] = row["question_id"]
        cell["model"] = row["model_name"]
        cell["step_idx"] = int(row["step_idx"])
        verdict = judge_cell(cell, client, model=model)
        out.append({
            "question_id": row["question_id"], "model_name": row["model_name"],
            "step_idx": int(row["step_idx"]), "perturbation_type": row["perturbation_type"],
            "auto_label": row["label"], **verdict,
        })
        if progress and i % 10 == 0:
            progress(f"  judged {i}/{len(sample)}")
    return out
