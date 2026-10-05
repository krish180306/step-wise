"""The Stepwise Faithfulness Benchmark — packaging the probe set as a real,
versioned benchmark rather than a loose dataset.

What makes it a benchmark and not just data:

1. **Ground truth.** Every item's load-bearing (critical) step is annotated, so
   a corruption has a correct outcome (see :mod:`stepwise.probe_score`).
2. **Difficulty tiers.** ``easy`` / ``medium`` / ``adversarial``. Adversarial
   items contain a *distractor* step — one dressed up to look pivotal that is
   not — so the benchmark separates real reasoning from reacting to
   authoritative-sounding filler.
3. **Validation.** :func:`validate_items` checks every item is internally
   consistent and *operationalisable* (each critical step actually contains
   something a perturbation can corrupt), so the ground truth is enforceable,
   not just asserted.
4. **A frozen manifest.** :func:`build_manifest` records the version, the item
   counts, and a content checksum, so a result can name exactly which version
   of the benchmark it was measured on.
5. **Independent annotation check.** :func:`llm_verify` has a second,
   independent labeller (a model) pick each item's most load-bearing step and
   reports how often it agrees with the author — inter-annotator agreement, the
   usual evidence that an annotation is real.

Running it is ``python main.py --dataset probe``; this module is the surrounding
apparatus — validate, manifest, report, verify.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .datasets import PROBE_PATH

#: Bump on any change to the item set, so results can pin the version.
BENCHMARK_VERSION = "1.0"
NAME = "Stepwise Faithfulness Benchmark"
TIERS = ("easy", "medium", "adversarial")

#: The consequential classes (the control, noise, applies to any step).
_CONSEQUENTIAL = ("fact_substitution", "logical_error", "arithmetic_change")


def load_items() -> list[dict]:
    """Read the raw benchmark items from the repo file."""
    if not PROBE_PATH.exists():
        return []
    return [json.loads(line) for line in PROBE_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _step_is_fallback_perturbable(text: str) -> bool:
    """Can the deterministic (no-API) fallback actually corrupt this step?

    The real test, not a word heuristic: try each consequential class's
    rule-based fallback and see if any changes the text. Steps that fail this
    are still perturbable by the LLM writer in a normal run — this is a coverage
    stat for the offline path, not a correctness gate.
    """
    from .perturbations import apply_fallback
    return any(apply_fallback(text, p).changed for p in _CONSEQUENTIAL)


def validate_items(items: list[dict] | None = None) -> list[str]:
    """Structural / consistency checks. Returns a list of problems (empty = OK).

    This is the deterministic "second annotator": it cannot judge whether a step
    is *conceptually* load-bearing, but it enforces that the labels are
    self-consistent and that each critical step is something a perturbation can
    actually corrupt — so the ground truth is enforceable rather than asserted.
    """
    items = load_items() if items is None else items
    issues: list[str] = []
    seen_ids: set[str] = set()
    for item in items:
        qid = item.get("id", "<no id>")
        if qid in seen_ids:
            issues.append(f"{qid}: duplicate id")
        seen_ids.add(qid)
        steps = item.get("steps", [])
        n = len(steps)
        crit = set(item.get("critical_steps", []))
        red = set(item.get("redundant_steps", []))
        distr = set(item.get("distractor_steps", []))
        rng = set(range(1, n + 1))

        if n < 2:
            issues.append(f"{qid}: fewer than 2 steps")
        if not item.get("gold_answer"):
            issues.append(f"{qid}: missing gold_answer")
        if item.get("tier") not in TIERS:
            issues.append(f"{qid}: tier {item.get('tier')!r} not in {TIERS}")
        if not crit:
            issues.append(f"{qid}: no critical steps")
        if not crit <= rng:
            issues.append(f"{qid}: critical steps out of range")
        if not red <= rng:
            issues.append(f"{qid}: redundant steps out of range")
        if crit & red:
            issues.append(f"{qid}: steps are both critical and redundant: {sorted(crit & red)}")
        if crit | red != rng:
            issues.append(f"{qid}: every step must be critical or redundant (missing "
                          f"{sorted(rng - (crit | red))})")
        if not distr <= red:
            issues.append(f"{qid}: distractor steps must be a subset of redundant")
        if item.get("tier") == "adversarial" and not distr:
            issues.append(f"{qid}: adversarial item has no distractor step")
        if item.get("answer_type") == "label" and not item.get("choices"):
            issues.append(f"{qid}: label item has no choices")
    return issues


def fallback_coverage(items: list[dict] | None = None) -> dict:
    """Fraction of critical steps the no-API fallback can corrupt.

    The LLM writer handles the rest in a normal run; this measures how much of
    the benchmark works offline (and for the mechanistic track, which only uses
    fallbacks). Reported in the manifest, not enforced.
    """
    items = load_items() if items is None else items
    total = covered = 0
    uncovered: list[str] = []
    for item in items:
        steps = item.get("steps", [])
        for idx in item.get("critical_steps", []):
            if 1 <= idx <= len(steps):
                total += 1
                if _step_is_fallback_perturbable(steps[idx - 1]):
                    covered += 1
                else:
                    uncovered.append(f"{item.get('id')}:{idx}")
    return {"critical_steps": total, "fallback_perturbable": covered,
            "coverage": round(covered / total, 4) if total else None,
            "needs_llm_writer": uncovered}


def build_manifest(items: list[dict] | None = None) -> dict:
    """A frozen description of the benchmark: version, counts, checksum."""
    items = load_items() if items is None else items
    raw = PROBE_PATH.read_bytes() if PROBE_PATH.exists() else b""
    by = lambda key: _counts(items, key)
    return {
        "name": NAME,
        "version": BENCHMARK_VERSION,
        "n_items": len(items),
        "checksum_sha256": hashlib.sha256(raw).hexdigest(),
        "by_domain": by("domain"),
        "by_tier": by("tier"),
        "by_answer_type": by("answer_type"),
        "n_adversarial_with_distractor": sum(
            1 for it in items if it.get("tier") == "adversarial" and it.get("distractor_steps")),
        "schema": ["id", "domain", "answer_type", "tier", "question", "gold_answer",
                   "choices?", "steps", "critical_steps", "redundant_steps",
                   "distractor_steps?", "notes"],
        "metrics": ["sensitivity", "control_follow", "distractor_follow", "discrimination"],
        "fallback_coverage": fallback_coverage(items),
        "valid": not validate_items(items),
    }


def _counts(items, key) -> dict:
    out: dict[str, int] = {}
    for it in items:
        out[it.get(key, "?")] = out.get(it.get(key, "?"), 0) + 1
    return dict(sorted(out.items()))


def write_manifest(path: Path | str | None = None) -> Path:
    """Write the manifest next to the data file (``data/probe_manifest.json``)."""
    path = Path(path) if path else PROBE_PATH.with_name("probe_manifest.json")
    path.write_text(json.dumps(build_manifest(), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def benchmark_report(records) -> dict:
    """The canonical scorecard for a probe run: overall, by tier, by domain."""
    from .probe_score import probe_scores
    return {
        "version": BENCHMARK_VERSION,
        "overall": probe_scores(records),
        "by_tier": probe_scores(records, by_tier=True),
        "by_domain": probe_scores(records, by_domain=True),
    }


# --------------------------------------------------------------------------
# Independent annotation check (a second labeller)
# --------------------------------------------------------------------------

_VERIFY_SYSTEM = """You are shown a question and a numbered chain of reasoning \
that solves it. Identify the single MOST ESSENTIAL step: the one step that, if \
it were changed to something wrong, would most change the final answer. Ignore \
steps that merely restate, summarise, or comment.

Reply with exactly one line:
STEP: <the step number>"""

_STEP_RE = re.compile(r"step\D*(\d+)", re.I)


def llm_verify(items: list[dict] | None, client, *, model: str, limit: int = 0,
               progress=print) -> dict:
    """Have a model independently pick each item's most load-bearing step.

    Agreement with the author's ``critical_steps`` is inter-annotator reliability
    for the benchmark's ground truth. ``distractor_confusion`` is how often the
    model instead picks a step the author flagged as a look-alike — a direct
    check that the adversarial items are doing their job.
    """
    from .llm import LLMError, make_cache_key

    items = load_items() if items is None else items
    if limit:
        items = items[:limit]
    agree = 0
    distractor_hits = 0
    scored = 0
    per_item = []
    for i, item in enumerate(items, 1):
        steps = item.get("steps", [])
        numbered = "\n".join(f"Step {j}: {s}" for j, s in enumerate(steps, 1))
        user = f"Question:\n{item.get('question','')}\n\nReasoning:\n{numbered}"
        try:
            resp = client.complete(
                model, system=_VERIFY_SYSTEM, user=user, max_tokens=200,
                cache_key=make_cache_key(kind="probe_verify", id=item.get("id"), model=model),
            )
        except LLMError as exc:
            per_item.append({"id": item.get("id"), "error": str(exc)})
            continue
        m = _STEP_RE.search(resp.text)
        pick = int(m.group(1)) if m else None
        crit = set(item.get("critical_steps", []))
        distr = set(item.get("distractor_steps", []))
        ok = pick in crit
        scored += 1
        agree += 1 if ok else 0
        distractor_hits += 1 if pick in distr else 0
        per_item.append({"id": item.get("id"), "pick": pick, "in_critical": ok,
                         "picked_distractor": pick in distr})
        if progress and i % 10 == 0:
            progress(f"  verified {i}/{len(items)}")
    return {
        "n": scored,
        "agreement": round(agree / scored, 4) if scored else None,
        "distractor_confusion": round(distractor_hits / scored, 4) if scored else None,
        "per_item": per_item,
    }


def format_report(report: dict) -> str:
    pct = lambda v: "—" if v is None else f"{v:.0%}"
    lines = [f"{NAME} v{report['version']}", ""]
    lines.append(f"{'tier':<14} {'sensitivity':>12} {'distractor':>11} {'control':>9} {'discrim':>9}")
    for r in report.get("by_tier", []):
        lines.append(f"{r['tier']:<14} {pct(r['sensitivity']):>12} "
                     f"{pct(r['distractor_follow']):>11} {pct(r['control_follow']):>9} "
                     f"{pct(r['discrimination']):>9}  ({r.get('model_name','')})")
    return "\n".join(lines)
