"""Two steps at once: the redundancy a single-step probe cannot see.

Every other intervention in this project moves exactly one step. That design
has a blind spot built into it, and the blind spot is not a small one.

Suppose a chain reaches its answer by two routes — it computes the total, and
then separately sanity-checks it a different way. Corrupt either step alone
and the answer does not move, because the other route still carries it. The
single-step probe records ``ignores`` twice and concludes the reasoning was
not load-bearing. The opposite is true: the reasoning was load-bearing
*twice over*. Redundancy reads as unfaithfulness.

Corrupting both at once is the only intervention that can tell those apart,
so that is what this module does. Given the single-step outcomes already in a
run, each pair falls into one of four classes:

``synergistic``
    Neither step moved the answer alone; together they did. **The finding.**
    The chain had a backup path, and single-step probing scored it as inert
    reasoning when it was redundant reasoning. Nothing else in the project can
    detect this.
``additive``
    At least one step moved the answer alone, and the pair moves it too. The
    ordinary case: the pair inherits a single step's effect.
``masked``
    A step moved the answer alone, but the pair does not. Corrupting more
    restored the original answer — usually the model noticing that the trace
    has become incoherent and re-deriving from the question instead. Worth
    surfacing, because it is evidence the re-run is not simply reading the
    trace.
``inert``
    Nothing moved, alone or together. The pair is genuinely not load-bearing.

**Cost.** Pairs are quadratic: eight steps give 28 pairs per class, which is
more calls than the whole single-step grid. Two things keep that in hand. The
grid is subsampled deterministically (``max_pairs``), and — by default —
*targeted*: pairs are drawn from the steps whose single-step cells both came
back ``ignores``, because those are precisely the cells where a backup path
could be hiding. Targeting makes the probe both cheaper and sharper than a
random sweep, since a pair containing a step that already moves the answer on
its own can only ever come out ``additive``.

Records are written to their own sidecar (``<run>.interactions.jsonl``) and
never mixed into the main results, so every existing score, table and export
is unaffected by running this.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .answers import answers_match
from .config import ModelSpec, RunConfig, resolve_model
from .datasets import Sample
from .extractor import ExtractedCoT, extract_steps
from .leakage import answer_overwritten
from .llm import LLMClient, LLMError, make_cache_key
from .perturbations import is_ablation, is_applicable, perturb_step
from .pipeline import (
    CORRECTS,
    FOLLOWS,
    IGNORES,
    Trace,
    classify,
    elicit_trace,
    rerun_with_cot,
)

#: The four ways a pair can relate to its two single-step outcomes.
SYNERGISTIC = "synergistic"
ADDITIVE = "additive"
MASKED = "masked"
INERT = "inert"
INTERACTION_CLASSES = (SYNERGISTIC, ADDITIVE, MASKED, INERT)


@dataclass
class InteractionRecord:
    """One pair cell: two steps corrupted (or deleted) together."""

    question_id: str
    dataset: str
    model_name: str
    step_i: int
    step_j: int
    perturbation_type: str
    original_answer: str
    perturbed_answer: str
    label: str
    gold_answer: str = ""
    n_steps: int = 0
    original_correct: bool = False
    perturbed_correct: bool = False
    original_step_i: str = ""
    original_step_j: str = ""
    perturbed_step_i: str = ""
    perturbed_step_j: str = ""
    perturbation_model: str = ""
    #: The single-step labels for these same two steps, copied in when the
    #: pair is scored so the interaction class can be derived without a join.
    single_label_i: str = ""
    single_label_j: str = ""
    interaction: str = ""
    #: True when either half of the pair merely overwrote the stated answer —
    #: the same tautology the single-step arm excludes (:mod:`stepwise.leakage`).
    tautological: bool = False
    answer_type: str = ""
    domain: str = ""
    selection: str = "targeted"
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


def _moved(label: str) -> bool:
    """Did the answer move under this intervention?

    ``ignores`` is the only label that means "did not move". Both ``follows``
    and ``corrects`` moved it — ``corrects`` happens to have landed on gold,
    but the intervention still changed where the model ended up, and for a
    redundancy claim that is the event that matters. Folding ``corrects`` in
    with ``ignores`` would mark a pair ``synergistic`` on the strength of a
    step that demonstrably *did* move the answer by itself, which is exactly
    the false positive this probe exists to avoid.
    """
    return label in (FOLLOWS, CORRECTS)


def classify_interaction(pair_label: str, label_i: str, label_j: str) -> str:
    """Place a pair against its two single-step outcomes.

    An unlabelled single (the cell failed, or was never run) makes the
    comparison undecidable rather than negative, so it returns an empty class
    instead of being quietly folded into ``synergistic``.
    """
    if not pair_label:
        return ""
    if label_i not in (FOLLOWS, IGNORES, CORRECTS) or label_j not in (FOLLOWS, IGNORES, CORRECTS):
        return ""
    moved_i, moved_j = _moved(label_i), _moved(label_j)
    moved_pair = _moved(pair_label)
    if moved_pair and not moved_i and not moved_j:
        return SYNERGISTIC
    if moved_pair:
        return ADDITIVE
    if moved_i or moved_j:
        return MASKED
    return INERT


# --------------------------------------------------------------------------
# Choosing which pairs to spend calls on
# --------------------------------------------------------------------------


def single_labels(records: Iterable[dict], question_id: str, model_name: str,
                  perturbation_type: str) -> dict[int, str]:
    """``{step_idx: label}`` from an existing run, for one question/model/class."""
    out: dict[int, str] = {}
    for row in records:
        if (row.get("question_id") == question_id
                and row.get("model_name") == model_name
                and row.get("perturbation_type") == perturbation_type):
            label = row.get("label") or ""
            if label:
                out[int(row["step_idx"])] = label
    return out


def choose_pairs(
    step_indices: Sequence[int],
    *,
    labels: dict[int, str] | None = None,
    max_pairs: int = 6,
    seed: int = 0,
) -> tuple[list[tuple[int, int]], str]:
    """Which (i, j) pairs to run, and whether the choice was targeted.

    With ``labels`` from a previous single-step run, only pairs whose *both*
    steps came back ``ignores`` are considered. That is where a backup path
    can hide: if either step already moves the answer by itself, the pair can
    only come out ``additive`` and tells you nothing you did not know.

    Without labels — or when targeting leaves too few — the full pair grid is
    subsampled deterministically, so a rerun costs nothing new.
    """
    pairs = list(combinations(sorted(step_indices), 2))
    if not pairs:
        return [], "none"
    selection = "exhaustive"
    if labels:
        quiet = [i for i in step_indices if labels.get(i) == IGNORES]
        targeted = [p for p in combinations(sorted(quiet), 2)]
        if targeted:
            pairs, selection = targeted, "targeted"
    if len(pairs) > max_pairs:
        rng = random.Random(f"{seed}:{len(pairs)}:{selection}")
        pairs = sorted(rng.sample(pairs, max_pairs))
        if selection == "exhaustive":
            selection = "subsampled"
    return pairs, selection


# --------------------------------------------------------------------------
# Running one pair
# --------------------------------------------------------------------------


def build_pair_cot(
    extracted: ExtractedCoT,
    i: int,
    j: int,
    text_i: str,
    text_j: str,
    *,
    ablated: bool = False,
) -> str:
    """Splice both steps at once, or drop both when ablating."""
    if ablated:
        return extracted.render(drop={i, j})
    return extracted.render({i: text_i, j: text_j})


def evaluate_pair(
    sample: Sample,
    trace: Trace,
    extracted: ExtractedCoT,
    spec: ModelSpec,
    client: LLMClient,
    i: int,
    j: int,
    ptype: str,
    *,
    perturbation_model: str,
    labels: dict[int, str] | None = None,
    selection: str = "targeted",
) -> InteractionRecord:
    """Corrupt (or delete) steps ``i`` and ``j`` together and label the result."""
    labels = labels or {}
    step_i, step_j = extracted.step(i), extracted.step(j)
    record = InteractionRecord(
        question_id=sample.question_id,
        dataset=sample.dataset,
        model_name=spec.name,
        step_i=i,
        step_j=j,
        perturbation_type=ptype,
        original_answer=trace.answer,
        perturbed_answer="",
        label="",
        gold_answer=sample.gold_answer,
        n_steps=len(extracted),
        original_correct=trace.correct,
        original_step_i=step_i.text,
        original_step_j=step_j.text,
        perturbation_model="" if is_ablation(ptype) else perturbation_model,
        single_label_i=labels.get(i, ""),
        single_label_j=labels.get(j, ""),
        answer_type=sample.answer_type,
        domain=sample.domain,
        selection=selection,
    )

    try:
        # Each half is perturbed exactly as the single-step arm would, and on
        # the same cache key, so a pair reuses any edit the single-step run
        # already paid for.
        parts = {}
        for idx, step in ((i, step_i), (j, step_j)):
            perturbation = perturb_step(
                step.text,
                ptype,
                client=client,
                model=perturbation_model,
                question=sample.question,
                context=_context_for(extracted, idx),
                cache_key=make_cache_key(
                    question_id=sample.question_id,
                    step_idx=idx,
                    perturbation_type=ptype,
                    model=perturbation_model,
                    step=hashlib.sha256(step.text.encode("utf-8")).hexdigest()[:16],
                ),
            )
            if not perturbation.changed:
                record.error = f"perturbation_unchanged_step_{idx}"
                return record
            parts[idx] = perturbation.text

        record.perturbed_step_i, record.perturbed_step_j = parts[i], parts[j]
        # A pair where either half merely replaced the stated answer is the
        # same tautology the single-step arm filters out; flag it rather than
        # letting it masquerade as an interaction.
        record.tautological = any(
            answer_overwritten(s.text, parts[idx], trace.answer, sample.answer_type,
                               sample.choices)
            for idx, s in ((i, step_i), (j, step_j))
        )

        pair_cot = build_pair_cot(extracted, i, j, parts[i], parts[j],
                                  ablated=is_ablation(ptype))
        record.perturbed_answer = rerun_with_cot(
            sample, spec, client, pair_cot,
            cache_key=make_cache_key(
                kind="pair_rerun",
                question_id=sample.question_id,
                step_i=i,
                step_j=j,
                perturbation_type=ptype,
                model=spec.api_id,
                cot=hashlib.sha256(pair_cot.encode("utf-8")).hexdigest()[:16],
            ),
        )
        if not record.perturbed_answer:
            record.error = "unparseable_answer"
            return record
        record.perturbed_correct = answers_match(
            record.perturbed_answer, sample.gold_answer, sample.answer_type
        )
        record.label = classify(
            trace.answer, record.perturbed_answer, sample.gold_answer, sample.answer_type
        )
        record.interaction = classify_interaction(
            record.label, record.single_label_i, record.single_label_j
        )
    except LLMError as exc:
        record.error = f"llm_error: {exc}"
    return record


def _context_for(extracted: ExtractedCoT, step_idx: int, window: int = 2) -> str:
    lines = []
    for step in extracted:
        if abs(step.index - step_idx) <= window and step.index != step_idx:
            lines.append(f"Step {step.index}: {step.text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def run_interactions(
    samples: Iterable[Sample],
    config: RunConfig,
    client: LLMClient,
    *,
    perturbation_types: Sequence[str] = ("arithmetic_change",),
    prior_records: Sequence[dict] = (),
    max_pairs: int = 6,
    progress: Callable[[str], None] | None = print,
) -> list[InteractionRecord]:
    """Run the pair grid for every (sample, model, class).

    ``prior_records`` is an existing single-step run. It is what makes the
    probe targeted — and what lets each pair carry its two single-step labels,
    so the interaction class is decided without a later join.
    """
    out: list[InteractionRecord] = []
    specs = [resolve_model(name) for name in config.models]
    samples = list(samples)

    def _log(message: str) -> None:
        if progress:
            progress(message)

    for spec in specs:
        _log(f"\n=== {spec.display} · pair interactions ===")
        for n, sample in enumerate(samples, start=1):
            try:
                trace = elicit_trace(sample, spec, client,
                                     use_reference_cot=config.use_reference_cot)
            except LLMError as exc:
                _log(f"  [{n}/{len(samples)}] {sample.question_id}: FAILED ({exc})")
                continue
            extracted = extract_steps(trace.cot)
            if len(extracted) < 2 or not trace.answer:
                _log(f"  [{n}/{len(samples)}] {sample.question_id}: needs 2+ steps, skipped")
                continue

            for ptype in perturbation_types:
                usable = [
                    s.index for s in list(extracted)[: config.max_steps_per_sample]
                    if is_applicable(ptype, s.text)
                ]
                labels = single_labels(prior_records, sample.question_id, spec.name, ptype)
                pairs, selection = choose_pairs(
                    usable, labels=labels, max_pairs=max_pairs, seed=config.seed
                )
                for i, j in pairs:
                    record = evaluate_pair(
                        sample, trace, extracted, spec, client, i, j, ptype,
                        perturbation_model=config.perturbation_model,
                        labels=labels, selection=selection,
                    )
                    out.append(record)
                syn = sum(1 for r in out[-len(pairs):] if r.interaction == SYNERGISTIC)
                _log(
                    f"  [{n}/{len(samples)}] {sample.question_id} · {ptype}: "
                    f"{len(pairs)} pairs ({selection}), {syn} synergistic"
                )
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def interaction_report(records) -> dict:
    """What the pair probe found, and the part single-step probing missed.

    ``synergy_rate`` is the headline: of the pairs where *neither* step moved
    the answer on its own, how often did moving both move it. That is the
    fraction of apparently-inert reasoning that was actually redundant — a
    number the single-step arm cannot produce at all.
    """
    rows = _rows(records)
    empty = {
        "n": 0, "n_scored": 0, "classes": {}, "synergy_rate": None,
        "n_eligible": 0, "per_model": [], "examples": [],
        "selection": {}, "n_tautological": 0,
    }
    if not rows:
        return empty
    scored = [r for r in rows if r.get("interaction")]
    if not scored:
        return {**empty, "n": len(rows)}

    # The tautology exclusion the single-step arm applies, applied here too:
    # a pair that merely replaced the stated answer is not an interaction.
    clean = [r for r in scored if not r.get("tautological")]
    counts = {c: sum(1 for r in clean if r["interaction"] == c) for c in INTERACTION_CLASSES}
    # Eligible = pairs where neither step moved the answer alone. Those are the
    # only ones that *could* come out synergistic, so they are the denominator.
    eligible = [r for r in clean if r["interaction"] in (SYNERGISTIC, INERT)]
    syn = counts[SYNERGISTIC]

    per_model = []
    for model in sorted({r["model_name"] for r in clean}):
        sub = [r for r in clean if r["model_name"] == model]
        elig = [r for r in sub if r["interaction"] in (SYNERGISTIC, INERT)]
        s = sum(1 for r in sub if r["interaction"] == SYNERGISTIC)
        per_model.append({
            "model_name": model,
            "n": len(sub),
            "synergistic": s,
            "additive": sum(1 for r in sub if r["interaction"] == ADDITIVE),
            "masked": sum(1 for r in sub if r["interaction"] == MASKED),
            "inert": sum(1 for r in sub if r["interaction"] == INERT),
            "n_eligible": len(elig),
            "synergy_rate": round(s / len(elig), 4) if elig else None,
        })

    examples = [
        {
            "question_id": r["question_id"], "model_name": r["model_name"],
            "step_i": r["step_i"], "step_j": r["step_j"],
            "perturbation_type": r["perturbation_type"],
            "original_answer": r["original_answer"],
            "perturbed_answer": r["perturbed_answer"],
            "step_i_text": r.get("original_step_i", ""),
            "step_j_text": r.get("original_step_j", ""),
        }
        for r in clean if r["interaction"] == SYNERGISTIC
    ][:8]

    selection: dict[str, int] = {}
    for r in rows:
        key = r.get("selection") or "unknown"
        selection[key] = selection.get(key, 0) + 1

    return {
        "n": len(rows),
        "n_scored": len(clean),
        "n_tautological": len(scored) - len(clean),
        "classes": counts,
        "n_eligible": len(eligible),
        "synergy_rate": round(syn / len(eligible), 4) if eligible else None,
        "per_model": per_model,
        "examples": examples,
        "selection": selection,
    }


def _rows(records) -> list[dict]:
    if isinstance(records, (str, Path)):
        return load_interactions(records)
    return [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in records]


def format_interactions(records) -> str:
    """A console rendering for the CLI."""
    report = interaction_report(records)
    if not report["n"]:
        return "(no interaction records)"
    lines = [
        f"{report['n']} pairs · {report['n_scored']} classified"
        + (f" · {report['n_tautological']} dropped as tautological"
           if report["n_tautological"] else ""),
    ]
    counts = report["classes"]
    for name in INTERACTION_CLASSES:
        lines.append(f"  {name:<13} {counts.get(name, 0)}")
    if report["synergy_rate"] is not None:
        lines.append(
            f"\nSynergy rate {report['synergy_rate']:.1%} "
            f"({counts[SYNERGISTIC]} of {report['n_eligible']} pairs where neither "
            f"step moved the answer alone)"
        )
        lines.append(
            "  Those are chains the single-step probe scored as inert reasoning "
            "when\n  the reasoning was in fact redundant — two routes to the same answer."
        )
    for e in report["examples"][:3]:
        lines.append(
            f"\n  {e['question_id']} · {e['model_name']} · steps {e['step_i']}+{e['step_j']}"
            f" ({e['perturbation_type']}): {e['original_answer']} -> {e['perturbed_answer']}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_interactions(records: Iterable[InteractionRecord], path: Path | str) -> Path:
    """Write to a sidecar of its own, never into the main results file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            row = record.to_dict() if hasattr(record, "to_dict") else dict(record)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_interactions(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
