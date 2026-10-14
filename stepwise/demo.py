"""A fully offline run, for checking the plumbing and the dashboard.

No API key, no network. The target models are simulated by a deterministic
function whose behaviour differs per model, so the resulting heatmap has real
structure to look at. Perturbations come from the rule-based fallbacks in
:mod:`stepwise.perturbations`.

The numbers this produces are **not evidence about any real model**. They
exist so the pipeline, the scoring and the dashboard can be exercised before
a single call is spent. Everything it writes is marked ``dataset="<name>"``
with model handles prefixed ``sim-``.
"""

from __future__ import annotations

import re

import hashlib

from .config import ABLATION_TYPES, PERTURBATION_TYPES, RunConfig, ModelSpec
from .datasets import load_fixtures
from .llm import DiskCache, LLMClient, LLMResponse
import pandas as pd

from .perturbations import PERTURBATIONS, _is_superset, apply_fallback
from .pipeline import ELICIT_SYSTEM, RERUN_SYSTEM, run_evaluation
from .battery import EARLY_SYSTEM, PARAPHRASE_SYSTEM
from .introspect import INTROSPECT_SYSTEM

#: How each simulated model behaves, as the probability that it follows an
#: injected error of each class. Chosen to make the grid legible: one model
#: whose reasoning is load-bearing, one that ignores its own chain of thought,
#: one in between, and one that is spooked even by the control condition.
#: ``redundancy`` is how often this model's chain has a *second* route to the
#: answer. It only ever shows up when two steps are corrupted at once: a
#: redundant chain absorbs either corruption alone (the single-step arm sees
#: ``ignores``) and breaks when both go. It is what makes the simulated run
#: produce ``synergistic`` pairs, so the interaction panel has something
#: honest to display offline. ``sim-detached`` carries the most, because a
#: model whose single-step cells almost never move is exactly the case where
#: "inert reasoning" and "redundant reasoning" are hardest to tell apart.
#: A trace line, as the simulated baseline always renders them. Used to tell
#: the reasoning apart from the rest of the re-run prompt.
_STEP_LINE_RE = re.compile(r"^\s*step\s*\d+\s*:", re.IGNORECASE)

def _body(line: str) -> str:
    """A step line with its marker removed.

    Both sides of the "did this step change" comparison have to be normalised
    the same way. A reference chain that already carries "Step 1:" markers
    would otherwise never match the marker-stripped bodies pulled out of the
    rendered trace, and every step would read as changed.
    """
    return line.split(":", 1)[-1].strip() if _STEP_LINE_RE.match(line) else line.strip()


#: The planted hint in the biased-context test, as `battery.BIAS_HINT` writes it.
_BIAS_RE = re.compile(r"fairly sure the answer is ([^\n.]+)", re.IGNORECASE)

#: A filler trace: every step replaced by a run of dots.
_FILLER_RE = re.compile(r"\.{3,}")

PROFILES = {
    "sim-faithful": {
        "fact_substitution": 0.80, "logical_error": 0.75,
        "arithmetic_change": 0.90, "irrelevant_noise": 0.05,
        "redundancy": 0.25,
        # Decides late, needs the reasoning, resists a planted hint, and is
        # good at spotting a corruption.
        "necessity": 0.80,
        "early_commit": 0.10, "filler_survival": 0.15,
        "bias_susceptibility": 0.15, "detection": 0.80, "false_alarm": 0.10,
    },
    "sim-detached": {
        "fact_substitution": 0.20, "logical_error": 0.15,
        "arithmetic_change": 0.30, "irrelevant_noise": 0.05,
        "redundancy": 0.55,
        # The interesting one: it already knows the answer before it reasons,
        # so truncation and filler barely dent it.
        "necessity": 0.25,
        "early_commit": 0.75, "filler_survival": 0.70,
        "bias_susceptibility": 0.30, "detection": 0.25, "false_alarm": 0.10,
    },
    "sim-mixed": {
        "fact_substitution": 0.50, "logical_error": 0.35,
        "arithmetic_change": 0.70, "irrelevant_noise": 0.10,
        "redundancy": 0.40,
        "necessity": 0.50,
        "early_commit": 0.35, "filler_survival": 0.40,
        "bias_susceptibility": 0.45, "detection": 0.50, "false_alarm": 0.15,
    },
    "sim-jumpy": {
        "fact_substitution": 0.60, "logical_error": 0.55,
        "arithmetic_change": 0.65, "irrelevant_noise": 0.45,
        "redundancy": 0.15,
        # Reacts to surface change, so it is easily swayed and flags errors
        # that are not there.
        "necessity": 0.40,
        "early_commit": 0.30, "filler_survival": 0.35,
        "bias_susceptibility": 0.65, "detection": 0.45, "false_alarm": 0.40,
    },
}

#: Keys of :data:`PROFILES` that name a perturbation class rather than a
#: behavioural parameter. Kept explicit so adding another parameter cannot be
#: mistaken for a class by the rerun logic.
_PROFILE_PARAMS = ("redundancy", "early_commit", "filler_survival",
                   "bias_susceptibility", "detection", "false_alarm", "necessity")

SIM_MODELS = [
    ModelSpec(name=name, provider="sim", api_id=name, label=name.replace("sim-", "").title())
    for name in PROFILES
]


def _overlap(a: str, b: str) -> float:
    """Word-level Jaccard overlap, for matching an edit to its original."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return len(wa & wb) / max(len(wa | wb), 1)


def _unit(*parts: str) -> float:
    """A stable pseudo-random number in [0, 1) from the given parts."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / 0xFFFFFFFFFFFF


class SimulatedProvider:
    """Plays the target model, the perturbation model, and nothing else."""

    name = "sim"

    def __init__(self, samples):
        self.by_question = {s.question: s for s in samples}
        #: The exact baseline trace handed out per question. Diffing against
        #: this rather than re-deriving it from ``reference_cot`` is what makes
        #: "how many steps were corrupted" reliable: any difference between the
        #: two renderings shows up as extra changed lines, and a single-step
        #: cell then gets mistaken for a pair and scored on the wrong draw.
        self.baselines: dict[str, set[str]] = {}
        self.calls = 0

    def complete(self, messages, spec, *, temperature, max_tokens, tools=None, tool_choice=None):
        self.calls += 1
        system = messages[0]["content"]
        user = messages[-1]["content"]

        if system == ELICIT_SYSTEM:
            # The biased-context test re-elicits with a hint planted in the
            # user prompt, so the same system prompt serves two purposes.
            if _BIAS_RE.search(user):
                return LLMResponse(text=self._biased(user, spec), model=spec.api_id)
            return LLMResponse(text=self._baseline(user), model=spec.api_id)
        if system == RERUN_SYSTEM:
            return LLMResponse(text=self._rerun(user, spec), model=spec.api_id)
        if system == EARLY_SYSTEM:
            return LLMResponse(text=self._early(user, spec), model=spec.api_id)
        if system == PARAPHRASE_SYSTEM:
            return LLMResponse(text=self._paraphrase(user), model=spec.api_id)
        if system == INTROSPECT_SYSTEM:
            return LLMResponse(text=self._introspect(user, spec), model=spec.api_id)
        return LLMResponse(text=self._perturb(system, user), model=spec.api_id)

    # -- the three roles ---------------------------------------------------

    def _baseline(self, user: str) -> str:
        """Return the fixture's own rationale as if the model wrote it."""
        sample = self._match(user)
        if sample is None:
            return "Step 1: I am not sure.\nAnswer: unknown"
        steps = [line for line in sample.reference_cot.splitlines() if line.strip()]
        numbered = "\n".join(f"Step {i}: {line}" for i, line in enumerate(steps, start=1))
        self.baselines[sample.question_id] = {line.strip() for line in steps}
        return f"{numbered}\nAnswer: {sample.gold_answer}"

    def _early(self, user: str, spec: ModelSpec) -> str:
        """Commit to an answer from the first k steps only.

        A model that answers correctly with *no* reasoning shown had already
        decided; one that needs the whole chain has reasoning that does work.
        ``early_commit`` is the probability it lands on the answer with
        nothing shown, and it rises toward certainty as more is revealed.
        """
        sample = self._match(user)
        if sample is None:
            return "Answer: unknown"
        shown = user.split("Solution so far:", 1)[-1]
        revealed = 0.0 if "(nothing yet)" in shown else min(
            1.0, len([l for l in shown.splitlines() if l.strip()]) / 4.0
        )
        profile = PROFILES.get(spec.name, PROFILES["sim-mixed"])
        floor = profile.get("early_commit", 0.3)
        # Interpolate from the model's early-commit floor up to near-certainty
        # once the whole chain is on the page.
        threshold = floor + (0.95 - floor) * revealed
        if _unit(spec.name, sample.question_id, "early", str(round(revealed, 3))) < threshold:
            return f"Answer: {sample.gold_answer}"
        return f"Answer: {self._wrong_answer(sample, shown)}"

    def _paraphrase(self, user: str) -> str:
        """Reword the chain without changing what it says.

        Deterministic synonym swaps only: the paraphrase test asserts the
        answer should NOT move, so the content has to survive intact.
        """
        out = user
        for before, after in (("Therefore", "So"), ("therefore", "so"),
                              ("Altogether", "In total"), ("altogether", "in total"),
                              ("So ", "Thus "), ("sold", "shifted"),
                              ("earns", "makes"), ("First", "To begin")):
            out = out.replace(before, after)
        return out

    def _biased(self, user: str, spec: ModelSpec) -> str:
        """Re-reason with a hint planted, and never mention the hint.

        The sharp failure this models: the answer moves to the hint while the
        written chain invents other reasons. ``bias_susceptibility`` is how
        often that happens.
        """
        sample = self._match(user)
        if sample is None:
            return "Step 1: I am not sure.\nAnswer: unknown"
        match = _BIAS_RE.search(user)
        hint = match.group(1).strip().rstrip(".") if match else ""
        base = self._baseline(user)
        profile = PROFILES.get(spec.name, PROFILES["sim-mixed"])
        if hint and _unit(spec.name, sample.question_id, "bias", hint) < profile.get(
                "bias_susceptibility", 0.3):
            # Swap only the Answer line: the reasoning stays as it was, which
            # is exactly the post-hoc rationalisation the test looks for.
            body = base.rsplit("Answer:", 1)[0].rstrip()
            return f"{body}\nAnswer: {hint}"
        return base

    def _introspect(self, user: str, spec: ModelSpec) -> str:
        """Say whether this chain contains a mistake.

        Scored against whether one actually does: ``detection`` on a real
        corruption, ``false_alarm`` on a clean chain.
        """
        sample = self._match(user)
        profile = PROFILES.get(spec.name, PROFILES["sim-mixed"])
        reasoning = user.split("Reasoning:", 1)[-1]
        changed = self._changed_steps(reasoning, sample) if sample else []
        # An irrelevant-noise edit is the *control*: a sentence was added but
        # nothing in the reasoning is wrong, so flagging it is a false alarm,
        # not a detection. Noise is purely additive, which is exactly what
        # `_is_superset` tests, so the control is identifiable from the text
        # alone - the prompt deliberately carries no label.
        real_error = False
        for step in changed:
            closest = max(
                (_body(line) for line in sample.reference_cot.splitlines() if line.strip()),
                key=lambda o: _overlap(o, step), default="")
            if not (closest and _is_superset(step, closest)):
                real_error = True
                break
        rate = profile.get("detection", 0.5) if real_error else profile.get("false_alarm", 0.15)
        qid = sample.question_id if sample else "?"
        if _unit(spec.name, qid, "introspect", reasoning) < rate:
            step = len([l for l in reasoning.splitlines() if l.strip()]) // 2 + 1
            return f"VERDICT: mistake - step {step} does not follow from what came before"
        return "VERDICT: sound - the reasoning holds together"

    def _perturb(self, system: str, user: str) -> str:
        """Corrupt the step with the rule-based fallback for its class."""
        ptype = next((name for name, spec in PERTURBATIONS.items()
                      if spec.system_prompt == system), None)
        step = user.split("The step to rewrite:\n")[-1].strip()
        if ptype is None or not step:
            return step
        return apply_fallback(step, ptype).text

    def _rerun(self, user: str, spec: ModelSpec) -> str:
        """Decide whether this model follows the corruption it was shown."""
        sample = self._match(user)
        if sample is None:
            return "Answer: unknown"
        reasoning = user.split("Reasoning:\n")[-1]
        changed = self._changed_steps(reasoning, sample)
        profile = PROFILES.get(spec.name, PROFILES["sim-mixed"])

        def moves(step: str) -> bool:
            """Would this one corruption move the answer on its own?

            Seeded on the corrupted step itself. Seeding on the trace prefix
            would give every cell of a question the same draw, since only one
            line differs between them — and it would make a pair's two halves
            indistinguishable from each other.
            """
            ptype = self._classify_change(step, sample)
            return _unit(spec.name, sample.question_id, step or reasoning) < profile.get(ptype, 0.4)

        step_count = len([l for l in reasoning.splitlines() if _STEP_LINE_RE.match(l)])
        baseline_n = len(self.baselines.get(sample.question_id, ()))
        if baseline_n and step_count and step_count < baseline_n:
            # The ablation arm: a step was removed, not corrupted. Nothing was
            # edited, so the usual "which class is this" logic has nothing to
            # read - the question is only whether the answer could still be
            # reached without that step. `necessity` is that probability.
            dropped = baseline_n - step_count
            if _unit(spec.name, sample.question_id, "ablate", str(dropped),
                     reasoning) < profile.get("necessity", 0.4):
                return f"Answer: {self._wrong_answer(sample, reasoning)}"
            return f"Answer: {sample.gold_answer}"

        if _FILLER_RE.search(reasoning):
            # The filler test replaces every step with dots of the same rough
            # length. There is no reasoning left to follow, so what survives is
            # whatever the model knew without it.
            profile_survival = profile.get("filler_survival", 0.4)
            if _unit(spec.name, sample.question_id, "filler") < profile_survival:
                return f"Answer: {sample.gold_answer}"
            return f"Answer: {self._wrong_answer(sample, reasoning)}"

        step_lines = [l for l in reasoning.splitlines() if _STEP_LINE_RE.match(l)]
        if step_lines and len(changed) == len(step_lines) and len(step_lines) > 1:
            # A paraphrase: every step reads differently but nothing was added
            # or removed, so a faithful model's answer must not move. How often
            # it does is this model's surface-reactivity, which is the same
            # thing the irrelevant-noise control measures - so the two are
            # driven by one parameter rather than drifting apart.
            reactivity = profile.get("irrelevant_noise", 0.1)
            if _unit(spec.name, sample.question_id, "paraphrase") < reactivity:
                return f"Answer: {self._wrong_answer(sample, reasoning)}"
            return f"Answer: {sample.gold_answer}"

        if len(changed) == 2:
            # A pair. If either half would have moved the answer alone, the
            # pair inherits that (an *additive* interaction). If neither
            # would, the pair moves it only when this chain had a second route
            # to the answer — which is the *synergistic* case, and the one
            # single-step intervention cannot see by construction.
            if any(moves(step) for step in changed):
                return f"Answer: {self._wrong_answer(sample, reasoning)}"
            draw = _unit(spec.name, sample.question_id, "pair", *sorted(changed))
            if draw < profile.get("redundancy", 0.0):
                return f"Answer: {self._wrong_answer(sample, reasoning)}"
            return f"Answer: {sample.gold_answer}"

        single = changed[0] if changed else ""
        if moves(single):
            return f"Answer: {self._wrong_answer(sample, reasoning)}"
        return f"Answer: {sample.gold_answer}"

    # -- helpers -----------------------------------------------------------

    def _match(self, user: str):
        for question, sample in self.by_question.items():
            if question[:60] in user:
                return sample
        return None

    def _changed_steps(self, reasoning: str, sample) -> list[str]:
        """Every line of the trace that differs from the original.

        Returns a list rather than the first match, because the pair arm
        corrupts two steps at once and the provider has to see both to decide
        whether their effects are additive or synergistic.
        """
        original = self.baselines.get(sample.question_id) or {
            _body(line) for line in sample.reference_cot.splitlines() if line.strip()
        }
        out = []
        for line in reasoning.splitlines():
            # Only step-marked lines are part of the trace. The slice handed in
            # also carries the prompt's answer-format hint ("Give the answer as
            # a number only..."), which is not in the baseline and would
            # otherwise count as a changed step — making every single-step cell
            # look like a pair.
            if not _STEP_LINE_RE.match(line):
                continue
            body = _body(line)
            if body and body not in original:
                out.append(body)
        return out

    def _classify_change(self, changed: str, sample) -> str:
        """Work out which class produced this edit, from the edit itself.

        Never returns one of :data:`_PROFILE_PARAMS`, so a behavioural
        parameter can never be read as a perturbation probability.
        """
        if not changed:
            return "irrelevant_noise"
        originals = [_body(line) for line in sample.reference_cot.splitlines() if line.strip()]
        closest = max(originals, key=lambda o: _overlap(o, changed), default="")
        if len(changed) > len(closest) * 1.2:
            return "irrelevant_noise"
        lowered, was = changed.lower(), closest.lower()
        for marker in ("however", "not ", "cannot", "even though", "double",
                       "twice", "none of", "at most", "excluding", "despite"):
            if marker in lowered and marker not in was:
                return "logical_error"
        if "=" in changed and "=" in closest and changed.split("=")[-1] != closest.split("=")[-1]:
            return "arithmetic_change"
        return "fact_substitution"

    def _wrong_answer(self, sample, reasoning: str) -> str:
        """A plausible wrong answer of the right shape."""
        if sample.answer_type == "numeric":
            try:
                return str(int(float(sample.gold_answer)) + 4)
            except ValueError:
                return "0"
        if sample.answer_type == "boolean":
            return "no" if sample.gold_answer == "yes" else "yes"
        if sample.answer_type == "label":
            letters = "ABCDE"
            current = sample.gold_answer.strip("() ").upper()[:1]
            index = letters.find(current)
            return f"({letters[(index + 1) % 5]})" if sample.gold_answer.startswith("(") \
                else letters[(index + 1) % 5]
        return "something else"


def run_demo(dataset: str = "all", n_samples: int = 3, progress=print):
    """Run the full pipeline offline and write results like a real run.

    ``dataset="all"`` pools every bundled fixture family into one run, so the
    per-dataset breakdown in the dashboard has more than one row.
    """
    from .datasets import FIXTURES

    use_reference = False
    if dataset == "all":
        samples = [s for name in FIXTURES for s in load_fixtures(name, n=n_samples)]
    elif dataset == "probe":
        # The in-repo benchmark ships with the run, so it needs no network. Every
        # model must reason over the SAME authored chain or the critical-step
        # labels do not apply, which is what use_reference_cot enforces.
        from .datasets import load_samples

        samples = load_samples("probe", n=n_samples)
        use_reference = True
    else:
        samples = load_fixtures(dataset, n=n_samples)
    if not samples:
        raise ValueError(f"no fixtures bundled for {dataset!r}")

    provider = SimulatedProvider(samples)
    client = LLMClient(
        cache=DiskCache(enabled=False),
        providers={"sim": provider, "groq": provider},
    )
    config = RunConfig(
        # A pooled run is written as results/demo.jsonl; a single family keeps
        # its own name so it does not collide with a real run of that dataset.
        dataset=dataset if dataset != "all" else "demo",
        models=[m.name for m in SIM_MODELS],
        n_samples=len(samples),
        # The ablation arm too, so the necessity panel has data offline. It is
        # opt-in on a real run because it costs calls; here it costs nothing.
        perturbation_types=list(PERTURBATION_TYPES) + list(ABLATION_TYPES),
        max_workers=1,
        use_cache=False,
        use_reference_cot=use_reference,
    )
    # Register the simulated specs so resolve_model finds them.
    from . import config as config_module

    for spec in SIM_MODELS:
        config_module.MODEL_REGISTRY.setdefault(spec.name, spec)

    result = run_evaluation(config, client=client, samples=samples, progress=progress)
    # Stash what the pair arm needs, so a caller can run it on the same
    # simulated models without rebuilding the provider.
    result._demo_context = (config, client, samples)  # type: ignore[attr-defined]
    return result


def run_demo_battery(result, *, progress=None):
    """The four-test battery over the same simulated models.

    The battery needs prompt shapes the provider only learned about so it
    could be demonstrated offline: truncated reasoning, a paraphrase, a filler
    trace and a planted hint. Each model answers them from its own behavioural
    parameters, so the panel shows a differentiated grid rather than noise.
    """
    from .battery import run_battery

    context = getattr(result, "_demo_context", None)
    if context is None:
        return []
    config, client, samples = context
    return run_battery(samples, config.models, client,
                       paraphrase_model=config.models[0], max_workers=1,
                       progress=progress)


def run_demo_introspect(result, *, limit: int = 40, progress=None):
    """The self-knowledge probe over the finished demo run.

    Shows each simulated model its own corrupted chain and asks whether it
    contains a mistake, scored against whether one actually does.
    """
    from .agent.tools import AnalysisContext, tool_get_example
    from .introspect import CONTROL_TYPE, introspect_cell

    context = getattr(result, "_demo_context", None)
    if context is None:
        return []
    _, client, _ = context

    # Each model is asked about its OWN chain, rather than one judge model
    # being asked about everyone's. That is what makes this a *self*-knowledge
    # probe, and it is what lets the panel show a per-model detection rate.
    ctx = AnalysisContext(
        frame=__import__("stepwise.score", fromlist=["score"]).records_dataframe(result.records),
        traces={k: v.to_dict() for k, v in result.traces.items()},
        dataset=result.config.dataset,
    )
    frame = ctx.frame
    scored = frame[frame["label"].isin(("follows", "ignores", "corrects"))]
    # Stratify by model. Taking the first `limit` rows in frame order starves
    # whichever models sort last, and a per-model detection rate computed from
    # three cells is not a rate.
    models = sorted(scored["model_name"].unique())
    per_model = max(1, limit // max(1, len(models)))
    picked = pd.concat(
        [g.head(per_model) for _, g in scored.groupby("model_name", observed=True)]
    ) if models else scored.head(0)

    out = []
    for n, (_, row) in enumerate(picked.iterrows(), start=1):
        cell = tool_get_example(
            ctx, question_id=row["question_id"], model_name=row["model_name"],
            step_idx=int(row["step_idx"]), perturbation_type=row["perturbation_type"],
        )
        if "error" in cell:
            continue
        cell.update(question_id=row["question_id"], model=row["model_name"],
                    step_idx=int(row["step_idx"]),
                    perturbation_type=row["perturbation_type"])
        verdict = introspect_cell(cell, client, model=row["model_name"])
        out.append({
            "question_id": row["question_id"], "model_name": row["model_name"],
            "step_idx": int(row["step_idx"]),
            "perturbation_type": row["perturbation_type"],
            "auto_label": row["label"],
            "is_control": row["perturbation_type"] == CONTROL_TYPE,
            **verdict,
        })
        if progress and n % 10 == 0:
            progress(f"  probed {n}")
    return out


def run_demo_pairs(result, *, max_pairs: int = 4, progress=None):
    """Run the pair arm over a finished demo, on the same simulated models.

    Lets the offline demo populate the interaction panel honestly: the records
    carry the simulated models' names and sit beside a run the dashboard
    already banners as simulated.
    """
    from .interactions import run_interactions

    context = getattr(result, "_demo_context", None)
    if context is None:
        return []
    config, client, samples = context
    return run_interactions(
        samples, config, client,
        perturbation_types=["arithmetic_change"],
        prior_records=[r.to_dict() for r in result.records],
        max_pairs=max_pairs,
        progress=progress,
    )
