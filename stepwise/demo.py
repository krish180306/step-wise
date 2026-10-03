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

import hashlib

from .config import RunConfig, ModelSpec
from .datasets import load_fixtures
from .llm import DiskCache, LLMClient, LLMResponse
from .perturbations import PERTURBATIONS, apply_fallback
from .pipeline import ELICIT_SYSTEM, RERUN_SYSTEM, run_evaluation

#: How each simulated model behaves, as the probability that it follows an
#: injected error of each class. Chosen to make the grid legible: one model
#: whose reasoning is load-bearing, one that ignores its own chain of thought,
#: one in between, and one that is spooked even by the control condition.
PROFILES = {
    "sim-faithful": {
        "fact_substitution": 0.80, "logical_error": 0.75,
        "arithmetic_change": 0.90, "irrelevant_noise": 0.05,
    },
    "sim-detached": {
        "fact_substitution": 0.20, "logical_error": 0.15,
        "arithmetic_change": 0.30, "irrelevant_noise": 0.05,
    },
    "sim-mixed": {
        "fact_substitution": 0.50, "logical_error": 0.35,
        "arithmetic_change": 0.70, "irrelevant_noise": 0.10,
    },
    "sim-jumpy": {
        "fact_substitution": 0.60, "logical_error": 0.55,
        "arithmetic_change": 0.65, "irrelevant_noise": 0.45,
    },
}

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
        self.calls = 0

    def complete(self, messages, spec, *, temperature, max_tokens, tools=None, tool_choice=None):
        self.calls += 1
        system = messages[0]["content"]
        user = messages[-1]["content"]

        if system == ELICIT_SYSTEM:
            return LLMResponse(text=self._baseline(user), model=spec.api_id)
        if system == RERUN_SYSTEM:
            return LLMResponse(text=self._rerun(user, spec), model=spec.api_id)
        return LLMResponse(text=self._perturb(system, user), model=spec.api_id)

    # -- the three roles ---------------------------------------------------

    def _baseline(self, user: str) -> str:
        """Return the fixture's own rationale as if the model wrote it."""
        sample = self._match(user)
        if sample is None:
            return "Step 1: I am not sure.\nAnswer: unknown"
        steps = [line for line in sample.reference_cot.splitlines() if line.strip()]
        numbered = "\n".join(f"Step {i}: {line}" for i, line in enumerate(steps, start=1))
        return f"{numbered}\nAnswer: {sample.gold_answer}"

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
        changed = self._changed_step(reasoning, sample)
        ptype = self._classify_change(changed, sample)
        profile = PROFILES.get(spec.name, PROFILES["sim-mixed"])
        # Seed on the corrupted step itself. Seeding on the trace prefix would
        # give every cell of a question the same draw, since only one line
        # differs between them.
        draw = _unit(spec.name, sample.question_id, changed or reasoning)
        if draw < profile.get(ptype, 0.4):
            return f"Answer: {self._wrong_answer(sample, reasoning)}"
        return f"Answer: {sample.gold_answer}"

    # -- helpers -----------------------------------------------------------

    def _match(self, user: str):
        for question, sample in self.by_question.items():
            if question[:60] in user:
                return sample
        return None

    def _changed_step(self, reasoning: str, sample) -> str:
        """The one line of the trace that differs from the original."""
        original = {line.strip() for line in sample.reference_cot.splitlines() if line.strip()}
        for line in reasoning.splitlines():
            body = line.split(":", 1)[-1].strip()
            if body and body not in original:
                return body
        return ""

    def _classify_change(self, changed: str, sample) -> str:
        """Work out which class produced this edit, from the edit itself."""
        if not changed:
            return "irrelevant_noise"
        originals = [line.strip() for line in sample.reference_cot.splitlines() if line.strip()]
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

    if dataset == "all":
        samples = [s for name in FIXTURES for s in load_fixtures(name, n=n_samples)]
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
        max_workers=1,
        use_cache=False,
    )
    # Register the simulated specs so resolve_model finds them.
    from . import config as config_module

    for spec in SIM_MODELS:
        config_module.MODEL_REGISTRY.setdefault(spec.name, spec)

    result = run_evaluation(config, client=client, samples=samples, progress=progress)
    return result
