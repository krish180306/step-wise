"""Central configuration: paths, model registry, and run defaults.

Everything a run depends on is declared here so that swapping providers or
models never requires touching the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"
DESIGN_DIR = ROOT / "design"

for _d in (DATA_DIR, RESULTS_DIR, CACHE_DIR):
    _d.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One target model under test, or one utility model.

    ``provider`` names an entry in ``stepwise.llm.PROVIDERS``; adding an
    OpenAI/Anthropic/Ollama model is a matter of adding a row here.
    """

    name: str  # short handle used on the CLI and in records
    provider: str  # "groq" | "openai" | "anthropic"
    api_id: str  # the id the provider's API expects
    label: str = ""  # display name for the dashboard
    max_tokens: int = 1024
    temperature: float = 0.0
    supports_tools: bool = True
    #: Provider-specific request parameters, sent as-is. A tuple of pairs
    #: rather than a dict, so the spec stays hashable.
    extra: tuple[tuple[str, Any], ...] = ()

    @property
    def display(self) -> str:
        return self.label or self.name


#: Qwen 3.x thinks inside a <think> block by default. Left on, that block
#: leaks into the visible output — the extractor then segments the scratchpad
#: instead of the reasoning — and a hidden reasoning channel sits alongside
#: the stated chain of thought the experiment is perturbing. Switched off, the
#: stated chain is the only reasoning there is. Verified live on 2026-09-11:
#: 716 output tokens down to 74, no <think> block, and a 256-token re-run
#: answers in 6 tokens instead of running out mid-thought.
_NO_THINKING = (("reasoning_effort", "none"),)

#: The default lineup: four models the project's free-tier Groq key can reach
#: (checked against the live /models endpoint on 2026-09-11), each verified
#: with a live elicit-and-rerun on GSM8K. Three vendors — OpenAI's open
#: weights, Alibaba's Qwen, SDAIA's ALLaM — spanning 7B to 120B.
#:
#: The Qwen ids are preview-tier, which Groq can retire without notice: if a
#: run suddenly 404s, list the key's models before debugging anything else.
#: ALLaM has a 4K context window; HotpotQA's multi-paragraph prompts are the
#: tightest fit, so look there first if it errors.
GROQ_MODELS = [
    ModelSpec(name="gpt-oss-120b", provider="groq", api_id="openai/gpt-oss-120b", label="GPT-OSS 120B"),
    ModelSpec(name="qwen3.6-27b", provider="groq", api_id="qwen/qwen3.6-27b",
              label="Qwen 3.6 27B", extra=_NO_THINKING),
    ModelSpec(name="qwen3.8-27b", provider="groq", api_id="qwen/qwen3.8-27b",
              label="Qwen 3.8 27B", extra=_NO_THINKING),
    ModelSpec(name="allam-2-7b", provider="groq", api_id="allam-2-7b", label="ALLaM 2 7B"),
]

#: Reachable on the free tier but not in the default lineup. gpt-oss-20b calls
#: a built-in python tool on every arithmetic question tested — under low
#: reasoning effort, at higher temperature, and when told explicitly not to —
#: and Groq rejects the request because no tools were offered. It works on
#: the non-arithmetic datasets, so it stays addressable with --models.
GROQ_LIMITED_MODELS = [
    ModelSpec(name="gpt-oss-20b", provider="groq", api_id="openai/gpt-oss-20b", label="GPT-OSS 20B"),
]

#: The original Llama lineup. Both now return 404 on the free tier; kept
#: addressable so a paid key can still name them with --models.
GROQ_PAID_MODELS = [
    ModelSpec(name="llama-3.3-70b", provider="groq", api_id="llama-3.3-70b-versatile", label="Llama 3.3 70B"),
    ModelSpec(name="llama-3.1-8b", provider="groq", api_id="llama-3.1-8b-instant", label="Llama 3.1 8B"),
]

#: Examples of swapping in another provider. Unused unless named on the CLI
#: and the matching API key is set.
OTHER_PROVIDER_MODELS = [
    ModelSpec(name="gpt-4o", provider="openai", api_id="gpt-4o", label="GPT-4o"),
    ModelSpec(
        name="claude-sonnet-4-6",
        provider="anthropic",
        api_id="claude-sonnet-4-6",
        label="Claude Sonnet 4.6",
    ),
]

MODEL_REGISTRY: dict[str, ModelSpec] = {
    m.name: m
    for m in GROQ_MODELS + GROQ_LIMITED_MODELS + GROQ_PAID_MODELS + OTHER_PROVIDER_MODELS
}
# Allow addressing a model by its raw API id too ("openai/gpt-oss-120b").
MODEL_REGISTRY.update({m.api_id: m for m in MODEL_REGISTRY.values()})


def resolve_model(name: str) -> ModelSpec:
    """Look up a model by short handle or API id.

    Unknown names fall through to a Groq model spec so a newly released Groq
    id works without a code change.
    """
    key = name.strip()
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]
    return ModelSpec(name=key, provider="groq", api_id=key, label=key)


def resolve_models(names: str | list[str]) -> list[ModelSpec]:
    if isinstance(names, str):
        names = [n for n in names.split(",") if n.strip()]
    return [resolve_model(n) for n in names]


DEFAULT_TARGET_MODELS = [m.name for m in GROQ_MODELS]

#: Model that writes the perturbations. A stronger model produces cleaner,
#: more minimal edits, which matters more here than raw throughput.
PERTURBATION_MODEL = "gpt-oss-120b"

#: Model that drives the agentic faithfulness analyst.
ANALYST_MODEL = "gpt-oss-120b"


# --------------------------------------------------------------------------
# Run configuration
# --------------------------------------------------------------------------

PERTURBATION_TYPES = (
    "fact_substitution",
    "logical_error",
    "arithmetic_change",
    "irrelevant_noise",
)

#: The ablation arm: remove the step instead of corrupting it. Opt-in via
#: ``--perturbations ...,step_deletion``, and kept out of the default grid so
#: a default run's cost and contents are exactly what they always were. See
#: :data:`stepwise.perturbations.ABLATION_TYPES` for why it is scored apart.
ABLATION_TYPES = ("step_deletion",)

#: Everything ``--perturbations`` accepts.
ALL_INTERVENTION_TYPES = PERTURBATION_TYPES + ABLATION_TYPES

LABELS = ("follows", "ignores", "corrects")

DATASETS = ("gsm8k", "svamp", "aqua", "strategyqa", "hotpotqa",
            "2wikimultihop", "commonsenseqa", "bbh", "probe")


@dataclass
class RunConfig:
    dataset: str = "gsm8k"
    models: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODELS))
    n_samples: int = 50
    perturbation_types: list[str] = field(default_factory=lambda: list(PERTURBATION_TYPES))
    #: Edit sizes to run per cell. The default is one ("moderate"), which keeps a
    #: normal run the same size; the severity ladder passes several.
    severities: list[str] = field(default_factory=lambda: ["moderate"])
    perturbation_model: str = PERTURBATION_MODEL
    max_steps_per_sample: int = 8
    seed: int = 0
    use_cache: bool = True
    max_workers: int = 4
    bbh_subtask: str = "logical_deduction_five_objects"
    #: "full" shows the model the whole perturbed trace and asks for the answer
    #: it leads to; "continue" truncates after the perturbed step and lets the
    #: model write the rest of the reasoning itself.
    rerun_mode: str = "full"
    #: Use the dataset's own reference chain of thought instead of eliciting one
    #: from each model. Required for the probe set, where every model must reason
    #: over the same authored chain so the critical-step labels apply.
    use_reference_cot: bool = False
    results_path: Path | None = None

    def resolved_results_path(self) -> Path:
        if self.results_path is not None:
            return Path(self.results_path)
        return RESULTS_DIR / f"{self.dataset}.jsonl"


def api_key(provider: str) -> str | None:
    return {
        "groq": os.environ.get("GROQ_API_KEY"),
        "openai": os.environ.get("OPENAI_API_KEY"),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
    }.get(provider)
