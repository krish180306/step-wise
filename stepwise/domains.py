"""Reasoning domains, one level above datasets.

The project's original question — "which model is most faithful?" — is a
leaderboard. The sharper question is "is a model's reasoning more load-bearing
in some *kinds* of reasoning than others?", which is a claim you can test. To
ask it you need a dimension coarser than the dataset: a *domain*.

Two rules keep a domain comparison honest, and both live here:

1. A domain must span more than one dataset, or a "domain effect" is really
   just one dataset's quirks. GSM8K alone is not "arithmetic"; GSM8K plus
   SVAMP plus AQuA is. The map below is built so that, once the extra datasets
   are added, every domain has at least two sources.
2. BIG-Bench Hard is 27 unrelated tasks behind one loader, so it cannot carry
   a single domain. Each subtask is mapped individually — logical deduction is
   ``logical``, boolean expressions are ``symbolic``, and so on.
"""

from __future__ import annotations

ARITHMETIC = "arithmetic"
LOGICAL = "logical"
SYMBOLIC = "symbolic"
COMMONSENSE = "commonsense"
MULTIHOP = "multihop"
OTHER = "other"

DOMAINS = (ARITHMETIC, LOGICAL, SYMBOLIC, COMMONSENSE, MULTIHOP, OTHER)

#: Dataset → domain, for the datasets that map whole. BBH is handled per
#: subtask below. Rows for datasets not yet wired in (SVAMP, AQuA, …) are
#: listed so the intended two-sources-per-domain structure is visible.
DATASET_DOMAIN: dict[str, str] = {
    "gsm8k": ARITHMETIC,
    "svamp": ARITHMETIC,
    "aqua": ARITHMETIC,
    "strategyqa": COMMONSENSE,
    "commonsenseqa": COMMONSENSE,
    "piqa": COMMONSENSE,
    "hotpotqa": MULTIHOP,
    "2wikimultihop": MULTIHOP,
    "folio": LOGICAL,
}

#: BBH subtask → domain. Only the subtasks the project is likely to run are
#: named; anything unlisted falls through to :data:`OTHER` rather than being
#: guessed, so an unmapped subtask shows up plainly instead of silently
#: landing in the wrong bucket.
BBH_SUBTASK_DOMAIN: dict[str, str] = {
    # Logical deduction and argument structure.
    "logical_deduction_three_objects": LOGICAL,
    "logical_deduction_five_objects": LOGICAL,
    "logical_deduction_seven_objects": LOGICAL,
    "formal_fallacies": LOGICAL,
    "web_of_lies": LOGICAL,
    # Symbolic / algorithmic manipulation.
    "boolean_expressions": SYMBOLIC,
    "multistep_arithmetic_two": SYMBOLIC,
    "dyck_languages": SYMBOLIC,
    "word_sorting": SYMBOLIC,
    "tracking_shuffled_objects_three_objects": SYMBOLIC,
    "tracking_shuffled_objects_five_objects": SYMBOLIC,
    "tracking_shuffled_objects_seven_objects": SYMBOLIC,
    # Commonsense and world knowledge.
    "causal_judgement": COMMONSENSE,
    "sports_understanding": COMMONSENSE,
    "date_understanding": COMMONSENSE,
    "ruin_names": COMMONSENSE,
}


def domain_for(dataset: str, subtask: str | None = None) -> str:
    """The domain a sample belongs to.

    ``subtask`` is only consulted for BBH, where the dataset name alone does
    not determine the kind of reasoning.
    """
    dataset = (dataset or "").strip().lower()
    if dataset == "bbh":
        return BBH_SUBTASK_DOMAIN.get((subtask or "").strip().lower(), OTHER)
    return DATASET_DOMAIN.get(dataset, OTHER)
