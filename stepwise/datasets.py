"""Dataset loaders, normalised onto one schema.

Five task families, chosen because a single reasoning step in each can be
pinned down and intervened on:

===============  ==========  =================================================
Dataset          Answer      Why it is here
===============  ==========  =================================================
GSM8K            numeric     Arithmetic chains with a checkable final number.
StrategyQA       boolean     Implicit multi-hop reasoning, yes/no.
HotpotQA         span        Explicit two-hop retrieval and composition.
CommonsenseQA    label       Five-way choice with no arithmetic to lean on.
BBH              label       Hard symbolic reasoning, one subtask at a time.
===============  ==========  =================================================

Each loader returns :class:`Sample` objects. Every HF repo is tried against a
list of mirrors, because these datasets get moved between namespaces; if none
resolve, a small bundled fixture set keeps the pipeline and dashboard
runnable offline.
"""

from __future__ import annotations

import random
import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable

# Answer comparison strategies, used by the pipeline to decide whether a
# re-run answer matches the original.
NUMERIC = "numeric"
BOOLEAN = "boolean"
LABEL = "label"
SPAN = "span"


@dataclass
class Sample:
    """One evaluation item, whatever dataset it came from."""

    question_id: str
    dataset: str
    question: str
    gold_answer: str
    answer_type: str = SPAN
    choices: list[str] = field(default_factory=list)
    reference_cot: str = ""  # the dataset's own rationale, where it has one
    domain: str = ""  # reasoning domain; filled by load_samples if left blank
    meta: dict = field(default_factory=dict)

    def prompt_question(self) -> str:
        """The question as a model should see it, choices included."""
        if not self.choices:
            return self.question
        options = "\n".join(self.choices)
        return f"{self.question}\n\nOptions:\n{options}"

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "dataset": self.dataset,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "answer_type": self.answer_type,
            "choices": self.choices,
            "reference_cot": self.reference_cot,
            "domain": self.domain,
            "meta": self.meta,
        }


# --------------------------------------------------------------------------
# HuggingFace access
# --------------------------------------------------------------------------


def _try_load(candidates: Iterable[tuple], split: str):
    """Load the first HF repo that resolves.

    ``candidates`` is a sequence of ``(repo,)`` or ``(repo, config)`` tuples.
    """
    from datasets import load_dataset

    errors = []
    for candidate in candidates:
        repo, *rest = candidate
        config = rest[0] if rest else None
        for split_name in (split, "train", "validation", "test"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if config:
                        return load_dataset(repo, config, split=split_name)
                    return load_dataset(repo, split=split_name)
            except Exception as exc:  # noqa: BLE001 - report every failure together
                errors.append(f"{repo}/{config or '-'}[{split_name}]: {type(exc).__name__}")
                continue
    raise RuntimeError("could not load any of the candidate repos -> " + "; ".join(errors[:6]))


def _sample_rows(rows, n: int, seed: int):
    """Take ``n`` rows deterministically, without loading everything."""
    total = len(rows)
    if n <= 0 or n >= total:
        indices = list(range(total))
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(total), n))
    return [rows[int(i)] for i in indices]


# --------------------------------------------------------------------------
# Per-dataset loaders
# --------------------------------------------------------------------------


def load_gsm8k(n: int = 50, seed: int = 0, **_) -> list[Sample]:
    """Grade-school arithmetic word problems with human rationales."""
    rows = _sample_rows(_try_load([("openai/gsm8k", "main"), ("gsm8k", "main")], "test"), n, seed)
    samples = []
    for i, row in enumerate(rows):
        rationale, _, final = row["answer"].partition("####")
        # The dataset embeds calculator annotations like <<48/2=24>>.
        rationale = re.sub(r"<<[^>]*>>", "", rationale).strip()
        samples.append(
            Sample(
                question_id=f"gsm8k-{i:04d}",
                dataset="gsm8k",
                question=row["question"].strip(),
                gold_answer=final.strip().replace(",", ""),
                answer_type=NUMERIC,
                reference_cot=rationale,
            )
        )
    return samples


def load_strategyqa(n: int = 50, seed: int = 0, **_) -> list[Sample]:
    """Yes/no questions needing implicit multi-hop reasoning."""
    rows = _sample_rows(
        _try_load(
            [("ChilleD/StrategyQA",), ("wics/strategy-qa",), ("voidful/StrategyQA",)],
            "test",
        ),
        n,
        seed,
    )
    samples = []
    for i, row in enumerate(rows):
        answer = row.get("answer")
        facts = row.get("facts") or row.get("evidence") or []
        if isinstance(facts, list):
            facts = " ".join(str(f) for f in facts)
        samples.append(
            Sample(
                question_id=f"strategyqa-{i:04d}",
                dataset="strategyqa",
                question=str(row.get("question", "")).strip(),
                gold_answer="yes" if answer in (True, "True", "true", "yes") else "no",
                answer_type=BOOLEAN,
                reference_cot=str(facts).strip(),
            )
        )
    return samples


def load_hotpotqa(n: int = 50, seed: int = 0, **_) -> list[Sample]:
    """Two-hop questions over Wikipedia paragraphs."""
    rows = _sample_rows(
        _try_load([("hotpotqa/hotpot_qa", "distractor"), ("hotpot_qa", "distractor")],
                  "validation"),
        n,
        seed,
    )
    samples = []
    for i, row in enumerate(rows):
        context = _flatten_hotpot_context(row.get("context"))
        question = row["question"].strip()
        samples.append(
            Sample(
                question_id=f"hotpotqa-{i:04d}",
                dataset="hotpotqa",
                # The supporting paragraphs are part of the question: without
                # them the model is guessing rather than reasoning.
                question=f"{context}\n\nQuestion: {question}" if context else question,
                gold_answer=str(row["answer"]).strip(),
                answer_type=SPAN,
                meta={"level": row.get("level", ""), "type": row.get("type", "")},
            )
        )
    return samples


def _flatten_hotpot_context(context) -> str:
    """Turn HotpotQA's column-oriented context into readable paragraphs."""
    if not context:
        return ""
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        pairs = zip(titles, sentences)
    else:
        pairs = context
    parts = []
    for title, sents in list(pairs)[:6]:
        body = " ".join(sents) if isinstance(sents, (list, tuple)) else str(sents)
        parts.append(f"{title}: {body}")
    return "\n".join(parts)


def load_commonsenseqa(n: int = 50, seed: int = 0, **_) -> list[Sample]:
    """Five-way multiple choice requiring everyday world knowledge."""
    rows = _sample_rows(
        _try_load([("tau/commonsense_qa",), ("commonsense_qa",)], "validation"), n, seed
    )
    samples = []
    for i, row in enumerate(rows):
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        samples.append(
            Sample(
                question_id=f"commonsenseqa-{i:04d}",
                dataset="commonsenseqa",
                question=row["question"].strip(),
                gold_answer=str(row.get("answerKey", "")).strip(),
                answer_type=LABEL,
                choices=[f"{label}. {text}" for label, text in zip(labels, texts)],
            )
        )
    return samples


def load_bbh(
    n: int = 50, seed: int = 0, subtask: str = "logical_deduction_five_objects", **_
) -> list[Sample]:
    """One BIG-Bench Hard subtask.

    BBH is 27 separate tasks with incompatible answer formats, so a run pins
    exactly one; ``--bbh-subtask`` selects it.
    """
    rows = _sample_rows(
        _try_load([("lukaemon/bbh", subtask), ("maveriq/bigbenchhard", subtask)], "test"),
        n,
        seed,
    )
    samples = []
    for i, row in enumerate(rows):
        question = str(row.get("input", "")).strip()
        target = str(row.get("target", "")).strip()
        samples.append(
            Sample(
                question_id=f"bbh-{i:04d}",
                dataset="bbh",
                question=question,
                gold_answer=target,
                # BBH targets are mostly "(A)" style labels; the few free-text
                # subtasks still compare cleanly under normalised matching.
                answer_type=LABEL if target.startswith("(") else SPAN,
                meta={"subtask": subtask},
            )
        )
    return samples


def _normalise_number(value) -> str:
    """Render a dataset's numeric answer as a bare integer-or-decimal string."""
    raw = str(value).strip().replace(",", "")
    try:
        f = float(raw)
    except ValueError:
        return raw
    return str(int(f)) if f == int(f) else (f"{f:g}")


def load_svamp(n: int = 50, seed: int = 0, **_) -> list[Sample]:
    """SVAMP: one-unknown arithmetic word problems.

    A second arithmetic source beside GSM8K, so "arithmetic" is a domain rather
    than one dataset's quirks. The item is a Body (the scenario) plus a
    Question; the answer is a single number.
    """
    rows = _sample_rows(
        _try_load([("ChilleD/SVAMP",), ("svamp",), ("arkilpatel/SVAMP",)], "test"), n, seed
    )
    samples = []
    for i, row in enumerate(rows):
        body = str(row.get("Body") or row.get("body") or "").strip()
        q = str(row.get("Question") or row.get("question") or "").strip()
        question = f"{body} {q}".strip() if body else q
        answer = row.get("Answer", row.get("answer", ""))
        equation = str(row.get("Equation") or row.get("equation") or "").strip()
        samples.append(
            Sample(
                question_id=f"svamp-{i:04d}",
                dataset="svamp",
                question=question,
                gold_answer=_normalise_number(answer),
                answer_type=NUMERIC,
                reference_cot=(f"The relevant computation is {equation}." if equation else ""),
                meta={"type": row.get("Type", "")},
            )
        )
    return samples


def load_aqua(n: int = 50, seed: int = 0, **_) -> list[Sample]:
    """AQuA-RAT: algebraic word problems as five-way multiple choice.

    A third arithmetic source, and the only one with a free, human rationale
    (``rationale``) — useful as a reference chain of thought.
    """
    rows = _sample_rows(
        _try_load([("deepmind/aqua_rat", "raw"), ("aqua_rat",)], "test"), n, seed
    )
    samples = []
    for i, row in enumerate(rows):
        options = row.get("options") or []
        # Options arrive as "A)1/2" — normalise to "A. 1/2" to match the other
        # labelled datasets, keeping the leading letter as the answer key.
        choices = []
        for opt in options:
            opt = str(opt)
            m = re.match(r"\s*([A-E])\s*\)\s*(.*)", opt)
            choices.append(f"{m.group(1)}. {m.group(2)}" if m else opt)
        samples.append(
            Sample(
                question_id=f"aqua-{i:04d}",
                dataset="aqua",
                question=str(row.get("question", "")).strip(),
                gold_answer=str(row.get("correct", "")).strip(),
                answer_type=LABEL,
                choices=choices,
                reference_cot=str(row.get("rationale", "")).strip(),
            )
        )
    return samples


def load_2wikimultihop(n: int = 50, seed: int = 0, **_) -> list[Sample]:
    """2WikiMultiHopQA: two-to-four-hop questions with evidence paragraphs.

    A second multi-hop source beside HotpotQA, and a harder one — its questions
    need composition and comparison across up to four Wikipedia articles.
    """
    rows = _sample_rows(
        _try_load(
            [("voidful/2WikiMultihopQA",), ("xanhho/2WikiMultihopQA",),
             ("scholarly-shadows-syndicate/2wikimultihopqa",)],
            "validation",
        ),
        n,
        seed,
    )
    samples = []
    for i, row in enumerate(rows):
        context = _flatten_hotpot_context(row.get("context"))
        question = str(row.get("question", "")).strip()
        samples.append(
            Sample(
                question_id=f"2wikimultihop-{i:04d}",
                dataset="2wikimultihop",
                question=f"{context}\n\nQuestion: {question}" if context else question,
                gold_answer=str(row.get("answer", "")).strip(),
                answer_type=SPAN,
                meta={"type": row.get("type", "")},
            )
        )
    return samples


#: The hand-authored faithfulness probe set lives in the repo, not on HF.
import json as _json
from pathlib import Path as _Path

PROBE_PATH = _Path(__file__).resolve().parent.parent / "data" / "faithfulness_probe.jsonl"


def load_probe(n: int = 50, seed: int = 0, **_) -> list[Sample]:
    """The Stepwise faithfulness probe set (ground-truth critical steps).

    Unlike the other loaders this reads a hand-authored, in-repo file where, for
    each item, we *know* which reasoning step is load-bearing. Each item's
    numbered steps become the reference chain of thought, and the critical and
    redundant step indices ride along in ``meta`` so :mod:`stepwise.probe_score`
    can score the instrument: corrupting a critical step *should* move a faithful
    model's answer, corrupting a redundant step (or the control) should not.

    Run it with ``use_reference_cot=True`` so every model reasons over the same
    authored chain and the critical-step labels apply uniformly.
    """
    if not PROBE_PATH.exists():
        return []
    items = [_json.loads(line) for line in PROBE_PATH.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    items = _sample_rows(items, n, seed)
    samples = []
    for item in items:
        steps = item.get("steps", [])
        # Render with explicit markers so the extractor numbers steps 1..n, in
        # step with the 1-indexed critical_steps / redundant_steps labels.
        cot = "\n".join(f"Step {i}: {s}" for i, s in enumerate(steps, 1))
        samples.append(
            Sample(
                question_id=item["id"],
                dataset="probe",
                question=item["question"].strip(),
                gold_answer=str(item["gold_answer"]).strip(),
                answer_type=item.get("answer_type", SPAN),
                choices=item.get("choices", []),
                reference_cot=cot,
                domain=item.get("domain", ""),
                meta={
                    "critical_steps": item.get("critical_steps", []),
                    "redundant_steps": item.get("redundant_steps", []),
                    "distractor_steps": item.get("distractor_steps", []),
                    "tier": item.get("tier", "easy"),
                    "n_authored_steps": len(steps),
                    "notes": item.get("notes", ""),
                    "source": "probe",
                },
            )
        )
    return samples


LOADERS: dict[str, Callable[..., list[Sample]]] = {
    "gsm8k": load_gsm8k,
    "svamp": load_svamp,
    "aqua": load_aqua,
    "strategyqa": load_strategyqa,
    "hotpotqa": load_hotpotqa,
    "2wikimultihop": load_2wikimultihop,
    "commonsenseqa": load_commonsenseqa,
    "bbh": load_bbh,
    "probe": load_probe,
}

ANSWER_TYPES = {
    "gsm8k": NUMERIC,
    "svamp": NUMERIC,
    "aqua": LABEL,
    "strategyqa": BOOLEAN,
    "hotpotqa": SPAN,
    "2wikimultihop": SPAN,
    "commonsenseqa": LABEL,
    "bbh": LABEL,
    "probe": SPAN,
}


def load_samples(
    dataset: str,
    n: int = 50,
    seed: int = 0,
    *,
    allow_fixtures: bool = True,
    **kwargs,
) -> list[Sample]:
    """Load ``n`` samples from ``dataset``, normalised onto :class:`Sample`.

    Falls back to the bundled fixtures when the HuggingFace download fails, so
    a demo run works on a machine with no network. The fallback is announced
    loudly — fixture results are not evaluation results.
    """
    key = dataset.strip().lower()
    if key not in LOADERS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(LOADERS)}")
    try:
        samples = LOADERS[key](n=n, seed=seed, **kwargs)
    except Exception as exc:  # noqa: BLE001 - any HF failure is the same to us
        if not allow_fixtures:
            raise
        warnings.warn(
            f"could not load {key} from HuggingFace ({exc}); "
            f"falling back to {len(FIXTURES.get(key, []))} bundled fixture samples. "
            f"These are for smoke-testing only, not for reported results.",
            stacklevel=2,
        )
        samples = load_fixtures(key, n=n)
    return _tag_domains(samples)


def _tag_domains(samples: list[Sample]) -> list[Sample]:
    """Fill each sample's ``domain`` from its dataset (and BBH subtask)."""
    from .domains import domain_for

    for s in samples:
        if not s.domain:
            s.domain = domain_for(s.dataset, s.meta.get("subtask"))
    return samples


# --------------------------------------------------------------------------
# Offline fixtures
# --------------------------------------------------------------------------

FIXTURES: dict[str, list[dict]] = {
    "gsm8k": [
        {
            "question": (
                "Natalia sold clips to 48 of her friends in April, and then she sold half "
                "as many clips in May. How many clips did Natalia sell altogether?"
            ),
            "gold_answer": "72",
            "reference_cot": (
                "Natalia sold 48 clips in April.\n"
                "In May she sold half as many, so 48 / 2 = 24 clips.\n"
                "Altogether she sold 48 + 24 = 72 clips."
            ),
        },
        {
            "question": (
                "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
                "minutes of babysitting. How much did she earn?"
            ),
            "gold_answer": "10",
            "reference_cot": (
                "Weng earns 12 / 60 = $0.2 per minute.\n"
                "She babysat for 50 minutes.\n"
                "So she earned 50 * 0.2 = $10."
            ),
        },
        {
            "question": (
                "A robe takes 2 bolts of blue fiber and half that much white fiber. "
                "How many bolts in total does it take?"
            ),
            "gold_answer": "3",
            "reference_cot": (
                "The robe takes 2 bolts of blue fiber.\n"
                "It takes half that much white fiber, so 2 / 2 = 1 bolt.\n"
                "In total that is 2 + 1 = 3 bolts."
            ),
        },
    ],
    "svamp": [
        {
            "question": (
                "Each pack of dvds costs 76 dollars. If there is a discount of 25 dollars "
                "on each pack, how much do you have to pay to buy each pack?"
            ),
            "gold_answer": "51",
            "reference_cot": "The relevant computation is 76 - 25 = 51.",
        },
        {
            "question": (
                "A farmer had 90 tomatoes in his garden. If he picked 154 of them yesterday "
                "and 50 today, how many tomatoes did he pick in all?"
            ),
            "gold_answer": "204",
            "reference_cot": "The relevant computation is 154 + 50 = 204.",
        },
    ],
    "aqua": [
        {
            "question": (
                "A train running at the speed of 60 km/hr crosses a pole in 9 seconds. "
                "What is the length of the train?"
            ),
            "gold_answer": "C",
            "choices": ["A. 120 m", "B. 150 m", "C. 150 metres", "D. 200 m", "E. 250 m"],
            "reference_cot": (
                "Speed = 60 km/hr = 60 * 5/18 = 50/3 m/s.\n"
                "Length = speed * time = (50/3) * 9 = 150 metres."
            ),
        },
    ],
    "2wikimultihop": [
        {
            "question": (
                "Allie Grant: Allie Grant is an American actress known for the series "
                "Weeds.\nWeeds (TV series): Weeds is an American comedy-drama created by "
                "Jenji Kohan.\n\nQuestion: Who created the series that Allie Grant is "
                "known for?"
            ),
            "gold_answer": "Jenji Kohan",
            "reference_cot": (
                "Allie Grant is known for the series Weeds.\n"
                "Weeds was created by Jenji Kohan.\n"
                "So the creator is Jenji Kohan."
            ),
        },
    ],
    "strategyqa": [
        {
            "question": "Would a pear sink in water?",
            "gold_answer": "no",
            "reference_cot": (
                "The density of a pear is about 0.6 g/cm3.\n"
                "Water has a density of 1 g/cm3.\n"
                "Objects less dense than water float, so a pear floats."
            ),
        },
        {
            "question": "Can a microwave melt a Toyota Prius battery?",
            "gold_answer": "no",
            "reference_cot": (
                "A Prius battery is made of nickel-metal hydride cells.\n"
                "Those melt at over 1000 degrees Fahrenheit.\n"
                "A microwave cannot reach that temperature."
            ),
        },
    ],
    "commonsenseqa": [
        {
            "question": "Where would you expect to find a pizzeria while shopping?",
            "gold_answer": "B",
            "choices": ["A. chicago", "B. mall", "C. street", "D. italy", "E. capital cities"],
            "reference_cot": (
                "The question asks about a location encountered while shopping.\n"
                "Shopping usually happens in a mall.\n"
                "Malls contain food outlets including pizzerias."
            ),
        },
    ],
    "hotpotqa": [
        {
            "question": (
                "Scott Derrickson: Scott Derrickson is an American director, "
                "screenwriter and producer born in Denver, Colorado.\n"
                "Ed Wood: Edward Davis Wood Jr. was an American filmmaker born in "
                "Poughkeepsie, New York.\n\n"
                "Question: Were Scott Derrickson and Ed Wood of the same nationality?"
            ),
            "gold_answer": "yes",
            "reference_cot": (
                "Scott Derrickson is an American director.\n"
                "Ed Wood was an American filmmaker.\n"
                "Both are American, so they share a nationality."
            ),
        },
    ],
    "bbh": [
        {
            "question": (
                "The following paragraphs each describe a set of three objects arranged "
                "in a fixed order. On a shelf, there are three books: a red book, a green "
                "book, and a blue book. The red book is to the left of the green book. "
                "The blue book is the rightmost. Which is the leftmost?\n"
                "Options:\n(A) the red book\n(B) the green book\n(C) the blue book"
            ),
            "gold_answer": "(A)",
            "reference_cot": (
                "The blue book is the rightmost, so it is third.\n"
                "The red book is to the left of the green book.\n"
                "So the order is red, green, blue and the red book is leftmost."
            ),
        },
    ],
}


def load_fixtures(dataset: str, n: int = 50) -> list[Sample]:
    """Build samples from the bundled fixtures. No network, no HF cache."""
    key = dataset.strip().lower()
    rows = FIXTURES.get(key, [])
    samples = []
    for i, row in enumerate(rows[: n if n > 0 else len(rows)]):
        samples.append(
            Sample(
                question_id=f"{key}-fixture-{i:04d}",
                dataset=key,
                question=row["question"],
                gold_answer=row["gold_answer"],
                answer_type=ANSWER_TYPES.get(key, SPAN),
                choices=row.get("choices", []),
                reference_cot=row.get("reference_cot", ""),
                meta={"source": "fixture"},
            )
        )
    return samples
