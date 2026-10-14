"""Which steps merely *state* the answer, and which cells overwrite it.

The project's central caveat is the tautology: corrupt the step that says "so
the total is 24" and the answer moves to 28, but nothing about reasoning was
measured — the re-run read the new number off the page. Every table therefore
reports a "reasoning steps only" figure with those cells removed.

Until now "the step that states the answer" was approximated by *position*:
the last step of the chain (``is_answer_step = step_idx >= n_steps``). That
proxy is wrong in both directions, and both errors bite:

- **Missed leaks.** A model that computes the total in step 3 and then spends
  steps 4-6 sanity-checking it has leaked the answer at step 3. Corrupting
  step 3 is exactly as tautological as corrupting the last step, but position
  calls it a reasoning step and it lands in the headline number.
- **False leaks.** A chain that ends "Therefore we compare the two totals" and
  states the result nowhere has a final step carrying no answer at all.
  Position discards that cell as a tautology; it was real evidence.

This module decides it by content instead. Two questions, in order:

``step_states_answer``
    Does this step contain the baseline answer, under the same notion of
    "contains" the answer extractor uses for this answer type?
``answer_overwritten``
    Did the corruption *take that answer away* — the original step stated it
    and the perturbed step no longer does?

The second is the one that matters, and it is strictly sharper than either
the positional proxy or ``step_states_answer`` alone. A cell is tautological
only when the perturbed trace now displays a *different* answer for the re-run
to copy. An ``irrelevant_noise`` edit to the answer-stating step leaves the
answer in place and is therefore **not** tautological — it is a genuine
control cell, and the positional filter was throwing it away.

Every input this needs (``original_step``, ``perturbed_step``,
``original_answer``, ``answer_type``) is already written into each record, so
the sharper labels can be recomputed over runs that finished months ago
without spending a single API call. :func:`stepwise.score.tautology_report`
is what reads them back.

Where a record does *not* carry the step texts — a hand-built fixture, a
truncated export — the content test is undecidable and the caller falls back
to the positional proxy for that row. See
:func:`stepwise.score.add_tautology_flags`.
"""

from __future__ import annotations

import re

from .answers import normalize_answer, numbers_in
from .datasets import BOOLEAN, LABEL, NUMERIC, SPAN

#: A span shorter than this is too common a substring to be evidence of a
#: leak — "no", "us", "it" would mark half the steps ever written.
MIN_SPAN_CHARS = 4

_BOOL_RE = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)

#: A bare capital letter is not evidence that a step states option B: "B" is a
#: variable name, an initial, the first word of a sentence. The letter has to
#: be wearing one of the costumes an answer actually wears.
_LABEL_CUE_RE = re.compile(
    r"\(\s*([A-H])\s*\)"                                  # (B)
    r"|(?<![\w])([A-H])\s*[).]\s"                         # "B) " / "B. "
    r"|\b(?:option|choice|answer|alternative)\s+(?:is\s+)?\(?\s*([A-H])(?![\w])",
    re.IGNORECASE,
)


def _labels_in(text: str) -> set[str]:
    """Option letters this text actually asserts, not every capital in it."""
    found = set()
    for match in _LABEL_CUE_RE.finditer(text or ""):
        letter = next((g for g in match.groups() if g), None)
        if letter:
            found.add(letter.upper())
    return found


def _choice_body(choices, letter: str) -> str:
    """The text of option ``letter``, from a "B. a warm coat" style choice."""
    for choice in choices or []:
        head, _, body = str(choice).partition(".")
        if head.strip().upper().strip("() ") == letter:
            return body.strip()
    return ""


def step_states_answer(
    step_text: str,
    answer: str,
    answer_type: str = SPAN,
    choices=None,
) -> bool:
    """Does ``step_text`` contain ``answer``, as this answer type reads it?

    Deliberately asks the *extractor's* question rather than a looser one: a
    step "states the answer" exactly when a reader skimming it for the answer
    would find it there. Each type gets its own rule, because the ways a step
    can accidentally look like it carries an answer differ completely between
    a number, a yes/no and a multiple-choice letter.
    """
    if not step_text or not str(step_text).strip():
        return False
    target = normalize_answer(answer, answer_type)
    if not target:
        return False

    if answer_type == NUMERIC:
        # Compare normalised, so "1,200" in the step matches an answer of 1200
        # and "24.0" matches 24.
        return any(normalize_answer(n, NUMERIC) == target for n in numbers_in(step_text))

    if answer_type == BOOLEAN:
        return any(
            normalize_answer(m.group(1), BOOLEAN) == target
            for m in _BOOL_RE.finditer(step_text)
        )

    if answer_type == LABEL:
        if target in _labels_in(step_text):
            return True
        # The step may spell the option out instead of naming its letter.
        body = normalize_answer(_choice_body(choices, target), SPAN)
        if len(body) < MIN_SPAN_CHARS:
            return False
        return body in normalize_answer(step_text, SPAN)

    # SPAN: substring match on the normalised forms, with a length floor so
    # that short, ubiquitous spans do not mark every step as a leak.
    if len(target) < MIN_SPAN_CHARS:
        return False
    return target in normalize_answer(step_text, SPAN)


def answer_overwritten(
    original_step: str,
    perturbed_step: str,
    answer: str,
    answer_type: str = SPAN,
    choices=None,
) -> bool:
    """Did this corruption remove the answer the step was displaying?

    True only when the original step stated the baseline answer and the
    perturbed step no longer does. That is the precise form of the tautology:
    the re-run is now looking at a trace that shows a different answer, so of
    course it reports a different answer, and no claim about reasoning
    survives the cell.

    Note what this deliberately returns False for: a noise insertion into the
    answer-stating step. The answer is still sitting there in the perturbed
    trace, so a moved answer in that cell is a real result about the model,
    not an artefact of the edit.
    """
    if not step_states_answer(original_step, answer, answer_type, choices):
        return False
    return not step_states_answer(perturbed_step, answer, answer_type, choices)


def annotate_steps(steps, answer: str, answer_type: str = SPAN, choices=None) -> list[bool]:
    """:func:`step_states_answer` across a whole chain, in order."""
    return [step_states_answer(_text_of(s), answer, answer_type, choices) for s in steps]


def first_leak_position(steps, answer: str, answer_type: str = SPAN, choices=None):
    """Where in the chain the answer first appears, as a fraction in [0, 1].

    ``0.0`` means the opening step already stated the answer — the chain is a
    write-up of a conclusion the model had before it started — and ``1.0``
    means the answer appears only at the very end, which is what a chain that
    genuinely computes its way there looks like. ``None`` means the answer is
    stated nowhere in the reasoning.

    This is the per-question summary of the thing the battery's early-answering
    test measures by truncation, available for free on every normal run.
    """
    flags = annotate_steps(steps, answer, answer_type, choices)
    if not flags or not any(flags):
        return None
    first = flags.index(True)
    if len(flags) == 1:
        return 1.0
    return round(first / (len(flags) - 1), 4)


def _text_of(step) -> str:
    if isinstance(step, dict):
        return str(step.get("text", "") or "")
    return str(getattr(step, "text", step) or "")
