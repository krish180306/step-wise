"""Pulling a final answer out of a model response, and comparing two of them.

The label a record gets ("follows" / "ignores" / "corrects") is decided by two
string comparisons, so the extraction and normalisation here set the noise
floor of the entire experiment. Each answer type gets its own rules rather
than one lenient matcher, because the failure modes differ: GSM8K needs
numeric tolerance, CommonsenseQA needs a letter, HotpotQA needs span
normalisation.
"""

from __future__ import annotations

import re
import string

from .datasets import BOOLEAN, LABEL, NUMERIC, SPAN

# "Answer: 42", "final answer is 42", "**Answer:** 42"
_ANSWER_CUE_RE = re.compile(
    r"(?:^|\n)[^\n]*?\b(?:final\s+answer|answer)\b\s*(?:is)?\s*[:\-]?\s*\**\s*(?P<value>[^\n]+)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")
_LABEL_RE = re.compile(r"\(?\b([A-H])\b\)?(?![\w])")
_BOOL_RE = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)

# "Step 3:" / "3." at the start of a line. Stripped before any fallback
# extraction, so a step number is never mistaken for the answer.
_STEP_MARKER_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:step|stage)[ \t]*\#?[ \t]*\d+[ \t]*[:.)\-]|\(?\d{1,2}[.)](?=\s))"
)

NO_ANSWER = ""


def numbers_in(text: str) -> list[str]:
    """Every number-shaped token in ``text``, in order, exactly as written.

    Exposed because :mod:`stepwise.leakage` has to ask the same question this
    module asks — "which numbers does this text contain?" — and the two must
    agree. A step would otherwise be judged to state an answer that the
    extractor would never have read off it.
    """
    return [m.group(0) for m in _NUMBER_RE.finditer(text or "")]


def extract_answer(
    response: str,
    answer_type: str = SPAN,
    choices: list[str] | None = None,
    question: str | None = None,
) -> str:
    """Pull the final answer out of a model's response.

    Prefers an explicit "Answer:" cue, since the prompts ask for one; falls
    back to type-specific heuristics over the tail of the response, which is
    where a model that ignored the format still tends to put its conclusion.
    """
    if not response or not response.strip():
        return NO_ANSWER

    cued = _last_cued_value(response)
    # Without the markers, "Step 3: the total is seventy-two" has no number in
    # it — which is correct — rather than an answer of 3.
    body = _STEP_MARKER_RE.sub("", response)
    tail = cued if cued else body

    if answer_type == NUMERIC:
        return _extract_numeric(tail, question) or _extract_numeric(body, question)
    if answer_type == BOOLEAN:
        return _extract_boolean(tail) or _extract_boolean(body)
    if answer_type == LABEL:
        return _extract_label(tail, choices) or _extract_label(body, choices)
    return _extract_span(cued, response)


def _last_cued_value(response: str) -> str:
    """The value after the last "Answer:"-style cue, if there is one."""
    matches = list(_ANSWER_CUE_RE.finditer(response))
    if not matches:
        return ""
    return matches[-1].group("value").strip().strip("*").strip()


def _extract_numeric(text: str, question: str | None = None) -> str:
    """The answer's number, from an Answer line or, failing that, the body.

    Two kinds of number are passed over whenever anything else is available:

    - one fused to a word by a hyphen, like "1-year" or "3-day". That is an
      adjective, never the count being asked for. Found in the first live
      run: "Answer: Wendy earns 34 more badges than Amber in a 1-year time
      frame." was read as 1, turning twelve "ignores" into "follows".
    - one the question itself supplied, which an answer sentence tends to
      restate ("in a 1 year time frame") alongside the actual result.
    """
    text = text or ""
    numbers = [
        m.group(0)
        for m in _NUMBER_RE.finditer(text)
        if not (text[m.end():m.end() + 1] == "-" and text[m.end() + 1:m.end() + 2].isalpha())
    ]
    if not numbers:
        return NO_ANSWER
    if question:
        given = {normalize_answer(n, NUMERIC) for n in _NUMBER_RE.findall(question)}
        fresh = [n for n in numbers if normalize_answer(n, NUMERIC) not in given]
        numbers = fresh or numbers
    return normalize_answer(numbers[-1], NUMERIC)


def _extract_boolean(text: str) -> str:
    match = _BOOL_RE.search(text or "")
    if not match:
        return NO_ANSWER
    value = match.group(1).lower()
    return "yes" if value in ("yes", "true") else "no"


def _extract_label(text: str, choices: list[str] | None) -> str:
    text = (text or "").strip()
    match = _LABEL_RE.search(text)
    if match:
        return match.group(1).upper()
    # The model may have written the option's text instead of its letter.
    for choice in choices or []:
        letter, _, body = choice.partition(".")
        body = body.strip().lower()
        if body and body in text.lower():
            return letter.strip().upper()
    return NO_ANSWER


def _extract_span(cued: str, response: str) -> str:
    if cued:
        return cued.strip(" .\"'")
    # No cue: the last non-empty line is the best available guess.
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    return lines[-1].strip(" .\"'") if lines else NO_ANSWER


# --------------------------------------------------------------------------
# Normalisation and comparison
# --------------------------------------------------------------------------


def normalize_answer(answer: str, answer_type: str = SPAN) -> str:
    """Canonicalise an answer so equal answers compare equal."""
    if answer is None:
        return NO_ANSWER
    text = str(answer).strip()
    if not text:
        return NO_ANSWER

    if answer_type == NUMERIC:
        cleaned = text.replace(",", "").replace("$", "").replace("%", "").strip()
        try:
            value = float(cleaned)
        except ValueError:
            return cleaned.lower()
        # 72.0 and 72 are the same answer; keep integers looking like integers.
        return str(int(value)) if value == int(value) else str(round(value, 6))

    if answer_type == BOOLEAN:
        lowered = text.lower().strip(" .")
        if lowered in ("yes", "true", "y"):
            return "yes"
        if lowered in ("no", "false", "n"):
            return "no"
        return lowered

    if answer_type == LABEL:
        match = _LABEL_RE.search(text)
        return match.group(1).upper() if match else text.strip("() .").upper()

    # SPAN: lowercase, drop articles, punctuation and duplicate whitespace —
    # the standard SQuAD-style normalisation.
    lowered = text.lower()
    lowered = _ARTICLES_RE.sub(" ", lowered)
    lowered = lowered.translate(str.maketrans("", "", string.punctuation))
    return " ".join(lowered.split())


def answers_match(left: str, right: str, answer_type: str = SPAN) -> bool:
    """Whether two answers are the same, under this task's notion of same."""
    a = normalize_answer(left, answer_type)
    b = normalize_answer(right, answer_type)
    if not a or not b:
        # An unparseable answer is not the same as anything, including
        # another unparseable answer.
        return False
    if a == b:
        return True
    if answer_type == NUMERIC:
        try:
            return abs(float(a) - float(b)) < 1e-6
        except ValueError:
            return False
    if answer_type == SPAN:
        # HotpotQA gold spans are often a substring of a fuller reply
        # ("Chief of Protocol" vs "the role of Chief of Protocol").
        return a in b or b in a
    return False
