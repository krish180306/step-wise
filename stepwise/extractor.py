"""Segment a chain of thought into discrete, individually addressable steps.

The pipeline intervenes on exactly one step at a time, so segmentation has to
be *lossless*: after splitting, re-rendering with no overrides must reproduce
the original trace byte for byte. Every marker, indent and blank line stays
attached to the step it came from.

Two shapes of CoT are handled:

``marked``
    Explicit step markers, either one per line ("Step 3:", "3.", "3)", "-")
    or run together inline ("Step 1: ... Step 2: ...").
``prose``
    Unmarked narrative reasoning, split on sentence boundaries with guards for
    decimals, abbreviations and initials.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Words that end in a period without ending a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc",
    "e.g", "i.e", "eg", "ie", "cf", "viz", "al", "fig", "eq", "approx", "no",
    "inc", "ltd", "co", "corp", "dept", "est", "min", "max", "avg", "hr",
    "hrs", "sec", "ft", "lb", "lbs", "oz", "qty", "vol", "ch", "pp", "resp",
}

# "Step 4:" / "Step 4." / "Step 4 -"
_STEP_WORD = r"(?:step|stage|part)\s*\#?\s*\d+\s*[:.)\-–]"
# "4." / "4)" / "(4)"
_BARE_NUMBER = r"\(?\d{1,2}[.)]"
# "-" / "*" / bullet glyphs
_BULLET = r"[-*•‣▪]"

_LINE_MARKER_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>" + _STEP_WORD + r"|" + _BARE_NUMBER + r"|" + _BULLET
    + r")(?P<gap>[ \t]+|$)",
    re.IGNORECASE,
)
_INLINE_STEP_RE = re.compile(
    r"(?<![\w])(?P<marker>" + _STEP_WORD + r")(?P<gap>[ \t]*)", re.IGNORECASE
)

# A candidate sentence end: terminal punctuation, optional closing quote or
# bracket, then whitespace.
_SENTENCE_END_RE = re.compile(r"[.!?]+[\"'’”)\]]*(?=\s)")
_TRAILING_WS_RE = re.compile(r"\s+$")

#: Fragments shorter than this get merged into the preceding step rather than
#: standing alone — "So 12." is a tail, not a reasoning step.
MIN_STEP_CHARS = 16


@dataclass
class ReasoningStep:
    """One step, split into the parts needed to rebuild the trace exactly."""

    index: int  # 1-based, as shown to models and stored in records
    text: str  # the step's own content, marker stripped
    marker: str = ""  # "Step 2: ", "3. ", "- ", or "" for prose
    indent: str = ""  # leading whitespace preserved from the source
    trailing: str = ""  # whitespace separating this step from the next
    start: int = 0  # offset of ``text`` within the original CoT
    end: int = 0

    @property
    def rendered(self) -> str:
        """The step exactly as it appeared, marker and spacing included."""
        return f"{self.indent}{self.marker}{self.text}{self.trailing}"

    def with_text(self, new_text: str) -> "ReasoningStep":
        return ReasoningStep(
            index=self.index,
            text=new_text,
            marker=self.marker,
            indent=self.indent,
            trailing=self.trailing,
            start=self.start,
            end=self.start + len(new_text),
        )

    def to_dict(self) -> dict:
        return {"index": self.index, "text": self.text, "marker": self.marker.strip()}


@dataclass
class ExtractedCoT:
    """A segmented chain of thought plus the machinery to splice it back."""

    raw: str
    steps: list[ReasoningStep] = field(default_factory=list)
    preamble: str = ""  # text before the first step, usually empty
    style: str = "prose"  # "marked" | "prose" | "single" | "empty"

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def step(self, index: int) -> ReasoningStep:
        """Fetch a step by its 1-based index."""
        for s in self.steps:
            if s.index == index:
                return s
        raise IndexError(f"no step with index {index} (have 1..{len(self.steps)})")

    def render(
        self,
        overrides: dict[int, str] | None = None,
        *,
        drop: "set[int] | list[int] | tuple[int, ...] | None" = None,
    ) -> str:
        """Rebuild the trace, substituting text for the given step indices.

        ``render()`` with no arguments returns ``self.raw`` unchanged.

        ``drop`` removes those steps from the trace entirely, which is the
        ablation intervention: it asks whether the answer survives the step's
        *absence* rather than whether it follows an injected error. Surviving
        numeric markers are renumbered so the chain still reads 1, 2, 3 — a
        gap at "Step 1, Step 3" is a visible tell that something was taken
        out, and a model that spots it is answering a different question than
        the one the ablation means to ask.
        """
        overrides = overrides or {}
        drop = set(drop or ())
        known = {s.index for s in self.steps}
        unknown = (set(overrides) | drop) - known
        if unknown:
            raise IndexError(f"cannot override nonexistent step(s): {sorted(unknown)}")
        kept = [s for s in self.steps if s.index not in drop]
        parts = [self.preamble]
        for position, s in enumerate(kept, start=1):
            text = overrides.get(s.index, s.text)
            marker = _renumber_marker(s.marker, position) if drop else s.marker
            parts.append(f"{s.indent}{marker}{text}{s.trailing}")
        return "".join(parts)

    def step_texts(self) -> list[str]:
        return [s.text for s in self.steps]

    def to_dict(self) -> dict:
        return {"style": self.style, "steps": [s.to_dict() for s in self.steps]}


_MARKER_NUMBER_RE = re.compile(r"\d+")


def _renumber_marker(marker: str, position: int) -> str:
    """Rewrite a marker's number to ``position``, leaving its shape alone.

    "Step 3: " becomes "Step 2: " and "3. " becomes "2. ", while a bullet or
    an unnumbered marker is returned untouched. Only used when steps have been
    dropped, so an ordinary render is byte-identical to the source.
    """
    if not marker or not _MARKER_NUMBER_RE.search(marker):
        return marker
    return _MARKER_NUMBER_RE.sub(str(position), marker, count=1)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def extract_steps(cot: str, *, min_step_chars: int = MIN_STEP_CHARS) -> ExtractedCoT:
    """Segment ``cot`` into reasoning steps.

    Marker-based segmentation is tried first — it is unambiguous when present.
    Prose falls through to the sentence splitter.
    """
    if not cot or not cot.strip():
        return ExtractedCoT(raw=cot or "", steps=[], preamble=cot or "", style="empty")

    marked = _extract_marked(cot)
    if marked is not None and len(marked.steps) >= 2:
        return _merge_short_steps(marked, min_step_chars)

    return _merge_short_steps(_extract_prose(cot), min_step_chars)


def splice_step(cot: str | ExtractedCoT, step_index: int, new_text: str) -> str:
    """Return ``cot`` with step ``step_index`` replaced by ``new_text``.

    Every other step, and all markers and spacing, are left untouched. This is
    the single mutation the whole experiment rests on, so it is deliberately
    the smallest operation that can express it.
    """
    extracted = cot if isinstance(cot, ExtractedCoT) else extract_steps(cot)
    return extracted.render({step_index: new_text})


def ablate_step(cot: str | ExtractedCoT, step_index: int) -> str:
    """Return ``cot`` with step ``step_index`` removed and the rest renumbered.

    The ablation counterpart to :func:`splice_step`: instead of corrupting the
    step, it deletes it, so the re-run measures whether the step was *needed*
    rather than whether an error in it propagates.
    """
    extracted = cot if isinstance(cot, ExtractedCoT) else extract_steps(cot)
    return extracted.render(drop={step_index})


def splice_steps(cot: str | ExtractedCoT, replacements: dict[int, str]) -> str:
    """Multi-step variant of :func:`splice_step`."""
    extracted = cot if isinstance(cot, ExtractedCoT) else extract_steps(cot)
    return extracted.render(replacements)


# --------------------------------------------------------------------------
# Marker-based segmentation
# --------------------------------------------------------------------------


def _extract_marked(cot: str) -> ExtractedCoT | None:
    """Split on explicit step markers, per line first then inline."""
    by_line = _extract_line_markers(cot)
    if by_line is not None and len(by_line.steps) >= 2:
        return by_line
    return _extract_inline_markers(cot)


def _extract_line_markers(cot: str) -> ExtractedCoT | None:
    """Segment when each step starts on its own line with a marker."""
    lines = cot.splitlines(keepends=True)
    marked: list[tuple[int, re.Match]] = []
    for i, line in enumerate(lines):
        m = _LINE_MARKER_RE.match(line)
        if m:
            marked.append((i, m))
    if len(marked) < 2:
        return None

    # A trailing "The answer is 42." line belongs to the last step, so each
    # block runs from one marker up to the next.
    preamble = "".join(lines[: marked[0][0]])
    steps: list[ReasoningStep] = []
    offset = len(preamble)
    for n, (line_no, match) in enumerate(marked):
        end_line = marked[n + 1][0] if n + 1 < len(marked) else len(lines)
        block = "".join(lines[line_no:end_line])
        indent = match.group("indent")
        marker = match.group("marker") + match.group("gap")
        body = block[len(indent) + len(marker):]
        trailing = _trailing_ws(body)
        text = body[: len(body) - len(trailing)]
        start = offset + len(indent) + len(marker)
        steps.append(
            ReasoningStep(
                index=n + 1,
                text=text,
                marker=marker,
                indent=indent,
                trailing=trailing,
                start=start,
                end=start + len(text),
            )
        )
        offset += len(block)
    return ExtractedCoT(raw=cot, steps=steps, preamble=preamble, style="marked")


def _extract_inline_markers(cot: str) -> ExtractedCoT | None:
    """Segment "Step 1: ... Step 2: ..." written on a single line."""
    matches = list(_INLINE_STEP_RE.finditer(cot))
    if len(matches) < 2:
        return None

    preamble = cot[: matches[0].start()]
    steps: list[ReasoningStep] = []
    for n, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[n + 1].start() if n + 1 < len(matches) else len(cot)
        body = cot[body_start:body_end]
        trailing = _trailing_ws(body)
        text = body[: len(body) - len(trailing)]
        steps.append(
            ReasoningStep(
                index=n + 1,
                text=text,
                marker=match.group("marker") + match.group("gap"),
                indent="",
                trailing=trailing,
                start=body_start,
                end=body_start + len(text),
            )
        )
    return ExtractedCoT(raw=cot, steps=steps, preamble=preamble, style="marked")


# --------------------------------------------------------------------------
# Prose segmentation
# --------------------------------------------------------------------------


def _extract_prose(cot: str) -> ExtractedCoT:
    """Split unmarked reasoning on sentence boundaries.

    Paragraph breaks always end a step; within a paragraph the sentence
    splitter decides, guarding decimals ("3.5"), abbreviations ("e.g.") and
    initials ("J. Smith").
    """
    leading = len(cot) - len(cot.lstrip())
    preamble = cot[:leading]
    body = cot[leading:]

    boundaries = _sentence_boundaries(body)
    if not boundaries:
        trailing = _trailing_ws(body)
        text = body[: len(body) - len(trailing)]
        if not text:
            return ExtractedCoT(raw=cot, steps=[], preamble=cot, style="empty")
        step = ReasoningStep(
            index=1, text=text, trailing=trailing, start=leading, end=leading + len(text)
        )
        return ExtractedCoT(raw=cot, steps=[step], preamble=preamble, style="single")

    steps: list[ReasoningStep] = []
    cursor = 0
    for cut in [*boundaries, len(body)]:
        chunk = body[cursor:cut]
        chunk_start = cursor
        cursor = cut
        if not chunk:
            continue
        if not chunk.strip():
            # Pure whitespace tail: attach it to the previous step.
            if steps:
                steps[-1].trailing += chunk
            else:
                preamble += chunk
            continue
        # The separator before a sentence belongs to the step that precedes
        # it, so step text starts at the first non-space character.
        gap = len(chunk) - len(chunk.lstrip())
        if gap:
            if steps:
                steps[-1].trailing += chunk[:gap]
            else:
                preamble += chunk[:gap]
            chunk = chunk[gap:]
            chunk_start += gap
        trailing = _trailing_ws(chunk)
        text = chunk[: len(chunk) - len(trailing)]
        start = leading + chunk_start
        steps.append(
            ReasoningStep(
                index=len(steps) + 1,
                text=text,
                trailing=trailing,
                start=start,
                end=start + len(text),
            )
        )
    return ExtractedCoT(raw=cot, steps=steps, preamble=preamble, style="prose")


def _sentence_boundaries(text: str) -> list[int]:
    """Offsets where one sentence ends and the next begins.

    A boundary points at the first whitespace character after the terminal
    punctuation, so that whitespace becomes the previous step's ``trailing``.
    """
    cuts: set[int] = set()
    for match in _SENTENCE_END_RE.finditer(text):
        end = match.end()
        if end >= len(text):
            continue
        if _is_paragraph_break(text, end):
            cuts.add(end)
            continue
        if _is_false_boundary(text, match):
            continue
        cuts.add(end)

    # Blank lines end a step even without terminal punctuation.
    for match in re.finditer(r"\n[ \t]*\n", text):
        cuts.add(match.start())
    return sorted(c for c in cuts if 0 < c < len(text))


def _is_paragraph_break(text: str, end: int) -> bool:
    return "\n" in text[end: end + 4]


def _is_false_boundary(text: str, match: re.Match) -> bool:
    """True when the punctuation ends a token rather than a sentence."""
    before = text[: match.start() + 1]
    after = text[match.end():].lstrip()

    # "we get 5. the total is ..." — a lowercase continuation is mid-sentence.
    if after and (after[0].islower() or after[0] in ",;:)"):
        return True

    word = re.search(r"([\w.$]+)\.$", before)
    if word:
        token = word.group(1).lower().rstrip(".")
        if token in _ABBREVIATIONS:
            return True
        # A single initial: "J. Smith", or a lone letter enumerator "a.".
        if len(token) == 1 and token.isalpha():
            return True
    return False


def _merge_short_steps(extracted: ExtractedCoT, min_chars: int) -> ExtractedCoT:
    """Fold sub-threshold fragments into the step before them.

    A two-word fragment is not something a perturbation can meaningfully act
    on, and leaving it in inflates the denominator of the faithfulness score.
    """
    if min_chars <= 0 or len(extracted.steps) <= 1:
        return extracted

    merged: list[ReasoningStep] = []
    for step in extracted.steps:
        too_short = len(step.text.strip()) < min_chars
        # Never merge a marked step: the author declared it a step.
        if too_short and merged and not step.marker:
            prev = merged[-1]
            prev.text = f"{prev.text}{prev.trailing}{step.indent}{step.text}"
            prev.trailing = step.trailing
            prev.end = step.end
        else:
            merged.append(step)

    for n, step in enumerate(merged, start=1):
        step.index = n
    extracted.steps = merged
    return extracted


def _trailing_ws(chunk: str) -> str:
    match = _TRAILING_WS_RE.search(chunk)
    return match.group(0) if match else ""
