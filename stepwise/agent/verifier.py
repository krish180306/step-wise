"""An adversarial check on the analyst's own numbers.

The report agent writes prose with numbers in it, and a language model will
occasionally state a figure that is real but attached to the wrong thing. This
project has a live example: the analyst once wrote that a model was
significantly worse "for arithmetic_change (p = 0.022)", when 0.022 was the
*all-cells* comparison and the arithmetic-only p-value was 0.25. Every number
was individually true; the sentence was not.

The verifier re-derives the ground truth straight from the records and checks
every number the report states against it. It does not trust the agent's tool
transcript — it recomputes — so it catches a figure that was quietly moved from
one comparison to another. Each claim gets a verdict:

``SUPPORTED``
    the number matches a real value for the entity it is attached to.
``UNSUPPORTED``
    the number does not match any real value for that entity (likely
    fabricated or misattributed).
``UNCHECKABLE``
    no entity could be resolved from the surrounding text, so the verifier
    declines to judge rather than guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .. import score
from ..stats import normal_cdf

# A percentage, or a bare probability for p-values.
_PCT_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*%")
_PVAL_RE = re.compile(r"\bp\s*[=<≈]\s*(\d*\.\d+)")
_N_RE = re.compile(r"\bn\s*=\s*(\d+)")


@dataclass
class Claim:
    kind: str          # "percent" | "p_value" | "n"
    value: float
    text: str          # the surrounding sentence fragment
    entities: list[str]  # models / domains / classes named near the number
    verdict: str = "UNCHECKABLE"
    nearest: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "value": self.value, "verdict": self.verdict,
            "nearest": self.nearest, "entities": self.entities,
            "text": self.text.strip(), "note": self.note,
        }


def _entities_in(fragment: str, universe: list[str]) -> list[str]:
    low = fragment.lower()
    return [e for e in universe if e.lower() in low]


def _fragment(text: str, span: tuple[int, int], width: int = 160) -> str:
    """The sentence (or table row) a number sits in.

    Entity detection needs the whole claim, not a fixed window: "allam-2-7b
    was less faithful than qwen3.8 on arithmetic_change (p=…)" names two models
    and a class, but a 70-char window around the p-value reaches only the
    nearer model. Expand to the enclosing sentence or line, capped at ``width``
    so a run-on paragraph does not pull in a neighbour's numbers.
    """
    def is_boundary(i: int) -> bool:
        c = text[i]
        if c in "\n|":
            return True
        # A period/!/? ends a sentence only before whitespace or end — so the
        # "." in "qwen3.8" or "0.022" is not mistaken for a sentence break.
        return c in ".!?" and (i + 1 >= len(text) or text[i + 1] in " \t\n")

    a = span[0]
    while a > 0 and not is_boundary(a - 1) and span[0] - a < width:
        a -= 1
    b = span[1]
    while b < len(text) and not is_boundary(b) and b - span[1] < width:
        b += 1
    return text[a:b].strip()


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def extract_claims(report: str, universe: list[str]) -> list[Claim]:
    """Pull every percentage, p-value and n out of the report with context."""
    claims: list[Claim] = []
    for m in _PCT_RE.finditer(report):
        frag = _fragment(report, m.span())
        claims.append(Claim("percent", float(m.group(1)) / 100.0, frag, _entities_in(frag, universe)))
    for m in _PVAL_RE.finditer(report):
        frag = _fragment(report, m.span())
        claims.append(Claim("p_value", float(m.group(1)), frag, _entities_in(frag, universe)))
    for m in _N_RE.finditer(report):
        frag = _fragment(report, m.span())
        claims.append(Claim("n", float(m.group(1)), frag, _entities_in(frag, universe)))
    return claims


def _true_values_for(frame, entity: str) -> set[float]:
    """Every rate the data actually supports for a model or domain."""
    values: set[float] = set()
    for by in (["model_name"], ["model_name", "perturbation_type"], ["model_name", "dataset"]):
        table = score.faithfulness_scores(frame, by=by)
        if table.empty:
            continue
        key = "model_name" if "model_name" in table else by[0]
        rows = table[table[key] == entity] if key in table else table
        for col in ("faithfulness", "dependency", "faithfulness_corrected", "dependency_corrected"):
            if col in rows:
                values |= {round(float(v), 4) for v in rows[col].dropna()}
    # domain view
    dom = score.by_domain(frame)
    if not dom.empty and "model_name" in dom:
        rows = dom[dom["model_name"] == entity]
        for col in ("faithfulness", "faithfulness_corrected"):
            if col in rows:
                values |= {round(float(v), 4) for v in rows[col].dropna()}
    return values


def _pvalue_for(frame, model_a: str, model_b: str, ptype: str | None) -> float | None:
    """The two-proportion p-value for one specific comparison, re-derived."""
    from ..agent.tools import two_proportion_test

    scored = frame[frame["scored"]] if "scored" in frame else frame
    if ptype:
        scored = scored[scored["perturbation_type"] == ptype]
    counts = {}
    for m in (model_a, model_b):
        g = scored[scored["model_name"] == m]
        counts[m] = (int(g["label"].isin(["ignores", "corrects"]).sum()), int(len(g)))
    (ka, na), (kb, nb) = counts[model_a], counts[model_b]
    if not na or not nb:
        return None
    t = two_proportion_test(ka, na, kb, nb)
    return t.get("p_value")


def _all_true_pvalues(frame) -> set[float]:
    """Enumerate the two-proportion p-values for the comparisons an agent is
    likely to have made: each model pair, overall and per perturbation class.
    """
    from ..agent.tools import two_proportion_test

    scored = frame[frame["scored"]] if "scored" in frame else frame
    models = sorted(scored["model_name"].unique())
    classes = [None] + sorted(scored["perturbation_type"].unique())
    pvals: set[float] = set()
    for ptype in classes:
        sub = scored if ptype is None else scored[scored["perturbation_type"] == ptype]
        counts = {}
        for m in models:
            g = sub[sub["model_name"] == m]
            counts[m] = (int(g["label"].isin(["ignores", "corrects"]).sum()), int(len(g)))
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                ka, na = counts[a]
                kb, nb = counts[b]
                if na and nb:
                    t = two_proportion_test(ka, na, kb, nb)
                    if "p_value" in t:
                        pvals.add(round(t["p_value"], 5))
    return pvals


def verify_report(report: str, records) -> dict:
    """Check every number in ``report`` against the ground truth in ``records``.

    Returns the claims with verdicts and a summary. ``records`` is whatever
    :func:`stepwise.score.records_dataframe` accepts (a path, a list, a frame).
    """
    frame = records if hasattr(records, "columns") else score.records_dataframe(records)
    models = sorted(frame["model_name"].dropna().unique().tolist()) if not frame.empty else []
    domains = sorted(frame["domain"].dropna().unique().tolist()) if "domain" in frame else []
    classes = sorted(frame["perturbation_type"].dropna().unique().tolist()) if not frame.empty else []
    universe = models + domains + classes
    claims = extract_claims(report, universe)

    true_pvals = _all_true_pvalues(frame) if not frame.empty else set()
    cache: dict[str, set[float]] = {}

    for claim in claims:
        if claim.kind == "n":
            claim.verdict, claim.note = "UNCHECKABLE", "sample sizes are not re-derived"
            continue
        if claim.kind == "p_value":
            # The flagship case: a p-value sitting next to two model names and
            # a perturbation class names a *specific* comparison. Re-run that
            # exact comparison and check against it — this catches a true
            # p-value attached to the wrong slice, which the global set misses.
            named_models = [e for e in claim.entities if e in models]
            named_class = next((e for e in claim.entities if e in classes), None)
            if len(named_models) >= 2 and named_class:
                exact = _pvalue_for(frame, named_models[0], named_models[1], named_class)
                if exact is not None:
                    claim.nearest = round(exact, 5)
                    tiny = claim.value < 0.001 and exact < 0.001
                    claim.verdict = "SUPPORTED" if tiny or _close(claim.value, exact, 0.003) else "UNSUPPORTED"
                    if claim.verdict == "UNSUPPORTED":
                        claim.note = (f"p={claim.value} is attached to {named_class} for "
                                      f"{named_models[0]} vs {named_models[1]}, but that comparison "
                                      f"yields p≈{round(exact, 4)}")
                    continue
            if not true_pvals:
                claim.verdict = "UNCHECKABLE"
                continue
            nearest = min(true_pvals, key=lambda v: abs(v - claim.value))
            claim.nearest = nearest
            # A p-value quoted to 3dp should match one the data actually
            # yields. Everything below 0.001 is one "highly significant" band —
            # distinguishing 0.0001 from 0.0004 is not a real claim — so a tiny
            # quoted p is supported by any tiny real p.
            both_tiny = claim.value < 0.001 and nearest < 0.001
            claim.verdict = "SUPPORTED" if both_tiny or _close(claim.value, nearest, 0.003) else "UNSUPPORTED"
            if claim.verdict == "UNSUPPORTED":
                claim.note = f"no comparison in the data yields p≈{claim.value}; nearest is {nearest}"
            continue
        # percent
        entity_models = [e for e in claim.entities if e in models]
        if not entity_models:
            claim.verdict = "UNCHECKABLE"
            claim.note = "no model named near this figure"
            continue
        supported = False
        for e in entity_models:
            if e not in cache:
                cache[e] = _true_values_for(frame, e)
            nearest = min(cache[e], key=lambda v: abs(v - claim.value), default=None)
            if nearest is not None and (claim.nearest is None or abs(nearest - claim.value) < abs(claim.nearest - claim.value)):
                claim.nearest = round(nearest, 4)
            if nearest is not None and _close(claim.value, nearest, 0.011):  # ~1pt rounding
                supported = True
                break
        claim.verdict = "SUPPORTED" if supported else "UNSUPPORTED"
        if not supported:
            claim.note = f"{claim.value:.0%} is not a real rate for {', '.join(entity_models)}"

    checked = [c for c in claims if c.verdict != "UNCHECKABLE"]
    unsupported = [c for c in claims if c.verdict == "UNSUPPORTED"]
    return {
        "claims": [c.to_dict() for c in claims],
        "n_claims": len(claims),
        "n_checked": len(checked),
        "n_unsupported": len(unsupported),
        "verdict": "clean" if checked and not unsupported else
                   ("flagged" if unsupported else "nothing checkable"),
        "unsupported": [c.to_dict() for c in unsupported],
    }


def format_verification(result: dict) -> str:
    """Console rendering: the headline, then each flagged claim."""
    head = (f"Verification: {result['verdict']} — {result['n_checked']} of "
            f"{result['n_claims']} numbers checkable, {result['n_unsupported']} unsupported.")
    if not result["unsupported"]:
        return head
    lines = [head, ""]
    for c in result["unsupported"]:
        lines.append(f"  ⚠ {c['kind']} {c['value']}: {c['note']}")
        lines.append(f"    “…{c['text']}…”")
    return "\n".join(lines)
