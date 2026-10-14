"""The statistics that make a faithfulness number mean something.

Three corrections turn the raw label counts into numbers you can compare and
quote. Each exists because the naive version is biased in a direction that
would quietly flatter the hypothesis the project is testing.

**Chance correction.** "Ignores" means the re-run answer equalled the
original. But if a corrupted step genuinely confuses the model and it falls
back to guessing, the guess *coincidentally* matches the original with
probability 1/K, where K is how many answers it is choosing between. So a
yes/no task has a 50% ignore floor and a five-way choice a 20% floor, with no
faithfulness involved at all. Comparing a numeric task (floor ≈ 0) against a
boolean one without correcting for this makes arithmetic look more
reasoning-dependent purely because of its answer format — which is exactly the
direction the "CoT helps math more" hypothesis predicts, so it would confirm
itself for free. :func:`corrected_rate` removes that floor.

**Clustered bootstrap.** The four cells of one question share a chain of
thought, so they are not independent observations. A confidence interval that
treats them as independent is too narrow, and a significance test over them
overstates its certainty. :func:`clustered_bootstrap_ci` resamples whole
*questions* with replacement, which keeps each question's cells together and
widens the interval to honesty.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

import pandas as pd

#: How many answers the model is effectively choosing between, by answer type.
#: ``0`` means "too many to matter" — numeric and free-span answers almost
#: never collide by chance, so they get no correction.
_DEFAULT_OPTIONS = {"boolean": 2, "numeric": 0, "span": 0, "label": 0}


def chance_rate(answer_type: str, n_choices: int = 0) -> float:
    """P(a disrupted model's guess matches the original answer by luck).

    ``n_choices`` wins when it is known (a 5-option MCQ gives 0.2); otherwise
    the answer type sets the default. An open-ended answer type returns 0.0,
    meaning "no correction needed".
    """
    if n_choices and n_choices > 1:
        return 1.0 / n_choices
    options = _DEFAULT_OPTIONS.get(answer_type, 0)
    return 1.0 / options if options > 1 else 0.0


def corrected_rate(observed: float, chance: float) -> float:
    """Rescale an observed ignore rate so pure guessing scores 0, not ``chance``.

    ``(observed - chance) / (1 - chance)`` is the standard chance correction
    (the same arithmetic as Cohen's kappa against a chance baseline). The
    result is clamped to [0, 1]; a value below chance would otherwise come out
    negative, which is real but not meaningful to average.
    """
    if chance <= 0:
        return observed
    if chance >= 1:
        return observed
    return max(0.0, min(1.0, (observed - chance) / (1.0 - chance)))


def mean_chance(frame: pd.DataFrame) -> float:
    """The average chance floor over a set of cells.

    Cells in one group can mix answer types (a pooled "all datasets" row), so
    the floor is the mean of each cell's own chance rate rather than a single
    type's value.
    """
    if frame.empty:
        return 0.0
    types = frame.get("answer_type")
    choices = frame.get("n_choices")
    if types is None:
        return 0.0
    rates = [
        chance_rate(t, int(c) if c and not pd.isna(c) else 0)
        for t, c in zip(types, choices if choices is not None else [0] * len(types))
    ]
    return sum(rates) / len(rates) if rates else 0.0


def clustered_bootstrap_ci(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    cluster: str = "question_id",
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """A confidence interval that respects clustering, by resampling clusters.

    Returns ``(point, low, high)``: the statistic on the real data, and the
    percentile interval from resampling whole clusters with replacement. When
    there is only one cluster — or none — the interval collapses to the point
    estimate, because there is nothing to resample over and a fake interval
    would be worse than an honest "n too small".
    """
    point = statistic(frame)
    if frame.empty or cluster not in frame.columns:
        return point, point, point

    groups = [g for _, g in frame.groupby(cluster, observed=True)]
    if len(groups) < 2:
        return point, point, point

    rng = random.Random(seed)
    k = len(groups)
    draws: list[float] = []
    for _ in range(resamples):
        picked = [groups[rng.randrange(k)] for _ in range(k)]
        sample = pd.concat(picked, ignore_index=True)
        try:
            value = statistic(sample)
        except (ZeroDivisionError, ValueError):
            continue
        if value == value:  # drop NaN
            draws.append(value)
    if not draws:
        return point, point, point

    draws.sort()
    tail = (1.0 - confidence) / 2.0
    lo = draws[max(0, int(tail * len(draws)))]
    hi = draws[min(len(draws) - 1, int((1.0 - tail) * len(draws)))]
    return round(point, 4), round(lo, 4), round(hi, 4)


def faithfulness_of(frame: pd.DataFrame, *, labels=("ignores", "corrects")) -> float:
    """(ignores + corrects) / scored, the statistic the bootstrap resamples."""
    scored = frame[frame["label"].isin(("follows", "ignores", "corrects"))] if "label" in frame else frame
    n = len(scored)
    if n == 0:
        return float("nan")
    return float(scored["label"].isin(labels).sum()) / n


def ratio_bootstrap_ci(
    frame: pd.DataFrame,
    *,
    hit_labels=("ignores", "corrects"),
    cluster: str = "question_id",
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Clustered CI for a rate, specialised and vectorised.

    Faithfulness is sum(hits) / sum(n) over clusters, so a bootstrap draw is
    just a random multiset of per-cluster (hits, n) pairs — no DataFrame
    rebuild per resample. This is the same interval
    :func:`clustered_bootstrap_ci` would give for :func:`faithfulness_of`, a
    few hundred times faster, which is what makes it cheap enough to run on
    every dashboard request.
    """
    import numpy as np

    scored = frame[frame["label"].isin(("follows", "ignores", "corrects"))] if "label" in frame else frame
    if scored.empty:
        return float("nan"), float("nan"), float("nan")
    grp = scored.assign(_hit=scored["label"].isin(hit_labels).astype(int)).groupby(
        cluster, observed=True
    )
    hits = grp["_hit"].sum().to_numpy(dtype=float)
    sizes = grp.size().to_numpy(dtype=float)
    point = hits.sum() / sizes.sum()
    k = len(hits)
    if k < 2:
        return round(float(point), 4), round(float(point), 4), round(float(point), 4)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(resamples, k))
    draws = hits[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return round(float(point), 4), round(float(lo), 4), round(float(hi), 4)


def normal_cdf(z: float) -> float:
    import math

    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def benjamini_hochberg(pvalues: Sequence[float], *, alpha: float = 0.05) -> dict:
    """Control the false-discovery rate across a family of tests.

    A run compares many model x class x domain cells, each with its own p-value.
    Testing them all at 0.05 means ~1 in 20 "findings" is noise. Benjamini-
    Hochberg adjusts for that: it returns a q-value per test (the FDR-adjusted
    p-value, order-preserving and monotone) and which tests stay significant at
    the chosen FDR. Report the q-value, not the raw p, once more than a couple
    of tests are on the table.
    """
    ps = [float(p) for p in pvalues]
    m = len(ps)
    if m == 0:
        return {"qvalues": [], "rejected": [], "alpha": alpha, "n": 0, "n_significant": 0}
    order = sorted(range(m), key=lambda i: ps[i])
    qvals = [1.0] * m
    prev = 1.0
    # Walk from the largest p to the smallest, enforcing monotonicity.
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, ps[i] * m / rank)
        qvals[i] = min(1.0, prev)
    rejected = [qvals[i] <= alpha for i in range(m)]
    return {
        "qvalues": [round(q, 6) for q in qvals],
        "rejected": rejected,
        "alpha": alpha,
        "n": m,
        "n_significant": int(sum(rejected)),
    }


def rate_diff_bootstrap_ci(
    frame: pd.DataFrame,
    group_col: str,
    a: str,
    b: str,
    *,
    hit_labels=("follows",),
    cluster: str = "question_id",
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Clustered CI for the difference in a rate between two groups.

    ``rate(a) - rate(b)`` where rate = sum(hits)/sum(n), resampling whole
    questions so the two groups move together under each draw (they answer the
    same questions). Used to decide whether one model's lead in a domain is real
    or within noise: the difference is significant when the interval excludes 0.
    """
    import numpy as np

    scored = frame[frame["label"].isin(("follows", "ignores", "corrects"))] if "label" in frame else frame
    fa = scored[scored[group_col] == a]
    fb = scored[scored[group_col] == b]

    def agg(f):
        g = f.assign(_h=f["label"].isin(hit_labels).astype(int)).groupby(cluster, observed=True)
        return g["_h"].sum(), g.size()

    ha, na = agg(fa)
    hb, nb = agg(fb)
    clusters = sorted(set(ha.index) | set(hb.index))
    if len(clusters) < 2:
        da = ha.sum() / na.sum() if na.sum() else float("nan")
        db = hb.sum() / nb.sum() if nb.sum() else float("nan")
        p = (da - db) if da == da and db == db else float("nan")
        return round(p, 4) if p == p else p, p, p
    HA = np.array([ha.get(c, 0) for c in clusters], float)
    NA = np.array([na.get(c, 0) for c in clusters], float)
    HB = np.array([hb.get(c, 0) for c in clusters], float)
    NB = np.array([nb.get(c, 0) for c in clusters], float)
    point = (HA.sum() / NA.sum() if NA.sum() else np.nan) - (HB.sum() / NB.sum() if NB.sum() else np.nan)
    k = len(clusters)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(resamples, k))
    sa, sb = NA[idx].sum(1), NB[idx].sum(1)
    da = np.divide(HA[idx].sum(1), sa, out=np.full(resamples, np.nan), where=sa > 0)
    db = np.divide(HB[idx].sum(1), sb, out=np.full(resamples, np.nan), where=sb > 0)
    draws = (da - db)
    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        return round(float(point), 4), float("nan"), float("nan")
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return round(float(point), 4), round(float(lo), 4), round(float(hi), 4)

# --------------------------------------------------------------------------
# How big does the run need to be?
# --------------------------------------------------------------------------
#
# The project's standing limitation is scale: the headline needs 50+ questions
# and the committed numbers come from 2. "How many do I need" has a real
# answer, and getting it wrong is the easiest way to run up a bill for a
# result that was never going to reach significance.
#
# The subtlety is the same one ``clustered_bootstrap_ci`` exists for. The unit
# of sampling is the *question*, not the cell: one question yields many cells
# that share a chain of thought, so they carry less information than the same
# number of independent observations. The standard sample-size formula assumes
# independence, and run on cells it will under-estimate what is needed by the
# design effect, which is routinely 2-4x here.


def design_effect(cells_per_question: float, icc: float) -> float:
    """How much clustering inflates the variance: ``1 + (m - 1) * icc``.

    ``m`` is cells per question and ``icc`` the intra-cluster correlation —
    how much cells from one question resemble each other. At ``icc = 0`` the
    cells are independent and the effect is 1; at ``icc = 1`` a question's
    cells are perfectly redundant and ``m`` of them count as one.

    This is the factor the naive sample size must be multiplied by, and the
    factor a cell-level confidence interval is too narrow by.
    """
    m = max(1.0, float(cells_per_question))
    icc = min(1.0, max(0.0, float(icc)))
    return 1.0 + (m - 1.0) * icc


def observed_icc(frame: pd.DataFrame, *, cluster: str = "question_id",
                 hit_labels=("follows",)) -> float:
    """Estimate the intra-cluster correlation of the follow indicator.

    One-way random-effects ANOVA estimator on the 0/1 outcome: the share of
    total variance that sits *between* questions rather than within them. A
    high value means a question's cells mostly agree — brittleness is a
    property of the question — and that each extra cell on the same question
    buys little new information.

    Clamped to [0, 1]: the estimator can go slightly negative on small or
    well-mixed data, which means "no detectable clustering", not a negative
    correlation. Returns 0.0 when there is too little data to estimate.
    """
    if frame is None or len(frame) == 0 or cluster not in getattr(frame, "columns", []):
        return 0.0
    scored = frame[frame["label"].isin(("follows", "ignores", "corrects"))] if "label" in frame else frame
    if len(scored) < 2:
        return 0.0
    y = scored["label"].isin(hit_labels).astype(float)
    groups = [g for _, g in y.groupby(scored[cluster], observed=True)]
    groups = [g for g in groups if len(g) > 0]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return 0.0
    grand = float(y.mean())
    ms_between = sum(len(g) * (float(g.mean()) - grand) ** 2 for g in groups) / (k - 1)
    ms_within = sum(
        float(((g - g.mean()) ** 2).sum()) for g in groups
    ) / (n - k)
    # Average cluster size, the balanced-design approximation.
    m0 = n / k
    denominator = ms_between + (m0 - 1.0) * ms_within
    if denominator <= 0:
        return 0.0
    icc = (ms_between - ms_within) / denominator
    return round(min(1.0, max(0.0, float(icc))), 4)


#: Two-sided z for the usual alpha levels, and one-sided z for the usual power
#: levels. Hardcoded rather than inverting the normal CDF, because these are
#: the only values anyone asks for and an approximate inverse is one more
#: thing that can be subtly wrong.
_Z_ALPHA_TWO_SIDED = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}
_Z_POWER = {0.70: 0.5244, 0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}


def _z(table: dict, value: float, name: str) -> float:
    key = min(table, key=lambda k: abs(k - value))
    if abs(key - value) > 1e-9:
        raise ValueError(
            f"{name}={value} not tabulated; choose one of {sorted(table)}"
        )
    return table[key]


def required_questions(
    p1: float,
    p2: float,
    *,
    cells_per_question: float = 8.0,
    icc: float = 0.3,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """How many questions to separate a follow-rate ``p1`` from a control ``p2``.

    The two-proportion sample size, then inflated by :func:`design_effect` and
    divided by the cells each question contributes. ``p1`` is the rate you
    expect on the consequential classes and ``p2`` the irrelevant-noise
    control — the comparison the whole design rests on, since a follow-rate
    means nothing except against its own control.

    Returns the cell and question counts, the design effect used, and an
    estimate of the API calls a run that size costs, which is usually the
    number that decides whether the experiment happens.
    """
    p1 = min(1.0, max(0.0, float(p1)))
    p2 = min(1.0, max(0.0, float(p2)))
    if p1 == p2:
        return {
            "detectable": False,
            "reason": "p1 == p2: no effect to detect",
            "questions": None, "cells_per_arm": None,
            "design_effect": design_effect(cells_per_question, icc),
        }
    za = _z(_Z_ALPHA_TWO_SIDED, alpha, "alpha")
    zb = _z(_Z_POWER, power, "power")
    pooled = (p1 + p2) / 2.0
    numerator = (
        za * (2 * pooled * (1 - pooled)) ** 0.5
        + zb * (p1 * (1 - p1) + p2 * (1 - p2)) ** 0.5
    ) ** 2
    cells_per_arm = numerator / (p1 - p2) ** 2
    deff = design_effect(cells_per_question, icc)
    cells_per_arm_clustered = cells_per_arm * deff
    # Each question contributes cells to both arms, split across the classes.
    per_question_per_arm = max(1.0, float(cells_per_question) / 2.0)
    questions = cells_per_arm_clustered / per_question_per_arm
    questions = int(-(-questions // 1))  # ceil
    return {
        "detectable": True,
        "p1": round(p1, 4),
        "p2": round(p2, 4),
        "alpha": alpha,
        "power": power,
        "icc": round(float(icc), 4),
        "cells_per_question": round(float(cells_per_question), 2),
        "design_effect": round(deff, 3),
        "cells_per_arm_independent": int(-(-cells_per_arm // 1)),
        "cells_per_arm": int(-(-cells_per_arm_clustered // 1)),
        "questions": questions,
        # One baseline elicitation per question per model, then one
        # perturbation write and one re-run per cell.
        "api_calls_per_model": int(questions * (1 + 2 * float(cells_per_question))),
    }


def precision_for(
    n_questions: int,
    *,
    rate: float = 0.5,
    cells_per_question: float = 8.0,
    icc: float = 0.3,
    confidence: float = 0.95,
) -> dict:
    """The 95% CI half-width a run of ``n_questions`` can expect to achieve.

    The complement of :func:`required_questions`: not "how many do I need" but
    "given what I can afford, how precise will the answer be". ``rate = 0.5``
    is the worst case and so the honest default.

    A half-width wider than the effect you are looking for is the signal to
    either scale up or stop quoting the point estimate as a finding — which is
    exactly the situation the project's 2-question run is in.
    """
    n_questions = max(0, int(n_questions))
    if n_questions < 2:
        return {"questions": n_questions, "half_width": None,
                "reason": "need at least 2 questions for an interval"}
    rate = min(1.0, max(0.0, float(rate)))
    cells = round(n_questions * max(1.0, float(cells_per_question)))
    deff = design_effect(cells_per_question, icc)
    effective_n = cells / deff
    za = _z(_Z_ALPHA_TWO_SIDED, round(1.0 - confidence, 4), "confidence")
    half = za * (rate * (1 - rate) / effective_n) ** 0.5
    return {
        "questions": n_questions,
        "cells": int(cells),  # already rounded above
        "design_effect": round(deff, 3),
        "effective_n": round(effective_n, 1),
        "rate": round(rate, 4),
        "half_width": round(float(half), 4),
        "ci": (round(max(0.0, rate - half), 4), round(min(1.0, rate + half), 4)),
    }


def power_plan(
    frame: "pd.DataFrame | None" = None,
    *,
    p1: float = 0.40,
    p2: float = 0.25,
    cells_per_question: float | None = None,
    icc: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """A sizing plan, measured from a real run wherever possible.

    Given a results frame, reads the two clustering parameters off the data —
    cells per question and the observed ICC — instead of guessing them, then
    reports both what the run can currently resolve and what it would take to
    resolve the target effect. That makes the answer specific to this design
    rather than a textbook number.
    """
    measured = {}
    if frame is not None and len(frame) > 0:
        scored = frame[frame["label"].isin(("follows", "ignores", "corrects"))]             if "label" in frame else frame
        if len(scored) and "question_id" in scored:
            n_questions = int(scored["question_id"].nunique())
            measured = {
                "questions": n_questions,
                "cells": int(len(scored)),
                "cells_per_question": round(len(scored) / max(1, n_questions), 2),
                "icc": observed_icc(scored),
                "observed_follow_rate": round(
                    float(scored["label"].eq("follows").mean()), 4
                ) if "label" in scored else None,
            }
    m = cells_per_question if cells_per_question is not None else measured.get(
        "cells_per_question", 8.0)
    i = icc if icc is not None else measured.get("icc", 0.3)
    return {
        "measured": measured or None,
        "current_precision": (
            precision_for(measured["questions"], rate=p1, cells_per_question=m, icc=i)
            if measured else None
        ),
        "target": required_questions(
            p1, p2, cells_per_question=m, icc=i, alpha=alpha, power=power
        ),
    }
