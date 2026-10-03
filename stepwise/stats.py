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
