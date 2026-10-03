"""Validating the instrument: is the automatic label right?

Every faithfulness number rests on two automatic judgements — the perturbation
was a clean minimal edit, and the re-run answer "matches" the original. If
those are wrong 20% of the time, so is the headline. This module measures how
often they are wrong, instead of waving at it in a caveat:

- :func:`cohen_kappa` and :func:`three_way_agreement` compare the automatic
  label against human annotations and against an independent LLM judge.
  Agreement beyond chance (κ) is the honest statement of label reliability.
- :func:`label_regression` fits a logistic model for "follows" against step
  position, whether it is the answer step, perturbation class and model, with
  cluster-robust standard errors by question. It separates confounds the
  cross-tabs cannot — most importantly, whether "position matters" is really
  just "the final step matters".

``statsmodels`` is imported lazily inside the regression, so the rest of the
module (the agreement statistics, which need only the standard library and
pandas) works without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import LABELS


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------


def cohen_kappa(a, b, labels=LABELS) -> dict:
    """Cohen's κ between two sets of labels for the same items.

    κ = (p_observed − p_chance) / (1 − p_chance): agreement corrected for the
    agreement you would get by chance given each rater's label frequencies.
    1.0 is perfect, 0 is chance, below 0 is worse than chance. The usual
    reading: <0.2 slight, 0.2–0.4 fair, 0.4–0.6 moderate, 0.6–0.8 substantial.
    """
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        return {"kappa": None, "n": 0, "note": "need equal, non-empty label lists"}
    labels = list(labels)
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    freq_a = {l: a.count(l) / n for l in labels}
    freq_b = {l: b.count(l) / n for l in labels}
    chance = sum(freq_a[l] * freq_b[l] for l in labels)
    if chance >= 1.0:
        kappa = 1.0 if observed >= 1.0 else 0.0
    else:
        kappa = (observed - chance) / (1.0 - chance)
    return {
        "kappa": round(kappa, 4), "observed_agreement": round(observed, 4),
        "chance_agreement": round(chance, 4), "n": n,
        "strength": _kappa_strength(kappa),
    }


def _kappa_strength(k: float) -> str:
    for threshold, name in [(0.8, "almost perfect"), (0.6, "substantial"),
                            (0.4, "moderate"), (0.2, "fair"), (-1, "slight or none")]:
        if k >= threshold:
            return name
    return "slight or none"


def three_way_agreement(rows) -> dict:
    """Agreement among automatic / human / judge labels, pairwise.

    ``rows`` is an iterable of dicts carrying some of ``auto_label``,
    ``human_label``, ``judge_label`` for the same cells. Each pair is compared
    over the cells where both are present.
    """
    frame = pd.DataFrame(list(rows))
    pairs = [("auto_label", "human_label"), ("auto_label", "judge_label"),
             ("human_label", "judge_label")]
    out = {}
    for x, y in pairs:
        if x in frame and y in frame:
            both = frame[[x, y]].dropna()
            if not both.empty:
                out[f"{x.split('_')[0]}_vs_{y.split('_')[0]}"] = cohen_kappa(both[x], both[y])
    return out


# --------------------------------------------------------------------------
# Human annotation store
# --------------------------------------------------------------------------


def cell_key(record: dict) -> str:
    """Stable id for one cell, so a human label attaches to the right record."""
    return "|".join(str(record.get(k, "")) for k in
                    ("question_id", "model_name", "step_idx", "perturbation_type"))


@dataclass
class AnnotationStore:
    """Human labels for cells, persisted as a small JSON file."""

    path: Path
    labels: dict[str, str]

    @classmethod
    def load(cls, path) -> "AnnotationStore":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return cls(path=path, labels=data.get("labels", {}))

    def set(self, key: str, label: str) -> None:
        if label in LABELS:
            self.labels[key] = label
        elif label in ("", None):
            self.labels.pop(key, None)
        else:
            raise ValueError(f"label must be one of {LABELS} (or empty to clear)")

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"labels": self.labels}, indent=2), encoding="utf-8")
        return self.path

    def __len__(self) -> int:
        return len(self.labels)


def validate_against_human(records, store: AnnotationStore) -> dict:
    """κ between the automatic label and the human label, over annotated cells."""
    auto, human = [], []
    for r in records:
        r = r if isinstance(r, dict) else r.to_dict()
        if not r.get("label"):
            continue
        key = cell_key(r)
        if key in store.labels:
            auto.append(r["label"])
            human.append(store.labels[key])
    result = cohen_kappa(auto, human)
    result["annotated"] = len(human)
    # Where they disagree, which way.
    disagreements = [(a, h) for a, h in zip(auto, human) if a != h]
    result["disagreements"] = len(disagreements)
    return result


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------


def label_regression(records) -> dict:
    """Logistic regression for P(follows), with cluster-robust SEs by question.

    Separates effects the cross-tabs confound: does step *position* predict
    following once ``is_answer_step`` is in the model, or was "position" just
    "the final step"? Coefficients are log-odds; a positive coefficient raises
    the probability of following. Standard errors are clustered by question, so
    the non-independence of a question's cells does not understate them.

    Returns per-term estimates or a clear reason it could not fit (too few
    cells, no variation in the outcome, statsmodels missing).
    """
    try:
        import numpy as np
        import statsmodels.formula.api as smf
    except ImportError:
        return {"error": "regression needs statsmodels and numpy: pip install statsmodels"}

    from . import score

    frame = records if hasattr(records, "columns") else score.records_dataframe(records)
    frame = frame[frame["scored"]].copy() if "scored" in frame else frame.copy()
    if len(frame) < 30:
        return {"error": f"only {len(frame)} scored cells; regression needs more to be meaningful"}

    frame["follows"] = (frame["label"] == "follows").astype(int)
    if frame["follows"].nunique() < 2:
        return {"error": "no variation in the outcome (every cell has the same label)"}

    frame["answer_step"] = frame.get("is_answer_step", frame["step_idx"] >= frame["n_steps"]).astype(int)
    frame["position"] = frame.get("step_position", 0.0)
    frame["log_len"] = np.log1p(frame["original_step"].fillna("").str.len()) if "original_step" in frame else 0.0

    terms = ["position", "answer_step"]
    if frame["perturbation_type"].nunique() > 1:
        terms.append("C(perturbation_type)")
    if frame["model_name"].nunique() > 1:
        terms.append("C(model_name)")
    formula = "follows ~ " + " + ".join(terms)

    try:
        spec = smf.logit(formula, data=frame)
        if "question_id" in frame:
            # Cluster-robust standard errors by question, passed at fit time
            # (statsmodels fits the covariance here, not via a later call).
            model = spec.fit(disp=False, maxiter=100, cov_type="cluster",
                             cov_kwds={"groups": frame["question_id"].to_numpy()})
        else:
            model = spec.fit(disp=False, maxiter=100)
    except Exception as exc:  # noqa: BLE001 - perfect separation etc.
        return {"error": f"model did not converge ({type(exc).__name__}: {exc})"}

    params, pvals, conf = model.params, model.pvalues, np.asarray(model.conf_int())
    out = []
    for i, name in enumerate(params.index):
        coef = float(params.iloc[i])
        out.append({
            "term": name,
            "coef_log_odds": round(coef, 4),
            # Cap the reported odds ratio: under separation exp(coef) overflows
            # to a meaningless 1e30, which reads as a finding when it is an
            # artefact. Report the direction and flag it instead.
            "odds_ratio": round(float(np.exp(coef)), 4) if abs(coef) < 15 else None,
            "p_value": round(float(pvals.iloc[i]), 5),
            "ci_low": round(float(conf[i, 0]), 4),
            "ci_high": round(float(conf[i, 1]), 4),
        })
    converged = bool(getattr(model, "mle_retvals", {}).get("converged", True))
    separated = any(abs(t["coef_log_odds"]) > 15 for t in out)
    result = {
        "formula": formula, "n": int(len(frame)),
        "n_questions": int(frame["question_id"].nunique()) if "question_id" in frame else None,
        "pseudo_r2": round(float(model.prsquared), 4),
        "terms": out, "converged": converged, "separation": separated,
        "note": "coefficients are log-odds of 'follows'; SEs clustered by question",
    }
    if separated or not converged:
        result["warning"] = (
            "perfect or near-perfect separation — too few questions for stable "
            "estimates; the signs are meaningful but the magnitudes are not. "
            "Re-run on more questions."
        )
    return result


def format_regression(result: dict) -> str:
    if "error" in result:
        return f"Regression: {result['error']}"
    lines = [f"Regression for P(follows) · {result['n']} cells, "
             f"{result.get('n_questions')} questions · pseudo-R² {result['pseudo_r2']}"]
    if result.get("warning"):
        lines.append(f"  ⚠ {result['warning']}")
    lines.append(f"  {'term':<34}{'odds ratio':>12}{'p':>9}")
    for t in result["terms"]:
        star = " *" if t["p_value"] < 0.05 else ""
        orv = "  (sep.)" if t["odds_ratio"] is None else f"{t['odds_ratio']:>12.3f}"
        lines.append(f"  {t['term']:<34}{orv:>12}{t['p_value']:>9.4f}{star}")
    return "\n".join(lines)
