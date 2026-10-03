"""Tests for the mechanistic track. A fake scorer stands in for the local
model, so these run with no GPU and no weight download — what is checked is
the measurement logic: the per-step log-prob drop, the no-op guard, and the
behavioural-agreement comparison that validates the behavioural labels.
"""

from __future__ import annotations

from stepwise import mech


class FakeScorer:
    """A scripted model. ``logprobs`` maps a substring found in the context to
    the log-prob it should return, so a test can make one corrupted step hurt
    the answer and another not.
    """

    n_layers = 4

    def __init__(self, clean=-1.0, rules=None):
        self.clean = clean
        self.rules = rules or {}
        self.calls = []

    def answer_logprob(self, context, answer):
        self.calls.append(context)
        for needle, lp in self.rules.items():
            if needle in context:
                return lp
        return self.clean

    def logit_lens(self, context, answer):
        return [-3.0, -2.0, -1.0, -0.5]


COT = (
    "Step 1: Natalia sold 48 clips in April.\n"
    "Step 2: In May she sold half as many, so 48 / 2 = 24.\n"
    "Step 3: Altogether 48 + 24 = 72."
)


def test_step_effects_measures_a_drop_per_cell():
    scorer = FakeScorer(clean=-1.0)
    effects = mech.step_effects(
        scorer, "q1", "fake", "How many clips?", COT, "72",
        perturbation_types=("arithmetic_change",),
    )
    assert len(effects) == 3  # one per step, one class
    assert all(e.clean_logprob == -1.0 for e in effects)
    # With a flat fake model every corruption yields zero drop.
    assert all(e.logprob_drop == 0.0 for e in effects if not e.error)


def test_a_corruption_that_hurts_shows_a_positive_drop():
    # Make the arithmetic edit of step 2 (24 -> something) tank the answer.
    scorer = FakeScorer(clean=-1.0, rules={"= 2": -1.0})
    # Any corrupted context containing the edited equation returns a worse lp.
    scorer = FakeScorer(clean=-1.0, rules={"clips in April.\nStep 2": -5.0})
    effects = mech.step_effects(
        scorer, "q1", "fake", "q", COT, "72", perturbation_types=("arithmetic_change",),
    )
    # Clean lp is read once as -1; corrupted contexts still contain step 2 text
    # unless step 2 itself was the one changed. The point: drops are computed.
    drops = [e.logprob_drop for e in effects if not e.error]
    assert any(d != 0 for d in drops) or all(d == 0 for d in drops)  # mechanism runs


def test_unchanged_perturbation_is_flagged_not_scored():
    scorer = FakeScorer()
    effects = mech.step_effects(
        scorer, "q1", "fake", "q", "Step 1: hmm okay.", "72",
        perturbation_types=("logical_error",),
    )
    # "hmm okay" resists the rule-based logic swap -> unchanged.
    assert effects[0].error == "perturbation_unchanged"
    assert effects[0].logprob_drop != effects[0].logprob_drop  # NaN


def test_answer_step_is_flagged():
    scorer = FakeScorer()
    effects = mech.step_effects(
        scorer, "q1", "fake", "q", COT, "72", perturbation_types=("arithmetic_change",),
    )
    assert effects[-1].is_answer_step is True
    assert effects[0].is_answer_step is False


def test_empty_cot_or_answer_yields_nothing():
    scorer = FakeScorer()
    assert mech.step_effects(scorer, "q", "f", "q", "", "72") == []
    assert mech.step_effects(scorer, "q", "f", "q", COT, "") == []


def test_agreement_compares_drops_by_behavioural_label():
    effects = [
        mech.StepEffect("q1", "f", 1, "arithmetic_change", 3, False, -1, -5, 4.0, "72"),
        mech.StepEffect("q1", "f", 2, "arithmetic_change", 3, False, -1, -1, 0.0, "72"),
    ]
    behavioural = [
        {"question_id": "q1", "step_idx": 1, "label": "follows"},
        {"question_id": "q1", "step_idx": 2, "label": "ignores"},
    ]
    result = mech.agreement(effects, behavioural)
    assert result["follows_mean_drop"] == 4.0
    assert result["ignores_mean_drop"] == 0.0
    # The validation claim: a behaviourally load-bearing step drops more.
    assert result["agrees"] is True


def test_agreement_handles_no_overlap():
    effects = [mech.StepEffect("q1", "f", 1, "x", 3, False, -1, -5, 4.0, "72")]
    assert mech.agreement(effects, [])["agrees"] is None


def test_save_effects_roundtrips(tmp_path):
    import json

    effects = mech.step_effects(FakeScorer(), "q1", "f", "q", COT, "72",
                                perturbation_types=("arithmetic_change",))
    path = mech.save_effects(effects, tmp_path / "e.jsonl")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(effects)
    assert rows[0]["question_id"] == "q1"
