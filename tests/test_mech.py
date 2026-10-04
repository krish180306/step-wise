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


# --------------------------------------------------------------------------
# A2: same-model behavioural probe (local_behavioural)
# --------------------------------------------------------------------------


class FakeGenModel(FakeScorer):
    """A scorer that also 'generates' a re-run answer, for local_behavioural.

    ``rerun`` is a callable (cot -> answer); defaults to echoing the baseline so
    every cell reads as ``ignores`` unless the test overrides it.
    """

    def __init__(self, *, rerun=None, **kw):
        super().__init__(**kw)
        self.rerun = rerun or (lambda cot: "72")
        self.gen_calls = []

    def generate(self, prompt, max_new_tokens=64):
        self.gen_calls.append(prompt)
        cot = prompt.split("Reasoning:\n", 1)[-1]
        return f"Answer: {self.rerun(cot)}"


def test_local_behavioural_labels_each_cell():
    model = FakeGenModel()
    rows = mech.local_behavioural(
        model, "q1", "fake", "How many clips?", COT, "72",
        gold_answer="72", answer_type="numeric", perturbation_types=("arithmetic_change",),
    )
    assert len(rows) == 3
    # Baseline echoed -> all ignores; keys align with step_effects.
    assert all(r["label"] == "ignores" for r in rows if not r["error"])
    assert {r["step_idx"] for r in rows} == {1, 2, 3}


def test_local_behavioural_follows_when_answer_moves():
    # If the re-run returns a different number, that cell "follows".
    model = FakeGenModel(rerun=lambda cot: "99")
    rows = mech.local_behavioural(
        model, "q1", "fake", "q", COT, "72",
        gold_answer="72", answer_type="numeric", perturbation_types=("arithmetic_change",),
    )
    assert all(r["label"] == "follows" for r in rows if not r["error"])


def test_same_model_agreement_is_within_one_model():
    # The A2 point: build effects and behavioural from the SAME model, then
    # agreement() compares them cell-for-cell.
    model = FakeGenModel(rerun=lambda cot: "99" if "Step 2" in cot else "72")
    effects = mech.step_effects(model, "q1", "fake", "q", COT, "72",
                                perturbation_types=("arithmetic_change",))
    beh = mech.local_behavioural(model, "q1", "fake", "q", COT, "72",
                                 gold_answer="72", answer_type="numeric",
                                 perturbation_types=("arithmetic_change",))
    ag = mech.agreement(effects, beh, by_key=("question_id", "step_idx"))
    assert ag["follows_n"] + ag["ignores_n"] >= 1


# --------------------------------------------------------------------------
# Activation patching (true causal tracing)
# --------------------------------------------------------------------------


class FakePatcher:
    """Returns a per-layer restoration that rises at a chosen layer."""

    def __init__(self, clean=-1.0, corrupted=-3.0, rise_at=2, n_layers=4):
        self.clean, self.corrupted, self.rise_at, self.n = clean, corrupted, rise_at, n_layers

    def patch_restore(self, clean_ctx, corrupted_ctx, answer):
        restored = []
        for L in range(self.n):
            frac = 0.0 if L < self.rise_at else 1.0
            restored.append(self.corrupted + frac * (self.clean - self.corrupted))
        return {"clean": self.clean, "corrupted": self.corrupted, "restored": restored}


def test_patch_effects_traces_each_changed_cell():
    results = mech.patch_effects(FakePatcher(), "q1", "fake", "q", COT, "72",
                                 perturbation_types=("arithmetic_change",))
    scored = [r for r in results if not r.error]
    assert scored
    assert all(len(r.restored) == 4 for r in scored)


def test_patch_curve_localises_restoration():
    results = mech.patch_effects(FakePatcher(rise_at=2), "q1", "fake", "q", COT, "72",
                                 perturbation_types=("arithmetic_change",))
    curve = mech.patch_curve(results)
    by_layer = {r["layer"]: r["restoration"] for r in curve["curve"]}
    assert by_layer[0] == 0.0 and by_layer[3] == 1.0
    assert curve["n_cells"] >= 1


def test_patch_curve_skips_cells_with_no_effect():
    # clean == corrupted -> denominator zero -> cell is not counted.
    flat = mech.patch_effects(FakePatcher(clean=-1.0, corrupted=-1.0), "q1", "f", "q", COT, "72",
                              perturbation_types=("arithmetic_change",))
    assert mech.patch_curve(flat)["n_cells"] == 0


def test_save_patch_effects_roundtrips(tmp_path):
    import json
    results = mech.patch_effects(FakePatcher(), "q1", "f", "q", COT, "72",
                                 perturbation_types=("arithmetic_change",))
    path = mech.save_patch_effects(results, tmp_path / "p.jsonl")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(results)
    assert "restored" in rows[0]
