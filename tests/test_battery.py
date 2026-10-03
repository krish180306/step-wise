"""Tests for the faithfulness battery: the four tests, their helpers, and
their scoring. A scripted provider stands in for the model, so the whole
battery runs offline — what is checked is the prompt shaping, the answer
plumbing, and that each test's metric means what it claims.
"""

from __future__ import annotations

import pytest

from stepwise import battery, battery_score
from stepwise.config import ModelSpec
from stepwise.datasets import Sample
from stepwise.llm import DiskCache, LLMClient, LLMResponse
from stepwise.pipeline import ELICIT_SYSTEM

SPEC = ModelSpec(name="m", provider="groq", api_id="test/m")


@pytest.fixture
def numeric_sample():
    return Sample(
        question_id="gsm8k-0001", dataset="gsm8k", domain="arithmetic",
        question="Natalia sold 48 clips in April and half as many in May. How many total?",
        gold_answer="72", answer_type="numeric",
    )


@pytest.fixture
def mcq_sample():
    return Sample(
        question_id="csqa-0001", dataset="commonsenseqa", domain="commonsense",
        question="Where would you find a pizzeria while shopping?",
        gold_answer="B", answer_type="label",
        choices=["A. chicago", "B. mall", "C. street", "D. italy", "E. capital"],
    )


BASELINE = (
    "Step 1: Natalia sold 48 clips in April.\n"
    "Step 2: In May she sold half as many, so 48 / 2 = 24.\n"
    "Step 3: Altogether 48 + 24 = 72.\n"
    "Answer: 72"
)


def responder(**roles):
    """Route by system prompt to a scripted reply; default 'Answer: 72'."""
    def respond(messages):
        system = messages[0]["content"]
        for marker, reply in roles.items():
            if marker in system:
                return reply(messages) if callable(reply) else reply
        return "Answer: 72"
    return respond


def make_client(respond, *, use_cache=False):
    from tests.conftest import FakeProvider

    provider = FakeProvider(respond)
    client = LLMClient(cache=DiskCache(enabled=use_cache),
                       providers={"groq": provider, "openai": provider, "anthropic": provider})
    client.fake = provider
    return client


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def test_render_first_k_truncates_with_markers():
    steps = [
        {"index": 1, "text": "First.", "marker": "Step 1:"},
        {"index": 2, "text": "Second.", "marker": "Step 2:"},
        {"index": 3, "text": "Third.", "marker": "Step 3:"},
    ]
    assert battery._render_first_k(steps, 0) == ""
    assert battery._render_first_k(steps, 2) == "Step 1: First.\nStep 2: Second."


def test_fillerise_keeps_structure_drops_content():
    steps = [{"index": 1, "text": "forty-eight clips", "marker": "Step 1:"},
             {"index": 2, "text": "half of that", "marker": "Step 2:"}]
    out = battery._fillerise(steps)
    assert out.count("\n") == 1
    assert "Step 1:" in out and "Step 2:" in out
    assert "forty-eight" not in out
    assert "." in out


def test_wrong_hint_picks_a_distractor():
    boolean = Sample("q", "strategyqa", "", "yes", answer_type="boolean")
    assert battery._wrong_hint(boolean, "yes") == "no"
    assert battery._wrong_hint(boolean, "no") == "yes"
    mcq = Sample("q", "csqa", "", "B", answer_type="label",
                 choices=["A. x", "B. y", "C. z"])
    hint = battery._wrong_hint(mcq, "B")
    assert hint in ("A", "C") and hint != "B"
    # Numeric has no clean distractor → skip, don't invent one.
    numeric = Sample("q", "gsm8k", "", "72", answer_type="numeric")
    assert battery._wrong_hint(numeric, "72") is None


def test_mentions_hint():
    assert battery._mentions_hint("I think the answer is A here", "A")
    assert battery._mentions_hint("the mall option", "B. mall")
    assert not battery._mentions_hint("completely unrelated text", "A")


# --------------------------------------------------------------------------
# Early answering
# --------------------------------------------------------------------------


def test_early_answering_asks_at_each_cut(numeric_sample):
    client = make_client(responder(**{EARLY_SYS_MARK: "Answer: 72"}))
    trace = _trace(client, numeric_sample)
    recs = battery.early_answering(numeric_sample, trace, SPEC, client,
                                   fractions=(0.0, 0.5, 1.0))
    ks = sorted(int(r.variant) for r in recs)
    assert ks[0] == 0  # zero-step point is always included
    assert all(r.test_type == "early_answering" for r in recs)
    assert all(r.result_answer == "72" for r in recs)


def test_early_answering_detects_already_knew(numeric_sample):
    # Model gives the final answer even with zero reasoning → already knew.
    client = make_client(responder(**{EARLY_SYS_MARK: "Answer: 72"}))
    trace = _trace(client, numeric_sample)
    recs = battery.early_answering(numeric_sample, trace, SPEC, client, fractions=(0.0,))
    zero = [r for r in recs if r.variant == "0"][0]
    assert zero.matches_baseline is True


def test_early_answering_detects_reasoning_needed(numeric_sample):
    # Model cannot answer from nothing, only later → reasoning was needed.
    def respond(messages):
        if EARLY_SYS_MARK in messages[0]["content"]:
            return "Answer: 99" if "(nothing yet)" in messages[-1]["content"] else "Answer: 72"
        return BASELINE
    client = make_client(respond)
    trace = _trace(client, numeric_sample)
    recs = battery.early_answering(numeric_sample, trace, SPEC, client, fractions=(0.0, 1.0))
    zero = [r for r in recs if r.variant == "0"][0]
    assert zero.matches_baseline is False


# --------------------------------------------------------------------------
# Paraphrase & filler
# --------------------------------------------------------------------------


def test_paraphrase_reruns_on_the_reworded_cot(numeric_sample):
    client = make_client(responder(**{
        battery.PARAPHRASE_SYSTEM: "Step 1: April had 48.\nStep 2: May 24.\nStep 3: Sum 72.",
    }))
    trace = _trace(client, numeric_sample)
    rec = battery.paraphrase(numeric_sample, trace, SPEC, client, paraphrase_model="m")
    assert rec.test_type == "paraphrase"
    assert rec.result_answer == "72"
    assert rec.matches_baseline is True
    assert rec.detail["changed"] is True


def test_filler_replaces_reasoning(numeric_sample):
    client = make_client(responder())
    trace = _trace(client, numeric_sample)
    rec = battery.filler(numeric_sample, trace, SPEC, client)
    assert rec.test_type == "filler"
    assert "..." in rec.detail["filler_cot"] or "." in rec.detail["filler_cot"]
    assert rec.result_answer == "72"


# --------------------------------------------------------------------------
# Biased context
# --------------------------------------------------------------------------


def test_biased_context_flips_and_stays_silent(mcq_sample):
    # Model flips to the planted hint but never mentions it → rationalisation.
    def respond(messages):
        system, user = messages[0]["content"], messages[-1]["content"]
        if ELICIT_SYSTEM in system and "colleague" in user:
            return "Step 1: The mall has food.\nStep 2: Pizzerias are food.\nAnswer: A"
        return "Step 1: Reason.\nAnswer: B"
    client = make_client(respond)
    rec = battery.biased_context(mcq_sample, SPEC, client,
                                 baseline_answer="B", baseline_correct=True)
    assert rec.test_type == "biased_context"
    assert rec.detail["flipped_to_hint"] is True
    assert rec.detail["cot_mentions_hint"] is False  # silent rationalisation


def test_biased_context_skipped_without_a_distractor(numeric_sample):
    client = make_client(responder())
    rec = battery.biased_context(numeric_sample, SPEC, client,
                                 baseline_answer="72", baseline_correct=True)
    assert rec.error == "no_distractor_available"


# --------------------------------------------------------------------------
# Run loop & scoring
# --------------------------------------------------------------------------


def test_run_battery_for_sample_runs_requested_tests(numeric_sample):
    client = make_client(responder(**{battery.PARAPHRASE_SYSTEM: "reworded"}))
    recs = battery.run_battery_for_sample(
        numeric_sample, SPEC, client, tests=("early_answering", "paraphrase", "filler"),
        paraphrase_model="m",
    )
    kinds = {r.test_type for r in recs}
    assert kinds == {"early_answering", "paraphrase", "filler"}


def test_battery_summary_computes_each_metric():
    recs = [
        _rec("early_answering", "0", matches=True, detail={"fraction": 0.0}),
        _rec("early_answering", "0", matches=False, detail={"fraction": 0.0}),
        _rec("paraphrase", "", matches=True),
        _rec("filler", "", matches=True, correct=True),
        _rec("biased_context", "A", detail={"flipped_to_hint": True, "cot_mentions_hint": False}),
        _rec("biased_context", "A", detail={"flipped_to_hint": False, "cot_mentions_hint": False}),
    ]
    s = battery_score.battery_summary(recs)
    row = s["per_model"][0]
    assert row["early_already_knew"] == 0.5          # 1 of 2 matched at k=0
    assert row["paraphrase_robustness"] == 1.0
    assert row["filler_accuracy"] == 1.0
    assert row["bias_flip_rate"] == 0.5              # 1 of 2 flipped
    assert row["bias_silent_rate"] == 1.0            # the flip was silent


def test_early_curve_has_a_point_per_fraction():
    recs = [_rec("early_answering", str(k), matches=(k > 0), detail={"fraction": f})
            for k, f in [(0, 0.0), (2, 0.5), (4, 1.0)] for _ in range(5)]
    curve = battery_score.early_answering_curve(recs)
    assert sorted(curve["fraction"].tolist()) == [0.0, 0.5, 1.0]
    assert (curve["ci_low"] <= curve["match_rate"]).all()


def test_unparseable_results_are_not_scored():
    recs = [_rec("paraphrase", "", matches=False, answer="")]
    s = battery_score.battery_summary(recs)
    assert "paraphrase_robustness" not in s["per_model"][0]


# --------------------------------------------------------------------------
# Fixtures / builders
# --------------------------------------------------------------------------

EARLY_SYS_MARK = "commit to your best final answer"


def _trace(client, sample):
    from stepwise.pipeline import elicit_trace

    return elicit_trace(sample, SPEC, client)


def _rec(test, variant, *, matches=False, correct=False, answer="X", detail=None):
    return battery.BatteryRecord(
        question_id="q", dataset="gsm8k", domain="arithmetic", model_name="m",
        test_type=test, variant=variant, baseline_answer="72",
        result_answer=answer, baseline_correct=True, result_correct=correct,
        matches_baseline=matches, answer_type="numeric", detail=detail or {},
    )
