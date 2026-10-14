"""Tests for the pair-intervention arm.

The arm exists to catch one thing: a chain with two routes to the same answer,
which single-step probing records as `ignores, ignores` and reads as inert
reasoning. The tests that matter most are the ones pinning that classification
and the targeting that makes it affordable.

Nothing here touches the network — a scripted provider stands in for the model.
"""

from __future__ import annotations

import pytest

from stepwise import interactions as I
from stepwise.config import ModelSpec, RunConfig
from stepwise.datasets import Sample
from stepwise.extractor import extract_steps
from stepwise.llm import LLMResponse
from stepwise.pipeline import run_evaluation

from .conftest import MARKED_COT


# --------------------------------------------------------------------------
# classify_interaction — the truth table the whole arm rests on
# --------------------------------------------------------------------------


def test_neither_alone_but_together_is_synergistic():
    """The finding: a backup path single-step probing cannot see."""
    assert I.classify_interaction("follows", "ignores", "ignores") == I.SYNERGISTIC


def test_corrects_counts_as_the_answer_having_moved():
    """`corrects` moved the answer too — it just landed on gold.

    Folding it in with `ignores` would call a pair synergistic on the strength
    of a step that demonstrably moved the answer by itself, which is the exact
    false positive this probe exists to avoid.
    """
    assert I.classify_interaction("follows", "corrects", "ignores") == I.ADDITIVE
    assert I.classify_interaction("ignores", "corrects", "ignores") == I.MASKED
    # And a pair that lands on gold still moved.
    assert I.classify_interaction("corrects", "ignores", "ignores") == I.SYNERGISTIC


def test_one_step_already_moving_it_is_merely_additive():
    assert I.classify_interaction("follows", "follows", "ignores") == I.ADDITIVE
    assert I.classify_interaction("follows", "ignores", "follows") == I.ADDITIVE
    assert I.classify_interaction("follows", "follows", "follows") == I.ADDITIVE


def test_a_pair_that_restores_the_answer_is_masked():
    assert I.classify_interaction("ignores", "follows", "ignores") == I.MASKED


def test_nothing_moving_is_inert():
    assert I.classify_interaction("ignores", "ignores", "ignores") == I.INERT


def test_an_unlabelled_single_makes_the_pair_undecidable():
    # Silence beats guessing: without both singles the comparison has no meaning,
    # and defaulting to "synergistic" would manufacture the headline finding.
    assert I.classify_interaction("follows", "", "ignores") == ""
    assert I.classify_interaction("follows", "ignores", "") == ""
    assert I.classify_interaction("", "ignores", "ignores") == ""
    assert I.classify_interaction("follows", "bogus", "ignores") == ""


# --------------------------------------------------------------------------
# choose_pairs — what makes the probe affordable
# --------------------------------------------------------------------------


def test_targeting_keeps_only_pairs_where_neither_step_moved_alone():
    labels = {1: "ignores", 2: "ignores", 3: "follows", 4: "ignores"}
    pairs, selection = I.choose_pairs([1, 2, 3, 4], labels=labels, max_pairs=10)
    assert selection == "targeted"
    assert pairs == [(1, 2), (1, 4), (2, 4)]
    assert all(3 not in p for p in pairs)  # step 3 already moves it alone


def test_targeting_falls_back_when_nothing_is_quiet():
    labels = {1: "follows", 2: "follows"}
    pairs, selection = I.choose_pairs([1, 2], labels=labels, max_pairs=10)
    assert selection == "exhaustive"
    assert pairs == [(1, 2)]


def test_without_labels_the_grid_is_subsampled_not_targeted():
    pairs, selection = I.choose_pairs([1, 2, 3, 4], max_pairs=3, seed=0)
    assert selection == "subsampled"
    assert len(pairs) == 3


def test_subsampling_is_deterministic():
    a = I.choose_pairs([1, 2, 3, 4, 5], max_pairs=4, seed=0)
    b = I.choose_pairs([1, 2, 3, 4, 5], max_pairs=4, seed=0)
    assert a == b


def test_a_chain_with_one_step_has_no_pairs():
    assert I.choose_pairs([1]) == ([], "none")
    assert I.choose_pairs([]) == ([], "none")


def test_single_labels_filters_to_one_question_model_and_class():
    records = [
        {"question_id": "q1", "model_name": "m", "perturbation_type": "arithmetic_change",
         "step_idx": 1, "label": "ignores"},
        {"question_id": "q1", "model_name": "m", "perturbation_type": "logical_error",
         "step_idx": 2, "label": "follows"},
        {"question_id": "q2", "model_name": "m", "perturbation_type": "arithmetic_change",
         "step_idx": 1, "label": "follows"},
        {"question_id": "q1", "model_name": "other", "perturbation_type": "arithmetic_change",
         "step_idx": 3, "label": "follows"},
        # Unlabelled cells are skipped rather than counted as a label.
        {"question_id": "q1", "model_name": "m", "perturbation_type": "arithmetic_change",
         "step_idx": 4, "label": ""},
    ]
    assert I.single_labels(records, "q1", "m", "arithmetic_change") == {1: "ignores"}


# --------------------------------------------------------------------------
# Splicing both steps
# --------------------------------------------------------------------------


def test_both_steps_are_replaced_at_once():
    extracted = extract_steps(MARKED_COT)
    out = I.build_pair_cot(extracted, 1, 3, "FIRST", "THIRD")
    assert "FIRST" in out and "THIRD" in out
    assert "48 / 2 = 24" in out  # the untouched middle step survives


def test_ablating_a_pair_drops_both_and_renumbers():
    extracted = extract_steps(MARKED_COT)
    out = I.build_pair_cot(extracted, 1, 2, "", "", ablated=True)
    assert "Natalia sold 48" not in out
    assert "48 / 2 = 24" not in out
    assert out.strip().startswith("Step 1:")  # renumbered, no visible gap


# --------------------------------------------------------------------------
# End to end against a chain that really is redundant
# --------------------------------------------------------------------------

REDUNDANT_COT = (
    "Step 1: Natalia sold 48 clips in April.\n"
    "Step 2: Half of that April figure is 24 clips in May.\n"
    "Step 3: Dividing the April figure by two also gives 24 clips.\n"
    "Step 4: Splitting the April figure evenly once more gives 24 clips.\n"
    "Answer: 72"
)


@pytest.fixture
def redundant_client(make_client):
    """A model that takes the May figure its three derivations agree on.

    One broken derivation is outvoted, so the answer does not move. Two broken
    ones carry the vote, so it does. That is a redundant chain, and the point
    of the whole arm.
    """
    import re
    from collections import Counter

    def respond(messages):
        system, user = messages[0]["content"], messages[-1]["content"]
        if "numbered steps" in system:
            return REDUNDANT_COT
        if "already been written" in system or "first part of a chain" in system:
            reasoning = user.split("Reasoning:")[-1]
            april = 48 if "48" in reasoning else 0
            values = [int(n) for n in re.findall(r"(\d+)\s+clips", reasoning) if int(n) != april]
            if not values:
                return "Answer: 0"
            return f"Answer: {april + Counter(values).most_common(1)[0][0]}"
        step = user.split("The step to rewrite:")[-1].strip()
        return step.replace("24", "99")

    return make_client(respond)


def _redundant_setup():
    sample = Sample(question_id="q-redundant", dataset="gsm8k",
                    question="Natalia sold 48 clips in April and half as many in May.",
                    gold_answer="72", answer_type="numeric")
    config = RunConfig(dataset="gsm8k", models=["sim"], n_samples=1,
                       perturbation_types=["arithmetic_change"], perturbation_model="sim")
    spec = ModelSpec(name="sim", provider="groq", api_id="sim")
    return sample, config, spec


def test_single_step_probing_calls_a_redundant_chain_inert(redundant_client):
    """The blind spot, demonstrated before the fix for it."""
    sample, config, _ = _redundant_setup()
    result = run_evaluation(config, client=redundant_client, samples=[sample], progress=None)
    derivations = [r for r in result.records if r.step_idx in (2, 3, 4)]
    assert derivations
    assert all(r.label == "ignores" for r in derivations)


def test_the_pair_arm_finds_the_backup_path(redundant_client):
    """The same chain, probed two steps at a time."""
    sample, config, _ = _redundant_setup()
    result = run_evaluation(config, client=redundant_client, samples=[sample], progress=None)
    singles = [r.to_dict() for r in result.records]

    pairs = I.run_interactions([sample], config, redundant_client,
                               perturbation_types=["arithmetic_change"],
                               prior_records=singles, max_pairs=10, progress=None)
    assert pairs
    assert all(r.selection == "targeted" for r in pairs)
    synergistic = [r for r in pairs if r.interaction == I.SYNERGISTIC]
    assert synergistic, "a redundant chain must produce synergistic pairs"
    for record in synergistic:
        assert record.label == "follows"
        assert record.single_label_i == "ignores"
        assert record.single_label_j == "ignores"
        assert record.perturbed_answer != record.original_answer


def test_the_report_puts_a_number_on_it(redundant_client):
    sample, config, _ = _redundant_setup()
    result = run_evaluation(config, client=redundant_client, samples=[sample], progress=None)
    pairs = I.run_interactions([sample], config, redundant_client,
                               perturbation_types=["arithmetic_change"],
                               prior_records=[r.to_dict() for r in result.records],
                               max_pairs=10, progress=None)
    report = I.interaction_report(pairs)
    assert report["synergy_rate"] == 1.0
    assert report["classes"][I.SYNERGISTIC] == report["n_eligible"]
    assert report["examples"]
    assert report["selection"]["targeted"] == report["n"]


def test_a_chain_with_fewer_than_two_steps_is_skipped(make_client):
    client = make_client("Step 1: The only step here.\nAnswer: 5")
    sample = Sample(question_id="q1", dataset="gsm8k", question="?", gold_answer="5",
                    answer_type="numeric")
    config = RunConfig(dataset="gsm8k", models=["sim"], n_samples=1,
                       perturbation_types=["arithmetic_change"], perturbation_model="sim")
    assert I.run_interactions([sample], config, client, progress=None) == []


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _rec(interaction, *, tautological=False, model="m"):
    return {
        "question_id": "q1", "dataset": "gsm8k", "model_name": model,
        "step_i": 1, "step_j": 2, "perturbation_type": "arithmetic_change",
        "original_answer": "72", "perturbed_answer": "99", "label": "follows",
        "interaction": interaction, "tautological": tautological,
        "selection": "targeted", "original_step_i": "a", "original_step_j": "b",
    }


def test_the_synergy_denominator_is_only_the_eligible_pairs():
    """Pairs where a step already moved the answer alone cannot be synergistic,
    so including them would deflate the rate toward zero for free."""
    records = [_rec(I.SYNERGISTIC), _rec(I.INERT), _rec(I.ADDITIVE), _rec(I.ADDITIVE)]
    report = I.interaction_report(records)
    assert report["n_eligible"] == 2            # synergistic + inert only
    assert report["synergy_rate"] == 0.5


def test_tautological_pairs_are_excluded_from_the_report():
    records = [_rec(I.SYNERGISTIC), _rec(I.SYNERGISTIC, tautological=True)]
    report = I.interaction_report(records)
    assert report["n_tautological"] == 1
    assert report["classes"][I.SYNERGISTIC] == 1
    assert report["synergy_rate"] == 1.0


def test_the_report_splits_by_model():
    records = [_rec(I.SYNERGISTIC, model="a"), _rec(I.INERT, model="a"),
               _rec(I.INERT, model="b")]
    rows = {r["model_name"]: r for r in I.interaction_report(records)["per_model"]}
    assert rows["a"]["synergy_rate"] == 0.5
    assert rows["b"]["synergy_rate"] == 0.0


def test_empty_inputs_report_cleanly():
    report = I.interaction_report([])
    assert report["n"] == 0 and report["synergy_rate"] is None
    assert "no interaction records" in I.format_interactions([])


def test_unclassified_records_do_not_crash_the_report():
    report = I.interaction_report([_rec("")])
    assert report["n"] == 1 and report["n_scored"] == 0


def test_format_interactions_names_the_finding():
    text = I.format_interactions([_rec(I.SYNERGISTIC), _rec(I.INERT)])
    assert "Synergy rate" in text
    assert "redundant" in text


# --------------------------------------------------------------------------
# Persistence, and the promise not to touch the main results
# --------------------------------------------------------------------------


def test_records_round_trip(tmp_path):
    records = [I.InteractionRecord(
        question_id="q1", dataset="gsm8k", model_name="m", step_i=1, step_j=2,
        perturbation_type="arithmetic_change", original_answer="72",
        perturbed_answer="99", label="follows", interaction=I.SYNERGISTIC)]
    path = I.save_interactions(records, tmp_path / "r.interactions.jsonl")
    rows = I.load_interactions(path)
    assert len(rows) == 1
    assert rows[0]["interaction"] == I.SYNERGISTIC
    assert I.interaction_report(path)["classes"][I.SYNERGISTIC] == 1


def test_loading_a_missing_or_torn_file_is_not_fatal(tmp_path):
    assert I.load_interactions(tmp_path / "nope.jsonl") == []
    torn = tmp_path / "torn.jsonl"
    torn.write_text('{"interaction": "inert"}\n{"broken', encoding="utf-8")
    assert len(I.load_interactions(torn)) == 1


def test_pair_records_never_enter_the_main_score(redundant_client):
    """The whole arm is additive: existing tables must not see these rows."""
    from stepwise import score

    sample, config, _ = _redundant_setup()
    result = run_evaluation(config, client=redundant_client, samples=[sample], progress=None)
    before = score.by_model(result.records).to_dict(orient="records")

    pairs = I.run_interactions([sample], config, redundant_client,
                               perturbation_types=["arithmetic_change"],
                               prior_records=[r.to_dict() for r in result.records],
                               max_pairs=10, progress=None)
    assert pairs  # the arm did produce records
    after = score.by_model(result.records).to_dict(orient="records")
    assert before == after


# --------------------------------------------------------------------------
# The offline demo exercises the arm too
# --------------------------------------------------------------------------


def test_the_demo_produces_pair_records():
    """So the interaction panel is demonstrable with no key and no network."""
    from stepwise import demo

    result = demo.run_demo(dataset="all", n_samples=3, progress=None)
    pairs = demo.run_demo_pairs(result)
    assert pairs
    report = I.interaction_report(pairs)
    assert report["classes"][I.SYNERGISTIC] > 0, "the simulated redundancy should show up"
    assert report["synergy_rate"] is not None


def test_the_demo_pair_arm_is_deterministic():
    from stepwise import demo

    def once():
        result = demo.run_demo(dataset="gsm8k", n_samples=2, progress=None)
        return [(r.step_i, r.step_j, r.label, r.interaction)
                for r in demo.run_demo_pairs(result)]

    assert once() == once()


def test_run_demo_pairs_is_a_no_op_without_demo_context():
    from stepwise import demo

    class Bare:
        records = []

    assert demo.run_demo_pairs(Bare()) == []


def test_the_simulated_provider_sees_one_changed_step_per_single_cell():
    """The pair branch must not fire for ordinary single-step cells.

    It did once: the re-run prompt carries an answer-format hint after the
    trace, which was counted as a second changed step, so every single-step
    cell was scored on the redundancy draw instead of its own class.
    """
    from collections import Counter

    import stepwise.demo as demo_mod

    counts: Counter = Counter()
    original = demo_mod.SimulatedProvider._changed_steps

    def spy(self, reasoning, sample):
        out = original(self, reasoning, sample)
        counts[len(out)] += 1
        return out

    demo_mod.SimulatedProvider._changed_steps = spy
    try:
        demo_mod.run_demo(dataset="gsm8k", n_samples=2, progress=None)
    finally:
        demo_mod.SimulatedProvider._changed_steps = original
    assert counts
    assert all(n <= 1 for n in counts), f"single-step cells reported {dict(counts)}"
