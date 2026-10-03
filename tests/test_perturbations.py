"""Unit tests for perturbation generation, cleaning and splicing.

The LLM is mocked throughout. What is under test is everything around the
call: prompt assembly, output cleaning, the validity gate that decides whether
a model's rewrite is usable, the deterministic fallbacks, and the fact that a
perturbed step lands in the trace without disturbing its neighbours.
"""

from __future__ import annotations

import pytest

from stepwise.extractor import extract_steps, splice_step
from stepwise.perturbations import (
    ARITHMETIC_CHANGE,
    FACT_SUBSTITUTION,
    IRRELEVANT_NOISE,
    LOGICAL_ERROR,
    PERTURBATION_TYPES,
    PERTURBATIONS,
    apply_fallback,
    clean_output,
    is_applicable,
    is_valid,
    perturb_step,
)

from .conftest import MARKED_COT

ARITHMETIC_STEP = "In May she sold half as many, so 48 / 2 = 24."
FACTUAL_STEP = "Paris is the capital of France and hosted the 1900 games."
LOGICAL_STEP = "Since every bird can fly, therefore a penguin can fly."


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_all_four_classes_are_registered():
    assert set(PERTURBATION_TYPES) == {
        FACT_SUBSTITUTION,
        LOGICAL_ERROR,
        ARITHMETIC_CHANGE,
        IRRELEVANT_NOISE,
    }


@pytest.mark.parametrize("ptype", PERTURBATION_TYPES)
def test_every_class_has_a_strict_system_prompt(ptype):
    prompt = PERTURBATIONS[ptype].system_prompt
    assert "Return ONLY the rewritten step" in prompt
    assert "no explanation" in prompt.lower()


def test_unknown_perturbation_type_is_rejected():
    with pytest.raises(ValueError, match="unknown perturbation type"):
        perturb_step("some step", "not_a_real_type")


# --------------------------------------------------------------------------
# The LLM path, mocked
# --------------------------------------------------------------------------


def test_llm_response_is_used_when_valid(make_client):
    client = make_client("In May she sold half as many, so 48 / 2 = 28.")
    result = perturb_step(ARITHMETIC_STEP, ARITHMETIC_CHANGE, client=client)
    assert result.method == "llm"
    assert result.text.endswith("48 / 2 = 28.")
    assert result.changed


def test_the_step_and_question_reach_the_prompt(make_client):
    client = make_client("something else entirely that is long enough to pass")
    perturb_step(
        ARITHMETIC_STEP,
        ARITHMETIC_CHANGE,
        client=client,
        question="How many clips did Natalia sell?",
        context="Step 1: Natalia sold 48 clips in April.",
    )
    sent = client.fake.last_user_message
    assert ARITHMETIC_STEP in sent
    assert "How many clips did Natalia sell?" in sent
    assert "Natalia sold 48 clips in April." in sent
    assert client.fake.last_system_message == PERTURBATIONS[ARITHMETIC_CHANGE].system_prompt


def test_class_specific_system_prompt_is_selected(make_client):
    for ptype in PERTURBATION_TYPES:
        client = make_client("a plausible rewritten step of roughly equal length here")
        perturb_step(FACTUAL_STEP, ptype, client=client)
        assert client.fake.last_system_message == PERTURBATIONS[ptype].system_prompt


def test_an_unchanged_response_triggers_a_retry(make_client):
    client = make_client([ARITHMETIC_STEP, "In May she sold half as many, so 48 / 2 = 31."])
    result = perturb_step(ARITHMETIC_STEP, ARITHMETIC_CHANGE, client=client)
    assert len(client.fake.calls) == 2
    assert result.method == "llm"
    assert result.text.endswith("48 / 2 = 31.")


def test_the_retry_states_what_was_wrong(make_client):
    client = make_client([ARITHMETIC_STEP, "In May she sold half as many, so 48 / 2 = 31."])
    perturb_step(ARITHMETIC_STEP, ARITHMETIC_CHANGE, client=client)
    retry_messages = client.fake.calls[1]["messages"]
    assert retry_messages[-1]["role"] == "user"
    assert "not usable" in retry_messages[-1]["content"]
    # The failed attempt is kept in the transcript so the model can see it.
    assert retry_messages[-2]["role"] == "assistant"


def test_two_bad_responses_fall_back_to_rules(make_client):
    client = make_client("I cannot help with that.")
    result = perturb_step(ARITHMETIC_STEP, ARITHMETIC_CHANGE, client=client)
    assert len(client.fake.calls) == 2
    assert result.method == "fallback"
    assert result.changed


def test_fallback_can_be_refused(make_client):
    client = make_client("I cannot help with that.")
    result = perturb_step(
        ARITHMETIC_STEP, ARITHMETIC_CHANGE, client=client, allow_fallback=False
    )
    assert result.method == "unchanged"


def test_no_client_means_pure_rule_based_perturbation():
    result = perturb_step(ARITHMETIC_STEP, ARITHMETIC_CHANGE)
    assert result.method == "fallback"


def test_temperature_rises_on_the_retry(make_client):
    client = make_client(["", "a genuinely different step of a similar length to before"])
    perturb_step(ARITHMETIC_STEP, ARITHMETIC_CHANGE, client=client)
    assert client.fake.calls[0]["temperature"] == 0.0
    assert client.fake.calls[1]["temperature"] > 0.0


def test_identical_cache_keys_avoid_a_second_call(make_client):
    client = make_client("In May she sold half as many, so 48 / 2 = 28.", use_cache=True)
    first = perturb_step(ARITHMETIC_STEP, ARITHMETIC_CHANGE, client=client, cache_key="q1|s2|arith")
    second = perturb_step(ARITHMETIC_STEP, ARITHMETIC_CHANGE, client=client, cache_key="q1|s2|arith")
    assert first.text == second.text
    assert len(client.fake.calls) == 1
    assert second.cached


# --------------------------------------------------------------------------
# Output cleaning
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "Here is the rewritten step: The total is 28.",
        "Here's the perturbed step:\nThe total is 28.",
        "Rewritten step: The total is 28.",
        '"The total is 28."',
        "```\nThe total is 28.\n```",
        "```text\nThe total is 28.\n```",
        "   The total is 28.   ",
    ],
)
def test_wrappers_are_stripped(raw):
    assert clean_output(raw, "The total is 24.") == "The total is 28."


def test_a_step_label_is_stripped_when_the_source_had_none():
    assert clean_output("Step 2: The total is 28.", "The total is 24.") == "The total is 28."


def test_a_step_label_is_kept_when_the_source_had_one():
    cleaned = clean_output("Step 2: The total is 28.", "Step 2: The total is 24.")
    assert cleaned == "Step 2: The total is 28."


def test_internal_quotes_survive_cleaning():
    raw = 'He said "yes" and left the room.'
    assert clean_output(raw, "He said nothing and left.") == raw


def test_trailing_rationale_is_dropped():
    raw = "The total is 28.\nNote: I changed 24 to 28 to introduce the error."
    assert clean_output(raw, "The total is 24.") == "The total is 28."


def test_cleaning_handles_empty_input():
    assert clean_output("", "original") == ""
    assert clean_output(None, "original") == ""


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_unchanged_output_is_invalid():
    assert not is_valid(ARITHMETIC_STEP, ARITHMETIC_STEP, ARITHMETIC_CHANGE)


def test_empty_output_is_invalid():
    assert not is_valid("", ARITHMETIC_STEP, ARITHMETIC_CHANGE)


def test_refusal_is_invalid():
    assert not is_valid("I cannot help with that request.", ARITHMETIC_STEP, ARITHMETIC_CHANGE)


def test_a_truncated_output_is_invalid():
    assert not is_valid("In May", ARITHMETIC_STEP, ARITHMETIC_CHANGE)


def test_a_bloated_output_is_invalid():
    assert not is_valid(ARITHMETIC_STEP + " and then " * 30, ARITHMETIC_STEP, ARITHMETIC_CHANGE)


def test_noise_must_preserve_the_original_wording():
    additive = ARITHMETIC_STEP + " Clips were a popular accessory that year."
    assert is_valid(additive, ARITHMETIC_STEP, IRRELEVANT_NOISE)


def test_noise_that_rewrites_the_step_is_invalid():
    rewritten = "She sold 24 items in May. Clips were popular that year."
    assert not is_valid(rewritten, ARITHMETIC_STEP, IRRELEVANT_NOISE)


def test_noise_may_grow_the_step_more_than_other_classes():
    # One long inserted sentence: fine as noise, far too much drift for a
    # class that is supposed to touch a single value.
    long_addition = (
        ARITHMETIC_STEP
        + " Clips of this kind were a popular accessory throughout that decade."
    )
    assert is_valid(long_addition, ARITHMETIC_STEP, IRRELEVANT_NOISE)
    assert not is_valid(long_addition, ARITHMETIC_STEP, ARITHMETIC_CHANGE)


def test_noise_that_triples_the_step_is_still_rejected():
    # The control condition is one sentence, not a paragraph.
    assert not is_valid(
        ARITHMETIC_STEP + " This is background colour about clips." * 4,
        ARITHMETIC_STEP,
        IRRELEVANT_NOISE,
    )


def test_a_logical_error_that_changes_a_number_is_rejected():
    # Observed live on gpt-oss-120b: asked to break the logic of a purely
    # arithmetic step, it changed the result instead. That is an arithmetic
    # edit, and scoring it as a logical error would contaminate the class.
    assert not is_valid(
        "In May she sold half as many, so 48 / 2 = 48.", ARITHMETIC_STEP, LOGICAL_ERROR
    )


def test_a_logical_error_that_keeps_every_number_is_accepted():
    assert is_valid(
        "In May she sold twice as many, so 48 / 2 = 24.", ARITHMETIC_STEP, LOGICAL_ERROR
    )


# --------------------------------------------------------------------------
# Deterministic fallbacks
# --------------------------------------------------------------------------


def test_arithmetic_fallback_changes_the_result_not_the_operands():
    result = apply_fallback(ARITHMETIC_STEP, ARITHMETIC_CHANGE)
    assert result.method == "fallback"
    assert "48 / 2 =" in result.text
    assert not result.text.endswith("= 24.")


def test_fact_fallback_swaps_a_known_entity():
    result = apply_fallback(FACTUAL_STEP, FACT_SUBSTITUTION)
    assert "France" not in result.text
    assert "Paris" in result.text


def test_logic_fallback_inverts_a_connective():
    result = apply_fallback(LOGICAL_STEP, LOGICAL_ERROR)
    assert result.changed
    assert "therefore" not in result.text.lower()


def test_noise_fallback_only_appends():
    result = apply_fallback(ARITHMETIC_STEP, IRRELEVANT_NOISE)
    assert result.text.startswith(ARITHMETIC_STEP)
    assert len(result.text) > len(ARITHMETIC_STEP)


def test_fallbacks_are_deterministic():
    for ptype in PERTURBATION_TYPES:
        first = apply_fallback(ARITHMETIC_STEP, ptype)
        second = apply_fallback(ARITHMETIC_STEP, ptype)
        assert first.text == second.text


def test_a_step_that_resists_every_rule_is_marked_unchanged():
    result = apply_fallback("Hmm okay", LOGICAL_ERROR)
    assert result.method == "unchanged"
    assert not result.changed


@pytest.mark.parametrize("ptype", PERTURBATION_TYPES)
def test_fallback_output_is_always_a_perturbation_record(ptype):
    result = apply_fallback(FACTUAL_STEP, ptype)
    assert result.type == ptype
    assert result.original == FACTUAL_STEP
    assert set(result.to_dict()) == {
        "type",
        "original_step",
        "perturbed_step",
        "method",
        "perturbation_model",
    }


# --------------------------------------------------------------------------
# Applicability
# --------------------------------------------------------------------------


def test_arithmetic_is_not_applicable_without_numbers():
    assert not is_applicable(ARITHMETIC_CHANGE, "The bird is blue and quite small.")
    assert is_applicable(ARITHMETIC_CHANGE, ARITHMETIC_STEP)


def test_noise_is_always_applicable():
    assert is_applicable(IRRELEVANT_NOISE, "The bird is blue.")


def test_logic_is_always_applicable():
    assert is_applicable(LOGICAL_ERROR, "The bird is blue.")


# --------------------------------------------------------------------------
# Splicing a perturbation back into a trace
# --------------------------------------------------------------------------


def test_perturbed_step_splices_without_disturbing_neighbours(make_client):
    client = make_client("In May she sold half as many, so 48 / 2 = 28.")
    extracted = extract_steps(MARKED_COT)
    perturbation = perturb_step(extracted.step(2).text, ARITHMETIC_CHANGE, client=client)
    perturbed_cot = splice_step(extracted, 2, perturbation.text)

    assert perturbed_cot != MARKED_COT
    original_lines = MARKED_COT.splitlines()
    perturbed_lines = perturbed_cot.splitlines()
    assert len(original_lines) == len(perturbed_lines)
    differing = [i for i, (a, b) in enumerate(zip(original_lines, perturbed_lines)) if a != b]
    assert differing == [1]


@pytest.mark.parametrize("ptype", PERTURBATION_TYPES)
def test_every_class_splices_cleanly_into_every_step(ptype):
    extracted = extract_steps(MARKED_COT)
    for step in extracted:
        perturbation = apply_fallback(step.text, ptype)
        spliced = splice_step(extracted, step.index, perturbation.text)
        # The other steps must survive verbatim.
        for other in extracted:
            if other.index != step.index:
                assert other.text in spliced
        assert spliced.count("Step 1:") == 1


def test_splicing_an_unchanged_perturbation_leaves_the_trace_intact():
    extracted = extract_steps(MARKED_COT)
    perturbation = apply_fallback("Hmm okay", LOGICAL_ERROR)
    assert not perturbation.changed
    assert splice_step(extracted, 1, extracted.step(1).text) == MARKED_COT
