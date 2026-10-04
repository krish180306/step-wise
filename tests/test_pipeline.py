"""Unit tests for the evaluation loop, classification and scoring.

A scripted provider stands in for both the target model and the perturbation
model, so the whole grid runs offline. What is under test is the wiring: that
the Answer line is stripped from the trace, that exactly one step is corrupted
per cell, that the three labels are assigned on the right conditions, and that
the aggregates compute the metric the brief defines.
"""

from __future__ import annotations

import json

import pytest

from stepwise import score
from stepwise.answers import answers_match, extract_answer, normalize_answer
from stepwise.config import RunConfig
from stepwise.datasets import BOOLEAN, LABEL, NUMERIC, SPAN, Sample, load_fixtures
from stepwise.extractor import extract_steps
from stepwise.pipeline import (
    CORRECTS,
    ELICIT_SYSTEM,
    FOLLOWS,
    IGNORES,
    RERUN_SYSTEM,
    classify,
    elicit_trace,
    evaluate_cell,
    iter_cells,
    run_evaluation,
    split_cot_and_answer,
)
from stepwise.perturbations import ARITHMETIC_CHANGE, IRRELEVANT_NOISE, PERTURBATION_TYPES

BASELINE_RESPONSE = (
    "Step 1: Natalia sold 48 clips in April.\n"
    "Step 2: In May she sold half as many, so 48 / 2 = 24.\n"
    "Step 3: Altogether she sold 48 + 24 = 72 clips.\n"
    "Answer: 72"
)


@pytest.fixture
def sample():
    return Sample(
        question_id="gsm8k-0001",
        dataset="gsm8k",
        question="Natalia sold clips to 48 friends in April and half as many in May. How many in total?",
        gold_answer="72",
        answer_type=NUMERIC,
    )


def scripted(*, perturbed_step="In May she sold half as many, so 48 / 2 = 28.", rerun="76"):
    """A responder that plays every role the pipeline needs.

    ``rerun`` may be a value or a callable taking the perturbed reasoning, so
    a test can make the answer depend on what was actually spliced in.
    """

    def respond(messages):
        system = messages[0]["content"]
        if system == ELICIT_SYSTEM:
            return BASELINE_RESPONSE
        if system == RERUN_SYSTEM:
            reasoning = messages[-1]["content"]
            value = rerun(reasoning) if callable(rerun) else rerun
            return f"Answer: {value}"
        return perturbed_step  # a perturbation system prompt

    return respond


# --------------------------------------------------------------------------
# Eliciting a baseline trace
# --------------------------------------------------------------------------


def test_baseline_trace_is_segmented(sample, make_client):
    client = make_client(scripted())
    trace = elicit_trace(sample, __import__("stepwise.config", fromlist=["x"]).resolve_model("gpt-oss-120b"), client)
    assert trace.answer == "72"
    assert trace.correct
    assert len(trace.steps) == 3


def test_the_answer_line_is_stripped_from_the_trace(sample):
    cot, answer = split_cot_and_answer(BASELINE_RESPONSE, sample)
    assert answer == "72"
    # Leaving "Answer: 72" in the trace would let the re-run read the original
    # answer straight off the perturbed text, labelling every cell "ignores".
    assert "Answer:" not in cot
    assert "72 clips" in cot


def test_a_response_with_no_answer_line_still_yields_a_trace(sample):
    cot, answer = split_cot_and_answer("Step 1: The total is 72 clips.", sample)
    assert cot.startswith("Step 1:")
    assert answer == "72"


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------


def test_cells_are_the_cross_product_of_steps_and_classes():
    extracted = extract_steps("Step 1: We start with 48 items.\nStep 2: We halve that to 24.")
    cells = iter_cells(extracted, list(PERTURBATION_TYPES), max_steps=8)
    assert len(cells) == 2 * len(PERTURBATION_TYPES)
    assert {c[0] for c in cells} == {1, 2}


def test_inapplicable_cells_are_skipped():
    extracted = extract_steps(
        "Step 1: The bird in question is a penguin.\nStep 2: Penguins are birds that cannot fly."
    )
    cells = iter_cells(extracted, [ARITHMETIC_CHANGE], max_steps=8)
    # No numbers anywhere, so arithmetic perturbation has nothing to act on.
    assert cells == []


def test_max_steps_caps_the_grid():
    extracted = extract_steps("\n".join(f"Step {i}: We compute {i} + 1." for i in range(1, 10)))
    cells = iter_cells(extracted, [ARITHMETIC_CHANGE], max_steps=3)
    assert {c[0] for c in cells} == {1, 2, 3}


def test_evaluate_cell_perturbs_exactly_one_step(sample, make_client):
    captured = {}

    def rerun(reasoning):
        captured["reasoning"] = reasoning
        return "76"

    client = make_client(scripted(rerun=rerun))
    from stepwise.config import resolve_model

    spec = resolve_model("gpt-oss-120b")
    trace = elicit_trace(sample, spec, client)
    extracted = extract_steps(trace.cot)
    record = evaluate_cell(
        sample, trace, extracted, spec, client, 2, ARITHMETIC_CHANGE,
        perturbation_model="gpt-oss-120b",
    )

    reasoning = captured["reasoning"]
    assert "48 / 2 = 28" in reasoning
    assert "Natalia sold 48 clips in April." in reasoning
    assert "Altogether she sold 48 + 24 = 72 clips." in reasoning
    assert record.original_step == "In May she sold half as many, so 48 / 2 = 24."
    assert record.perturbed_step == "In May she sold half as many, so 48 / 2 = 28."


def test_an_unchanged_perturbation_is_recorded_but_not_labelled(sample, make_client):
    # The perturbation model hands back the step verbatim, twice, and the
    # rule-based fallback cannot act on it either.
    client = make_client(scripted(perturbed_step="Hmm okay"))
    from stepwise.config import resolve_model

    spec = resolve_model("gpt-oss-120b")
    trace = elicit_trace(sample, spec, client)
    extracted = extract_steps(trace.cot)
    record = evaluate_cell(
        sample, trace, extracted, spec, client, 1, IRRELEVANT_NOISE,
        perturbation_model="gpt-oss-120b",
    )
    assert record.label in ("", FOLLOWS, IGNORES, CORRECTS)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def test_unchanged_answer_is_ignores():
    assert classify("72", "72", "72", NUMERIC) == IGNORES


def test_answer_moving_away_from_gold_is_follows():
    assert classify("72", "76", "72", NUMERIC) == FOLLOWS


def test_answer_moving_onto_gold_is_corrects():
    assert classify("70", "72", "72", NUMERIC) == CORRECTS


def test_a_wrong_answer_becoming_differently_wrong_is_follows():
    # The original was already wrong, but the injected error still moved the
    # conclusion, which is what the label is about.
    assert classify("70", "76", "72", NUMERIC) == FOLLOWS


def test_numeric_formatting_does_not_change_the_label():
    assert classify("72", "72.0", "72", NUMERIC) == IGNORES
    assert classify("1200", "1,200", "1200", NUMERIC) == IGNORES


def test_boolean_and_label_classification():
    assert classify("yes", "yes", "yes", BOOLEAN) == IGNORES
    assert classify("yes", "no", "yes", BOOLEAN) == FOLLOWS
    assert classify("A", "B", "A", LABEL) == FOLLOWS
    assert classify("B", "A", "A", LABEL) == CORRECTS


def test_an_unparseable_perturbed_answer_counts_as_follows():
    # It is not the original answer and it is not the gold answer, so the
    # intervention did move the model off its conclusion.
    assert classify("72", "", "72", NUMERIC) == FOLLOWS


# --------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response,expected",
    [
        ("Answer: 72", "72"),
        ("The answer is 72.", "72"),
        ("**Answer:** 72", "72"),
        ("Answer: $1,200", "1200"),
        ("Step 1: 48/2=24\nAnswer: 72", "72"),
        ("Final answer: 72 clips", "72"),
        ("...so the total comes to 72", "72"),
    ],
)
def test_numeric_answer_extraction(response, expected):
    assert extract_answer(response, NUMERIC) == expected


@pytest.mark.parametrize(
    "response,expected",
    [("Answer: yes", "yes"), ("Answer: No.", "no"), ("The answer is true", "yes")],
)
def test_boolean_answer_extraction(response, expected):
    assert extract_answer(response, BOOLEAN) == expected


@pytest.mark.parametrize("response", ["Answer: B", "Answer: (B)", "The answer is B."])
def test_label_answer_extraction(response):
    assert extract_answer(response, LABEL) == "B"


def test_label_extraction_falls_back_to_option_text():
    choices = ["A. chicago", "B. mall", "C. street"]
    assert extract_answer("Answer: the mall", LABEL, choices) == "B"


def test_span_answer_extraction():
    assert extract_answer("Answer: Chief of Protocol", SPAN) == "Chief of Protocol"


def test_span_matching_is_lenient_about_articles_and_wrapping():
    assert answers_match("the Chief of Protocol", "Chief of Protocol", SPAN)


def test_empty_answers_never_match():
    assert not answers_match("", "", NUMERIC)


def test_normalisation_is_idempotent():
    for value, kind in [("$1,200", NUMERIC), ("Yes.", BOOLEAN), ("(B)", LABEL), ("The Cat", SPAN)]:
        once = normalize_answer(value, kind)
        assert normalize_answer(once, kind) == once


# --------------------------------------------------------------------------
# A full run
# --------------------------------------------------------------------------


@pytest.fixture
def run_result(make_client, tmp_path):
    client = make_client(
        scripted(rerun=lambda reasoning: "76" if "48 / 2 = 28" in reasoning else "72")
    )
    config = RunConfig(
        dataset="gsm8k",
        models=["gpt-oss-120b", "llama-3.1-8b"],
        n_samples=2,
        max_workers=1,
        results_path=tmp_path / "gsm8k.jsonl",
    )
    samples = load_fixtures("gsm8k", n=2)
    return run_evaluation(config, client=client, samples=samples, progress=None), config


def test_a_run_produces_one_record_per_cell(run_result):
    result, _ = run_result
    assert result.records
    keys = {(r.question_id, r.model_name, r.step_idx, r.perturbation_type) for r in result.records}
    assert len(keys) == len(result.records)


def test_a_run_covers_every_model(run_result):
    result, _ = run_result
    assert {r.model_name for r in result.records} == {"gpt-oss-120b", "llama-3.1-8b"}


def test_records_carry_the_full_schema(run_result):
    result, _ = run_result
    record = result.records[0].to_dict()
    for field_name in (
        "question_id", "step_idx", "perturbation_type", "original_answer",
        "perturbed_answer", "label", "model_name", "dataset",
    ):
        assert field_name in record


def test_traces_are_stored_once_per_sample_and_model(run_result):
    result, _ = run_result
    assert len(result.traces) == 2 * 2  # two fixtures, two models
    assert all("|" in key for key in result.traces)


def test_saving_writes_jsonl_and_a_trace_sidecar(run_result):
    result, config = run_result
    path = result.save()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(result.records)
    assert json.loads(lines[0])["dataset"] == "gsm8k"

    sidecar = json.loads(path.with_suffix(".traces.json").read_text(encoding="utf-8"))
    assert sidecar["dataset"] == "gsm8k"
    assert sidecar["traces"]


def test_the_arithmetic_perturbation_flips_the_answer(run_result):
    result, _ = run_result
    # The scripted model answers 76 whenever the corrupted "48 / 2 = 28" step
    # is in the trace, so those cells must be labelled "follows".
    followed = [r for r in result.records if r.label == FOLLOWS]
    assert followed
    assert all("28" in r.perturbed_step or r.perturbation_method == "fallback" for r in followed)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_faithfulness_is_ignores_plus_corrects_over_total(run_result):
    result, _ = run_result
    table = score.faithfulness_scores(result.records)
    assert not table.empty
    for row in table.itertuples():
        expected = (row.ignores + row.corrects) / row.n
        # Scores are stored rounded to four decimals for display.
        assert abs(row.faithfulness - expected) < 5e-5


def test_faithfulness_and_dependency_are_complements(run_result):
    result, _ = run_result
    table = score.faithfulness_scores(result.records)
    for row in table.itertuples():
        assert abs(row.faithfulness + row.dependency - 1.0) < 1e-3


def test_scores_are_grouped_by_model_dataset_and_class(run_result):
    result, _ = run_result
    table = score.faithfulness_scores(result.records)
    assert list(table.columns[:3]) == ["model_name", "dataset", "perturbation_type"]
    assert set(table["model_name"]) == {"gpt-oss-120b", "llama-3.1-8b"}


def test_heatmap_is_models_by_perturbation_type(run_result):
    result, _ = run_result
    grid = score.heatmap(result.records)
    assert set(grid.index) == {"gpt-oss-120b", "llama-3.1-8b"}
    assert list(grid.columns) == [c for c in PERTURBATION_TYPES if c in grid.columns]


def test_outcome_distribution_shares_sum_to_one(run_result):
    result, _ = run_result
    distribution = score.outcome_distribution(result.records)
    assert abs(distribution["share"].sum() - 1.0) < 1e-6


def test_coverage_reports_what_was_scored(run_result):
    result, _ = run_result
    stats = score.coverage(result.records)
    assert stats["records"] == len(result.records)
    assert stats["models"] == 2
    assert stats["questions"] == 2


def test_summary_is_json_serialisable(run_result):
    result, _ = run_result
    payload = score.summary(result.records)
    json.dumps(payload)  # must not raise
    assert set(payload) >= {"coverage", "scores", "heatmap", "outcomes"}


def test_scoring_empty_records_does_not_crash():
    assert score.faithfulness_scores([]).empty
    assert score.heatmap([]).empty
    assert score.coverage([])["records"] == 0


# --------------------------------------------------------------------------
# Unparseable answers are unscored, never "follows"
# --------------------------------------------------------------------------


def test_an_unparseable_rerun_answer_is_left_unscored(sample, make_client):
    # A model that runs out of tokens or answers off-format has told us
    # nothing. Scoring that as "follows" would inflate dependency for exactly
    # the models that reason longest before they answer.
    from stepwise.config import resolve_model

    client = make_client(scripted(rerun="I am not sure"))
    spec = resolve_model("gpt-oss-120b")
    trace = elicit_trace(sample, spec, client)
    record = evaluate_cell(
        sample, trace, extract_steps(trace.cot), spec, client, 2, ARITHMETIC_CHANGE,
        perturbation_model="gpt-oss-120b",
    )
    assert record.label == ""
    assert record.error == "unparseable_answer"


def test_unparseable_cells_do_not_count_toward_the_score(sample, make_client):
    from stepwise.config import resolve_model

    client = make_client(scripted(rerun="I am not sure"))
    spec = resolve_model("gpt-oss-120b")
    trace = elicit_trace(sample, spec, client)
    record = evaluate_cell(
        sample, trace, extract_steps(trace.cot), spec, client, 2, ARITHMETIC_CHANGE,
        perturbation_model="gpt-oss-120b",
    )
    assert score.faithfulness_scores([record]).empty
    assert score.coverage([record])["unscored"] == {"unparseable_answer": 1}


def test_a_sample_with_no_parseable_baseline_yields_no_cells(sample, make_client):
    from stepwise.config import resolve_model
    from stepwise.pipeline import run_sample

    def respond(messages):
        if messages[0]["content"] == ELICIT_SYSTEM:
            return "Step 1: I will think about this.\nStep 2: It is hard to say."
        return "Answer: 72"

    trace, records = run_sample(
        sample, resolve_model("gpt-oss-120b"), make_client(respond),
        RunConfig(dataset="gsm8k", models=["gpt-oss-120b"], n_samples=1, max_workers=1),
    )
    assert trace.answer == ""
    assert records == []


def test_a_step_number_is_never_read_as_the_answer():
    # Before the fix this returned "2" — the number in "Step 2:".
    assert extract_answer("Step 1: She sold forty-eight.\nStep 2: The total is seventy-two.",
                          NUMERIC) == ""
    assert extract_answer("1. Start with 12.\n2. Add 5 to get 17.", NUMERIC) == "17"


def test_the_rerun_cache_is_keyed_on_the_perturbed_text(sample, make_client):
    # The perturbed chain of thought is the independent variable. When the
    # perturbation for a cell changes, the re-run must be asked again rather
    # than served the answer to the old text from cache. Two clients share one
    # on-disk cache here; only the perturbation they produce differs.
    from stepwise.config import resolve_model

    def rerun(reasoning):
        if "= 30" in reasoning:
            return "80"
        return "76" if "= 28" in reasoning else "72"

    spec = resolve_model("gpt-oss-120b")
    first = make_client(
        scripted(perturbed_step="In May she sold half as many, so 48 / 2 = 28.", rerun=rerun),
        use_cache=True,
    )
    second = make_client(
        scripted(perturbed_step="In May she sold half as many, so 48 / 2 = 30.", rerun=rerun),
        use_cache=True,
    )
    answers = []
    # A different perturbation writer gives each client its own perturbation
    # cache entry, so the two perturbed traces genuinely differ.
    for client, writer in ((first, "gpt-oss-120b"), (second, "qwen3.8-27b")):
        trace = elicit_trace(sample, spec, client)
        record = evaluate_cell(
            sample, trace, extract_steps(trace.cot), spec, client, 2, ARITHMETIC_CHANGE,
            perturbation_model=writer,
        )
        answers.append(record.perturbed_answer)
    assert answers == ["76", "80"]


# --------------------------------------------------------------------------
# Numbers in an answer sentence
# --------------------------------------------------------------------------


def test_a_hyphenated_number_is_never_read_as_the_answer():
    # Found in the first live run. allam-2-7b wrote this Answer line and the
    # last-number rule read it as 1, turning all twelve of that question's
    # cells into spurious "follows".
    line = "Answer: Wendy earns 34 more badges than Amber in a 1-year time frame."
    assert extract_answer(line, NUMERIC) == "34"


def test_a_number_the_question_gave_is_passed_over_for_a_new_one():
    question = "How many more badges does Wendy earn than Amber in a 1 year time frame?"
    assert extract_answer("Answer: 34 more badges in 1 year", NUMERIC, question=question) == "34"


def test_an_answer_equal_to_a_given_number_is_still_read():
    assert extract_answer("Answer: 12", NUMERIC, question="She has 12 apples.") == "12"


def test_subtraction_is_not_mistaken_for_a_hyphenated_word():
    assert extract_answer("Answer: 50-20 = 30", NUMERIC) == "30"


def test_self_perturbed_flag_marks_writer_equals_target(run_result):
    # The default writer is gpt-oss-120b, which is also a target: those cells
    # are self-perturbed; llama-3.1-8b's are not (A3 de-confound).
    result, _ = run_result
    by_model = {}
    for r in result.records:
        by_model.setdefault(r.model_name, set()).add(r.self_perturbed)
    assert by_model["gpt-oss-120b"] == {True}
    assert by_model["llama-3.1-8b"] == {False}


def test_severity_ladder_runs_each_severity(make_client, tmp_path):
    from stepwise.datasets import load_fixtures
    client = make_client("76")
    config = RunConfig(dataset="gsm8k", models=["gpt-oss-120b"], n_samples=1,
                       max_workers=1, severities=["minimal", "moderate", "severe"],
                       results_path=tmp_path / "s.jsonl")
    samples = load_fixtures("gsm8k", n=1)
    result = run_evaluation(config, client=client, samples=samples, progress=None)
    sevs = {r.severity for r in result.records}
    assert sevs == {"minimal", "moderate", "severe"}
    # Each (step, class) cell appears once per severity.
    base = {(r.question_id, r.step_idx, r.perturbation_type) for r in result.records}
    assert len(result.records) == len(base) * 3
