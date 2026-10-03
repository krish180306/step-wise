"""Unit tests for the analysis tools and the agent loop.

The agent is driven by a scripted provider that emits tool calls, so the loop,
the dispatch, the transcript shape and the stopping conditions are all
exercised without an API key.
"""

from __future__ import annotations

import json

import pytest

from stepwise import score
from stepwise.agent import analyst
from stepwise.agent.tools import (
    AnalysisContext,
    TOOLS_BY_NAME,
    call_tool,
    tool_schemas,
    two_proportion_test,
)
from stepwise.config import ModelSpec
from stepwise.llm import DiskCache, LLMClient, LLMResponse


def make_records():
    """A small run: two models, two classes, a deliberate asymmetry."""
    rows = []
    for model, follow_rate in (("model-a", 3), ("model-b", 1)):
        for q in range(4):
            for step in (1, 2):
                for ptype, follows in (
                    ("arithmetic_change", q < follow_rate),
                    ("irrelevant_noise", False),
                ):
                    rows.append(
                        {
                            "question_id": f"gsm8k-{q:04d}",
                            "dataset": "gsm8k",
                            "model_name": model,
                            "step_idx": step,
                            "n_steps": 2,
                            "perturbation_type": ptype,
                            "original_answer": "72",
                            "perturbed_answer": "76" if follows else "72",
                            "label": "follows" if follows else "ignores",
                            "gold_answer": "72",
                            "original_correct": True,
                            "perturbed_correct": not follows,
                            "original_step": "In May she sold half as many, so 48 / 2 = 24.",
                            "perturbed_step": "In May she sold half as many, so 48 / 2 = 28.",
                            "perturbation_method": "llm",
                            "answer_type": "numeric",
                            "error": "",
                        }
                    )
    return rows


@pytest.fixture
def ctx():
    return AnalysisContext(
        frame=score.records_dataframe(make_records()),
        traces={
            "gsm8k-0000|model-a": {
                "question": "How many clips did Natalia sell?",
                "gold_answer": "72",
                "cot": "Step 1: She sold 48 in April.\nStep 2: In May she sold half as many, so 48 / 2 = 24.",
                "answer": "72",
                "steps": [
                    {"index": 1, "text": "She sold 48 in April.", "marker": "Step 1:"},
                    {"index": 2, "text": "In May she sold half as many, so 48 / 2 = 24.",
                     "marker": "Step 2:"},
                ],
            }
        },
        dataset="gsm8k",
    )


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


def test_every_tool_has_a_valid_schema():
    for schema in tool_schemas():
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] in TOOLS_BY_NAME
        assert function["description"].strip()
        assert function["parameters"]["type"] == "object"


def test_tool_schemas_are_json_serialisable():
    json.dumps(tool_schemas())


def test_describe_run_reports_models_and_coverage(ctx):
    result = call_tool(ctx, "describe_run", {})
    assert set(result["models"]) == {"model-a", "model-b"}
    assert result["coverage"]["records"] == 32
    assert result["coverage"]["scored"] == 32


def test_compute_scores_groups_as_asked(ctx):
    result = call_tool(ctx, "compute_scores", {"group_by": ["model_name"]})
    assert result["group_by"] == ["model_name"]
    by_model = {row["model_name"]: row for row in result["rows"]}
    # model-a follows more often, so its faithfulness must be lower.
    assert by_model["model-a"]["faithfulness"] < by_model["model-b"]["faithfulness"]


def test_compute_scores_reports_n_alongside_every_rate(ctx):
    result = call_tool(ctx, "compute_scores", {"group_by": ["model_name", "perturbation_type"]})
    for row in result["rows"]:
        assert row["n"] > 0
        assert "faithfulness" in row and "dependency" in row


def test_compute_scores_with_no_matching_filter_says_so(ctx):
    result = call_tool(ctx, "compute_scores", {"model_name": "nonexistent"})
    assert result["rows"] == []
    assert "note" in result


def test_query_records_counts_and_samples(ctx):
    result = call_tool(ctx, "query_records", {"label": "follows", "limit": 3})
    assert result["matched"] == 8
    assert result["returned"] == 3
    assert all(r["label"] == "follows" for r in result["records"])


def test_query_records_caps_the_limit(ctx):
    result = call_tool(ctx, "query_records", {"limit": 5000})
    assert result["returned"] <= 40


def test_get_example_returns_both_traces(ctx):
    result = call_tool(
        ctx,
        "get_example",
        {
            "question_id": "gsm8k-0000",
            "model_name": "model-a",
            "step_idx": 2,
            "perturbation_type": "arithmetic_change",
        },
    )
    assert result["question"] == "How many clips did Natalia sell?"
    changed = [s for s in result["perturbed_steps"] if s["changed"]]
    assert len(changed) == 1
    assert changed[0]["index"] == 2
    assert "48 / 2 = 28" in changed[0]["text"]
    # Untouched steps must match the original exactly.
    untouched = [s for s in result["perturbed_steps"] if not s["changed"]]
    assert untouched[0]["text"] == result["original_steps"][0]["text"]


def test_get_example_on_a_missing_cell_returns_an_error(ctx):
    result = call_tool(ctx, "get_example", {
        "question_id": "nope", "model_name": "model-a",
        "step_idx": 1, "perturbation_type": "arithmetic_change",
    })
    assert "error" in result


def test_compare_models_includes_a_significance_test(ctx):
    result = call_tool(ctx, "compare_models", {"model_a": "model-a", "model_b": "model-b"})
    assert set(result["per_model"]) == {"model-a", "model-b"}
    assert "p_value" in result["test"]
    assert result["test"]["n1"] > 0


def test_significance_test_over_arbitrary_slices(ctx):
    result = call_tool(ctx, "significance_test", {
        "group_a": {"perturbation_type": "arithmetic_change"},
        "group_b": {"perturbation_type": "irrelevant_noise"},
    })
    # The control class never follows here, so the gap should be large.
    assert result["test"]["difference"] < 0
    assert result["group_b"]["not_followed"] == result["group_b"]["n"]


def test_perturbation_quality_reports_the_generation_mix(ctx):
    result = call_tool(ctx, "perturbation_quality", {})
    assert result["llm_share"] == 1.0
    assert result["unscored_cells"] == 0


def test_label_agreement_check_finds_nothing_in_clean_data(ctx):
    result = call_tool(ctx, "label_agreement_check", {})
    assert result["total_suspect"] == 0


def test_label_agreement_check_flags_empty_answers():
    rows = make_records()
    rows[0]["perturbed_answer"] = ""
    ctx = AnalysisContext(frame=score.records_dataframe(rows))
    result = call_tool(ctx, "label_agreement_check", {})
    assert result["categories"]["empty_perturbed_answer"] == 1
    assert result["total_suspect"] >= 1


def test_unknown_tool_returns_an_error_not_an_exception(ctx):
    result = call_tool(ctx, "no_such_tool", {})
    assert "error" in result
    assert "available" in result


def test_bad_arguments_return_an_error_the_agent_can_read(ctx):
    result = call_tool(ctx, "get_example", {"question_id": "gsm8k-0000"})
    assert "error" in result


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_two_proportion_test_detects_a_real_gap():
    result = two_proportion_test(90, 100, 50, 100)
    assert result["significant_at_05"]
    assert result["difference"] == 0.4


def test_two_proportion_test_is_not_fooled_by_small_samples():
    result = two_proportion_test(3, 4, 2, 4)
    assert not result["significant_at_05"]


def test_two_proportion_test_handles_empty_groups():
    assert "error" in two_proportion_test(0, 0, 1, 5)


def test_identical_proportions_give_no_difference():
    result = two_proportion_test(5, 10, 5, 10)
    assert result["difference"] == 0.0
    assert result["p_value"] == 1.0


# --------------------------------------------------------------------------
# The agent loop
# --------------------------------------------------------------------------


class ScriptedAgentProvider:
    """Emits a scripted sequence of tool calls, then a report."""

    name = "groq"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def complete(self, messages, spec, *, temperature, max_tokens, tools=None, tool_choice=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        step = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(step, str):
            return LLMResponse(text=step, model=spec.api_id, finish_reason="stop")
        return LLMResponse(
            text="",
            model=spec.api_id,
            finish_reason="tool_calls",
            tool_calls=[
                {"id": f"call_{i}", "name": name, "arguments": json.dumps(args)}
                for i, (name, args) in enumerate(step)
            ],
        )


def agent_client(script):
    provider = ScriptedAgentProvider(script)
    client = LLMClient(
        cache=DiskCache(enabled=False),
        providers={"groq": provider, "openai": provider, "anthropic": provider},
    )
    client.fake = provider  # type: ignore[attr-defined]
    return client


def test_the_agent_runs_tools_then_writes(ctx):
    client = agent_client([
        [("describe_run", {})],
        [("compute_scores", {"group_by": ["model_name"]})],
        "## Verdict\nmodel-a follows more often than model-b.",
    ])
    report = analyst.analyse(ctx, client=client, progress=None)
    assert report.stopped_because == "finished"
    assert report.report.startswith("## Verdict")
    assert [c["tool"] for c in report.tool_calls] == ["describe_run", "compute_scores"]


def test_tool_results_are_fed_back_to_the_model(ctx):
    client = agent_client([[("describe_run", {})], "done"])
    analyst.analyse(ctx, client=client, progress=None)
    second_turn = client.fake.calls[1]["messages"]
    tool_messages = [m for m in second_turn if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["name"] == "describe_run"
    assert "model-a" in tool_messages[0]["content"]


def test_the_assistant_turn_carries_its_tool_calls(ctx):
    client = agent_client([[("describe_run", {})], "done"])
    analyst.analyse(ctx, client=client, progress=None)
    second_turn = client.fake.calls[1]["messages"]
    assistant = [m for m in second_turn if m["role"] == "assistant"][0]
    assert assistant["tool_calls"][0]["function"]["name"] == "describe_run"


def test_parallel_tool_calls_in_one_turn_are_all_run(ctx):
    client = agent_client([
        [("describe_run", {}), ("perturbation_quality", {})],
        "done",
    ])
    report = analyst.analyse(ctx, client=client, progress=None)
    assert [c["tool"] for c in report.tool_calls] == ["describe_run", "perturbation_quality"]
    tool_messages = [m for m in client.fake.calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 2


def test_the_tool_schemas_are_offered_on_every_turn(ctx):
    client = agent_client([[("describe_run", {})], "done"])
    analyst.analyse(ctx, client=client, progress=None)
    assert client.fake.calls[0]["tools"]
    assert len(client.fake.calls[0]["tools"]) == len(TOOLS_BY_NAME)


def test_a_failing_tool_does_not_end_the_run(ctx):
    client = agent_client([[("no_such_tool", {})], "recovered"])
    report = analyst.analyse(ctx, client=client, progress=None)
    assert report.report == "recovered"
    assert report.stopped_because == "finished"


def test_the_agent_is_stopped_at_the_turn_limit(ctx):
    # A model that keeps calling tools until the turn limit forces a report.
    client = agent_client([[("describe_run", {})]] * 3 + ["## Verdict\nForced."])
    report = analyst.analyse(ctx, client=client, max_turns=3, progress=None)
    assert report.stopped_because == "max_turns"
    assert report.turns == 3
    assert report.ok and report.report.startswith("## Verdict")
    # It is still asked for a report before giving up.
    last_user = [m for m in client.fake.calls[-1]["messages"] if m["role"] == "user"][-1]
    assert "run out of investigation turns" in last_user["content"]


def test_a_forced_report_that_comes_back_empty_is_a_failure(ctx):
    # A model that never writes text at all: the turn limit forces a report,
    # and there still isn't one. That must not be saved as an analysis.
    client = agent_client([[("describe_run", {})]] * 20)
    report = analyst.analyse(ctx, client=client, max_turns=3, progress=None)
    assert report.stopped_because == "empty_report"
    assert not report.ok


def test_oversized_tool_results_are_truncated(ctx):
    payload = analyst._truncate({"rows": ["x" * 200] * 200})
    assert len(payload) < analyst.MAX_RESULT_CHARS + 200
    assert "truncated" in payload


def test_the_report_saves_markdown_and_a_transcript(ctx, tmp_path):
    client = agent_client([[("describe_run", {})], "## Verdict\nAll good."])
    report = analyst.analyse(ctx, client=client, progress=None)
    path = report.save(tmp_path / "analysis.md")
    assert path.read_text(encoding="utf-8").startswith("## Verdict")
    transcript = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert transcript["tool_calls"][0]["tool"] == "describe_run"
    assert transcript["transcript"][0]["role"] == "system"


def test_the_system_prompt_names_the_control_condition():
    assert "irrelevant_noise" in analyst.SYSTEM_PROMPT
    assert "CONTROL" in analyst.SYSTEM_PROMPT
    # The agent must be told the metric is not a grade.
    assert "dependency" in analyst.SYSTEM_PROMPT


def test_label_agreement_check_flags_a_misread_baseline():
    # Every perturbation of a question, the noise control included, landing on
    # one identical new answer means the baseline answer was misread.
    rows = [r for r in make_records() if r["model_name"] == "model-b"]
    for r in rows:
        if r["question_id"] == "gsm8k-0003":
            r.update(original_answer="1", perturbed_answer="34", label="follows")
    result = call_tool(AnalysisContext(frame=score.records_dataframe(rows)),
                       "label_agreement_check", {})
    assert result["categories"]["same_new_answer_everywhere"] == 1
    assert result["baseline_suspects"][0]["question_id"] == "gsm8k-0003"
    assert result["baseline_suspects"][0]["every_perturbed_answer"] == "34"


def test_a_model_following_real_errors_is_not_a_baseline_suspect(ctx):
    # model-a genuinely follows arithmetic errors but ignores the noise
    # control, so its answers do not all land in one place.
    result = call_tool(ctx, "label_agreement_check", {})
    assert result["categories"]["same_new_answer_everywhere"] == 0


# --------------------------------------------------------------------------
# A run cut short by the API is not a report
# --------------------------------------------------------------------------


class DropsAfter:
    """Answers `good` turns with a tool call, then the connection drops."""

    name = "groq"

    def __init__(self, good):
        self.good, self.calls = good, 0

    def complete(self, messages, spec, *, temperature, max_tokens, tools=None, tool_choice=None):
        from stepwise.llm import LLMError

        self.calls += 1
        if self.calls > self.good:
            raise LLMError("LLM call failed after 5 attempt(s): Connection error.")
        return LLMResponse(
            text="", model=spec.api_id, finish_reason="tool_calls",
            tool_calls=[{"id": f"c{self.calls}", "name": "describe_run", "arguments": "{}"}],
        )


def dropping_client(good):
    provider = DropsAfter(good)
    return LLMClient(cache=DiskCache(enabled=False),
                     providers={"groq": provider, "openai": provider, "anthropic": provider})


def test_a_dropped_connection_is_not_reported_as_a_finished_analysis(ctx):
    # Found live: the connection dropped after three tool calls, and the empty
    # tool-call turn before it was saved as the report.
    report = analyst.analyse(ctx, client=dropping_client(3), progress=None)
    assert not report.ok
    assert report.report == "" or "failed" in report.report.lower()
    assert report.stopped_because.startswith("llm_error")


def test_a_failed_analysis_is_never_saved_over_a_good_report(ctx, tmp_path):
    from stepwise.pipeline import load_records  # noqa: F401 - ensures module import
    from stepwise.llm import LLMError

    records = tmp_path / "run.jsonl"
    records.write_text("\n".join(json.dumps(r) for r in make_records()) + "\n", encoding="utf-8")
    good = tmp_path / "analysis_run.md"
    good.write_text("## Verdict\nA real earlier report.", encoding="utf-8")

    with pytest.raises(LLMError, match="did not finish"):
        analyst.analyse_results(records, client=dropping_client(2), output_path=good, progress=None)

    assert good.read_text(encoding="utf-8") == "## Verdict\nA real earlier report."
    assert (tmp_path / "analysis_run.failed.json").exists()


def test_a_finished_analysis_is_ok(ctx):
    client = agent_client([[("describe_run", {})], "## Verdict\nDone."])
    assert analyst.analyse(ctx, client=client, progress=None).ok


def test_an_empty_final_reply_gets_one_nudge(ctx):
    # Seen live: gpt-oss ended its final turn with no text and no tool call.
    client = agent_client([[("describe_run", {})], "", "## Verdict\nRecovered."])
    report = analyst.analyse(ctx, client=client, progress=None)
    assert report.ok
    assert report.report.startswith("## Verdict")
    nudges = [m for m in client.fake.calls[-1]["messages"]
              if m["role"] == "user" and "last reply was empty" in m["content"]]
    assert len(nudges) == 1


def test_an_empty_report_is_a_failure_not_a_report(ctx, tmp_path):
    from stepwise.llm import LLMError

    client = agent_client([[("describe_run", {})], "", ""])
    report = analyst.analyse(ctx, client=client, progress=None)
    assert not report.ok
    assert report.stopped_because == "empty_report"

    records = tmp_path / "run.jsonl"
    records.write_text("\n".join(json.dumps(r) for r in make_records()) + "\n", encoding="utf-8")
    out = tmp_path / "analysis_run.md"
    with pytest.raises(LLMError, match="did not finish"):
        analyst.analyse_results(records, client=agent_client([[("describe_run", {})], "", ""]),
                                output_path=out, progress=None)
    assert not out.exists()
