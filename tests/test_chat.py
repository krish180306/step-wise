"""Unit tests for the interactive chat agent and its SSE endpoint.

A scripted provider emits tool calls, so the loop, the event stream, session
state and the SSE framing are all exercised without an API key.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from stepwise import demo, score
from stepwise.agent import chat as chat_mod
from stepwise.agent.chat import (
    MAX_HISTORY_MESSAGES,
    ChatSession,
    SessionStore,
    ask,
    stream_reply,
)
from stepwise.agent.tools import AnalysisContext, call_tool
from stepwise.dashboard import app as dashboard_app
from stepwise.llm import DiskCache, LLMClient, LLMResponse


@pytest.fixture
def ctx():
    result = demo.run_demo(dataset="gsm8k", n_samples=2, progress=None)
    return AnalysisContext(
        frame=score.records_dataframe(result.records),
        traces={k: v.to_dict() for k, v in result.traces.items()},
        dataset="demo",
    )


class ScriptedProvider:
    """Emits scripted tool calls, then text."""

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
            text="", model=spec.api_id, finish_reason="tool_calls",
            tool_calls=[
                {"id": f"c{i}", "name": n, "arguments": json.dumps(a)}
                for i, (n, a) in enumerate(step)
            ],
        )


def client_for(script):
    provider = ScriptedProvider(script)
    client = LLMClient(
        cache=DiskCache(enabled=False),
        providers={"groq": provider, "openai": provider, "anthropic": provider},
    )
    client.fake = provider  # type: ignore[attr-defined]
    return client


def session_for(ctx):
    return ChatSession(session_id="s1", run="demo", ctx=ctx)


def collect(ctx, script, message="which model is least faithful?"):
    client = client_for(script)
    return list(stream_reply(session_for(ctx), message, client=client)), client


# --------------------------------------------------------------------------
# The event stream
# --------------------------------------------------------------------------


def test_events_arrive_in_loop_order(ctx):
    events, _ = collect(ctx, [[("describe_run", {})], "model-a is least faithful."])
    assert [e["type"] for e in events] == [
        "start", "tool_call", "tool_result", "text", "done",
    ]


def test_a_tool_call_event_names_the_tool_and_its_arguments(ctx):
    events, _ = collect(ctx, [[("compute_scores", {"group_by": ["model_name"]})], "done"])
    call = next(e for e in events if e["type"] == "tool_call")
    assert call["tool"] == "compute_scores"
    assert call["arguments"] == {"group_by": ["model_name"]}


def test_a_tool_result_event_carries_a_short_summary(ctx):
    events, _ = collect(ctx, [[("describe_run", {})], "done"])
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["tool"] == "describe_run"
    assert result["summary"] and len(result["summary"]) < 80


def test_parallel_tool_calls_all_stream(ctx):
    events, _ = collect(
        ctx, [[("describe_run", {}), ("perturbation_quality", {})], "done"]
    )
    assert [e["tool"] for e in events if e["type"] == "tool_call"] == [
        "describe_run", "perturbation_quality",
    ]
    assert sum(1 for e in events if e["type"] == "tool_result") == 2


def test_the_answer_is_the_text_event(ctx):
    events, _ = collect(ctx, [[("describe_run", {})], "sim-detached, at 82%."])
    assert next(e for e in events if e["type"] == "text")["content"] == "sim-detached, at 82%."


def test_an_llm_failure_becomes_an_error_event(ctx):
    class Failing:
        name = "groq"

        def complete(self, *a, **k):
            from stepwise.llm import LLMError
            raise LLMError("rate limited")

    client = LLMClient(cache=DiskCache(enabled=False),
                       providers={"groq": Failing(), "openai": Failing(), "anthropic": Failing()})
    events = list(stream_reply(session_for(ctx), "hi", client=client))
    assert events[-1]["type"] == "error"
    assert "rate limited" in events[-1]["message"]


def test_a_failing_tool_does_not_break_the_stream(ctx):
    events, _ = collect(ctx, [[("no_such_tool", {})], "recovered"])
    assert next(e for e in events if e["type"] == "text")["content"] == "recovered"
    assert events[-1]["type"] == "done"


def test_the_turn_budget_forces_an_answer(ctx):
    client = client_for([[("describe_run", {})]] * 20)
    events = list(stream_reply(session_for(ctx), "hi", client=client, max_turns=2))
    assert sum(1 for e in events if e["type"] == "tool_call") == 2
    last_user = [m for m in client.fake.calls[-1]["messages"] if m["role"] == "user"][-1]
    assert "tool budget" in last_user["content"]


# --------------------------------------------------------------------------
# Tools-first: no run data in context until a tool returns it
# --------------------------------------------------------------------------


def test_the_first_request_carries_no_run_data(ctx):
    _, client = collect(ctx, [[("describe_run", {})], "done"])
    first = client.fake.calls[0]["messages"]
    assert [m["role"] for m in first] == ["system", "user"]
    blob = json.dumps(first)
    # No step text, no per-cell rows — just the prompt and the question.
    assert "gsm8k-fixture" not in blob
    assert len(blob) < 12_000


def test_run_data_only_enters_context_through_tool_results(ctx):
    _, client = collect(ctx, [[("query_records", {"limit": 3})], "done"])
    second = client.fake.calls[1]["messages"]
    tool_messages = [m for m in second if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "gsm8k-fixture" in tool_messages[0]["content"]


def test_the_tool_schemas_are_offered_every_turn(ctx):
    _, client = collect(ctx, [[("describe_run", {})], "done"])
    assert all(c["tools"] for c in client.fake.calls[:-1])


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------


def test_history_persists_across_messages(ctx):
    client = client_for(["first answer", "second answer"])
    session = session_for(ctx)
    list(stream_reply(session, "one", client=client))
    list(stream_reply(session, "two", client=client))
    roles = [m["role"] for m in session.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert session.turn_count == 2


def test_the_second_message_sees_the_first(ctx):
    client = client_for(["a", "b"])
    session = session_for(ctx)
    list(stream_reply(session, "what about arithmetic?", client=client))
    list(stream_reply(session, "and logic?", client=client))
    sent = json.dumps(client.fake.calls[-1]["messages"])
    assert "what about arithmetic?" in sent


def test_history_is_trimmed_but_keeps_the_system_prompt(ctx):
    client = client_for(["ok"])
    session = session_for(ctx)
    for i in range(MAX_HISTORY_MESSAGES):
        list(stream_reply(session, f"question {i}", client=client))
    assert len(session.messages) <= MAX_HISTORY_MESSAGES + 4
    assert session.messages[0]["role"] == "system"


def test_trimming_never_orphans_a_tool_call(ctx):
    """A tool_use turn separated from its result is a 400 from every provider."""
    client = client_for([[("describe_run", {})], "ok"])
    session = session_for(ctx)
    for i in range(40):
        list(stream_reply(session, f"q{i}", client=client))
    session.trim()
    ids = {m["tool_call_id"] for m in session.messages if m.get("role") == "tool"}
    announced = {
        c["id"]
        for m in session.messages if m.get("role") == "assistant"
        for c in m.get("tool_calls", [])
    }
    assert ids <= announced


def test_sessions_are_keyed_and_reused(ctx):
    store = SessionStore()
    a = store.get_or_create(None, "demo", ctx, "gpt-oss-120b")
    b = store.get_or_create(a.session_id, "demo", ctx, "gpt-oss-120b")
    assert a is b
    assert len(store) == 1


def test_switching_run_starts_a_new_session(ctx):
    store = SessionStore()
    a = store.get_or_create(None, "demo", ctx, "gpt-oss-120b")
    b = store.get_or_create(a.session_id, "other", ctx, "gpt-oss-120b")
    assert a.session_id != b.session_id


def test_sessions_are_evicted_past_the_cap(ctx):
    store = SessionStore(max_sessions=3)
    for _ in range(6):
        store.get_or_create(None, "demo", ctx, "gpt-oss-120b")
    assert len(store) <= 3


def test_reset_forgets_a_session(ctx):
    store = SessionStore()
    session = store.get_or_create(None, "demo", ctx, "gpt-oss-120b")
    assert store.reset(session.session_id) is True
    assert store.get(session.session_id) is None


def test_ask_is_a_one_shot_wrapper(ctx):
    client = client_for([[("describe_run", {})], "four models."])
    result = ask(ctx, "how many models?", client=client)
    assert result["answer"] == "four models."
    assert result["tool_calls"] == ["describe_run"]


# --------------------------------------------------------------------------
# The search_steps tool
# --------------------------------------------------------------------------


def test_search_steps_finds_literal_text(ctx):
    result = call_tool(ctx, "search_steps", {"query": "clips", "limit": 5})
    assert result["matched"] > 0
    assert all("clips" in (r["original_step"] + r["perturbed_step"]).lower()
               for r in result["results"])


def test_search_steps_can_target_the_perturbed_column(ctx):
    result = call_tool(ctx, "search_steps", {"query": "48", "field": "perturbed"})
    assert result["searched"] == ["perturbed_step"]


def test_search_steps_supports_regex(ctx):
    result = call_tool(ctx, "search_steps", {"query": r"\d+ / \d+", "regex": True})
    assert result["matched"] >= 0


def test_a_bad_regex_returns_an_error_not_a_crash(ctx):
    assert "error" in call_tool(ctx, "search_steps", {"query": "([unclosed", "regex": True})


def test_search_steps_warns_it_is_not_a_rate(ctx):
    result = call_tool(ctx, "search_steps", {"query": "the"})
    assert "not a rate" in result["note"] or "not semantic" in result["note"].lower()


def test_search_steps_respects_filters(ctx):
    result = call_tool(ctx, "search_steps", {"query": "a", "model_name": "sim-faithful"})
    assert all(r["model_name"] == "sim-faithful" for r in result["results"])


# --------------------------------------------------------------------------
# The SSE endpoint
# --------------------------------------------------------------------------


@pytest.fixture
def http(tmp_path, monkeypatch, ctx):
    monkeypatch.setattr(dashboard_app, "RESULTS_DIR", tmp_path)
    dashboard_app._cached_payload.cache_clear()
    demo.run_demo(dataset="gsm8k", n_samples=2, progress=None).save(tmp_path / "demo.jsonl")

    client = client_for([[("describe_run", {})], "sim-detached is least dependent."])
    monkeypatch.setattr(chat_mod, "LLMClient", lambda *a, **k: client)
    chat_mod.SESSIONS = chat_mod.SessionStore()
    monkeypatch.setattr(dashboard_app, "SESSIONS", chat_mod.SESSIONS)
    return TestClient(dashboard_app.app)


def parse_sse(text):
    return [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def test_chat_streams_sse_frames(http):
    response = http.post("/api/chat", json={"run": "demo", "message": "which model?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [e["type"] for e in events][:3] == ["start", "tool_call", "tool_result"]
    assert events[-1]["type"] == "done"


def test_the_answer_comes_back_over_the_stream(http):
    events = parse_sse(
        http.post("/api/chat", json={"run": "demo", "message": "which?"}).text
    )
    assert any(e.get("content") == "sim-detached is least dependent." for e in events)


def test_the_stream_returns_a_session_id_for_follow_ups(http):
    events = parse_sse(http.post("/api/chat", json={"run": "demo", "message": "hi"}).text)
    assert events[0]["session_id"]


def test_an_empty_message_is_rejected(http):
    assert http.post("/api/chat", json={"run": "demo", "message": "   "}).status_code == 400


def test_an_overlong_message_is_rejected(http):
    response = http.post("/api/chat", json={"run": "demo", "message": "x" * 5000})
    assert response.status_code == 400


def test_chatting_about_an_unknown_run_is_a_404(http):
    assert http.post("/api/chat", json={"run": "nope", "message": "hi"}).status_code == 404


def test_chat_status_reports_availability(http):
    payload = http.get("/api/chat/status").json()
    assert "available" in payload and "model" in payload and "requires" in payload


def test_reset_endpoint(http):
    events = parse_sse(http.post("/api/chat", json={"run": "demo", "message": "hi"}).text)
    session_id = events[0]["session_id"]
    assert http.post("/api/chat/reset", json={"session_id": session_id}).json()["reset"] is True
    assert http.post("/api/chat/reset", json={"session_id": session_id}).json()["reset"] is False


def test_the_chat_panel_is_in_the_page(http):
    body = http.get("/").text
    assert 'id="chat-log"' in body
    assert 'id="chat-send"' in body


def test_an_empty_chat_reply_gets_one_nudge(ctx):
    events, client = collect(ctx, [[("describe_run", {})], "", "Recovered answer."])
    assert next(e for e in events if e["type"] == "text")["content"] == "Recovered answer."
    assert events[-1]["type"] == "done"


def test_two_empty_chat_replies_become_an_error_not_a_blank_answer(ctx):
    events, _ = collect(ctx, [[("describe_run", {})], "", ""])
    assert events[-1]["type"] == "error"
    assert not any(e["type"] == "text" and not e["content"].strip() for e in events)
