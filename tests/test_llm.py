"""Unit tests for the provider layer: request shaping, output cleaning, errors.

Every behaviour here was prompted by the first live run against Groq on
2026-09-11 — Qwen leaking its <think> scratchpad, gpt-oss-20b calling a
built-in tool, and a 404 reported as "failed after 5 attempts". The fake
client below stands in for the Groq SDK so none of it needs the network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stepwise.config import MODEL_REGISTRY, ModelSpec
from stepwise.llm import GroqProvider, LLMError, _with_retries, strip_thinking


class FakeCompletions:
    """Records the kwargs it was called with; returns or raises as told."""

    def __init__(self, content="Answer: 72", error=None):
        self.content, self.error, self.kwargs = content, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.content, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
        )


class FakeAPIError(Exception):
    """Shaped like the SDK's errors: a message plus a status code."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def provider_with(completions):
    return GroqProvider(client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))


def call(provider, spec, tools=None):
    return provider.complete(
        [{"role": "user", "content": "hi"}], spec, temperature=0.0, max_tokens=50, tools=tools
    )


PLAIN = ModelSpec(name="m", provider="groq", api_id="test/m")
THINKER = ModelSpec(
    name="q", provider="groq", api_id="test/q", extra=(("reasoning_effort", "none"),)
)


# --------------------------------------------------------------------------
# strip_thinking
# --------------------------------------------------------------------------


def test_a_closed_think_block_is_removed():
    assert strip_thinking("<think>\nlet me see...\n</think>\nAnswer: 72") == "Answer: 72"


def test_an_unclosed_think_block_drops_everything_after_it():
    # The model ran out of tokens mid-scratchpad. There is no answer, and the
    # numbers inside half-finished thinking must not be scraped as one.
    assert strip_thinking("<think>\n48 / 2 = 24, so maybe 76") == ""


def test_text_without_thinking_is_only_trimmed():
    assert strip_thinking("  Step 1: 48 / 2 = 24.\nAnswer: 72 ") == "Step 1: 48 / 2 = 24.\nAnswer: 72"


def test_think_tags_are_matched_case_insensitively():
    assert strip_thinking("<THINK>x</THINK>Answer: 1") == "Answer: 1"


def test_several_think_blocks_are_all_removed():
    assert strip_thinking("<think>a</think>Step 1.<think>b</think> Answer: 2") == "Step 1. Answer: 2"


# --------------------------------------------------------------------------
# Request shaping
# --------------------------------------------------------------------------


def test_extra_params_are_sent_in_extra_body():
    completions = FakeCompletions()
    call(provider_with(completions), THINKER)
    assert completions.kwargs["extra_body"] == {"reasoning_effort": "none"}


def test_no_extra_body_when_a_spec_has_none():
    completions = FakeCompletions()
    call(provider_with(completions), PLAIN)
    assert "extra_body" not in completions.kwargs


def test_think_blocks_are_stripped_from_provider_output():
    completions = FakeCompletions(content="<think>hmm</think>Answer: 72")
    assert call(provider_with(completions), PLAIN).text == "Answer: 72"


def test_tools_are_only_sent_when_offered():
    completions = FakeCompletions()
    call(provider_with(completions), PLAIN)
    assert "tools" not in completions.kwargs and "tool_choice" not in completions.kwargs


def test_the_qwen_models_are_registered_with_thinking_off():
    for name in ("qwen3.6-27b", "qwen3.8-27b"):
        assert dict(MODEL_REGISTRY[name].extra) == {"reasoning_effort": "none"}


def test_model_specs_with_extras_stay_hashable():
    # ModelSpec is frozen; a dict field would make hash() raise.
    assert isinstance(hash(THINKER), int)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_a_builtin_tool_call_gets_an_explanatory_error():
    error = FakeAPIError("Error code: 400 - Tool choice is none, but model called a tool", 400)
    with pytest.raises(LLMError, match="built-in tools"):
        call(provider_with(FakeCompletions(error=error)), PLAIN)


def test_a_non_retryable_error_reports_the_attempts_actually_made():
    def fail():
        raise FakeAPIError("Error code: 404 - model does not exist", 404)

    with pytest.raises(LLMError, match=r"after 1 attempt\(s\)"):
        _with_retries(fail)


def test_an_auth_error_is_raised_without_retrying():
    calls = []

    def fail():
        calls.append(1)
        raise FakeAPIError("invalid api key", 401)

    with pytest.raises(LLMError):
        _with_retries(fail)
    assert len(calls) == 1


def test_a_groq_output_parse_failure_is_retried():
    # A 400 that is really a flaky generation, not a bad request: seen live
    # when gpt-oss wrote its reasoning as plain text mid tool call.
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise FakeAPIError(
                "Error code: 400 - {'error': {'code': 'output_parse_failed', "
                "'message': 'Parsing failed.'}}", 400)
        return "ok"

    import stepwise.llm as llm_module
    original = llm_module.time.sleep
    llm_module.time.sleep = lambda s: None
    try:
        assert _with_retries(flaky) == "ok"
    finally:
        llm_module.time.sleep = original
    assert len(calls) == 2


def test_an_ordinary_bad_request_is_still_not_retried():
    calls = []

    def bad():
        calls.append(1)
        raise FakeAPIError("Error code: 400 - invalid model parameter", 400)

    with pytest.raises(LLMError, match=r"after 1 attempt\(s\)"):
        _with_retries(bad)
    assert len(calls) == 1


# --------------------------------------------------------------------------
# A4: bring-your-own-key
# --------------------------------------------------------------------------


def test_byo_key_builds_provider_with_that_key():
    from stepwise.llm import LLMClient
    from stepwise.config import resolve_model

    client = LLMClient(api_keys={"groq": "gsk_visitor_key"})
    provider = client.provider_for(resolve_model("gpt-oss-120b"))
    assert provider._api_key == "gsk_visitor_key"


def test_empty_byo_key_falls_back_to_env_path():
    from stepwise.llm import LLMClient
    from stepwise.config import resolve_model

    client = LLMClient(api_keys={"groq": ""})
    provider = client.provider_for(resolve_model("gpt-oss-120b"))
    # Empty string is dropped, so the provider uses its normal env lookup.
    assert provider._api_key is None


def test_explicit_key_overrides_env_in_make_client(monkeypatch):
    # The OpenAI-compatible base prefers the explicit key when forming a client.
    from stepwise.llm import GroqProvider

    captured = {}

    class _FakeGroq:
        def __init__(self, api_key=None):
            captured["key"] = api_key

    # Install the stand-in module FIRST. monkeypatch.setattr resolves the
    # dotted path by importing it, so with the real groq package absent the
    # patch itself raises ModuleNotFoundError before the fallback below could
    # ever run. The suite is meant to pass with no provider SDK installed.
    import sys
    import types

    if "groq" not in sys.modules:
        module = types.ModuleType("groq")
        module.Groq = _FakeGroq
        monkeypatch.setitem(sys.modules, "groq", module)
    monkeypatch.setattr("groq.Groq", _FakeGroq, raising=False)
    provider = GroqProvider(api_key="explicit-key")
    _ = provider.client
    assert captured["key"] == "explicit-key"
