"""Shared test fixtures.

Nothing in the unit suite touches the network: :class:`FakeProvider` stands in
for Groq and returns scripted text, so the tests exercise prompt assembly,
cleaning, validation and splicing without an API key.
"""

from __future__ import annotations

import pytest

from stepwise.config import ModelSpec
from stepwise.llm import DiskCache, LLMClient, LLMResponse


class FakeProvider:
    """A scripted stand-in for a real provider.

    ``responses`` may be a single string, a list consumed in order, or a
    callable taking the message list. Every call is recorded on ``calls`` so a
    test can assert on the prompt that was actually sent.
    """

    name = "groq"

    def __init__(self, responses="perturbed step"):
        self.responses = responses
        self.calls: list[dict] = []

    def complete(self, messages, spec, *, temperature, max_tokens, tools=None, tool_choice=None):
        self.calls.append(
            {
                "messages": list(messages),
                "model": spec.api_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": tools,
            }
        )
        if callable(self.responses):
            text = self.responses(list(messages))
        elif isinstance(self.responses, (list, tuple)):
            index = min(len(self.calls) - 1, len(self.responses) - 1)
            text = self.responses[index]
        else:
            text = self.responses
        return LLMResponse(text=text, model=spec.api_id, finish_reason="stop")

    @property
    def last_user_message(self) -> str:
        return self.calls[-1]["messages"][-1]["content"]

    @property
    def last_system_message(self) -> str:
        return self.calls[-1]["messages"][0]["content"]


@pytest.fixture
def fake_provider():
    return FakeProvider()


@pytest.fixture
def make_client(tmp_path):
    """Build an :class:`LLMClient` wired to a fake provider and a temp cache."""

    def _make(responses="perturbed step", *, use_cache=False):
        provider = FakeProvider(responses)
        client = LLMClient(
            cache=DiskCache(tmp_path / "cache", enabled=use_cache),
            providers={"groq": provider, "openai": provider, "anthropic": provider},
        )
        client.fake = provider  # type: ignore[attr-defined]
        return client

    return _make


@pytest.fixture
def spec():
    return ModelSpec(name="test-model", provider="groq", api_id="test/model")


# --------------------------------------------------------------------------
# Sample chains of thought, one per shape the extractor has to handle.
# --------------------------------------------------------------------------

MARKED_COT = (
    "Step 1: Natalia sold 48 clips in April.\n"
    "Step 2: In May she sold half as many, so 48 / 2 = 24.\n"
    "Step 3: Altogether she sold 48 + 24 = 72 clips."
)

NUMBERED_COT = (
    "1. Weng earns $12 an hour, which is $0.20 per minute.\n"
    "2. She babysat for 50 minutes.\n"
    "3. So she earned 50 * 0.2 = $10."
)

PROSE_COT = (
    "Natalia sold 48 clips in April. "
    "In May she sold half as many, i.e. 48 / 2 = 24 clips. "
    "Altogether that is 48 + 24 = 72. "
    "The answer is 72."
)

PARAGRAPH_COT = (
    "First, note that the tank starts full.\n\n"
    "Then it drains at 6 litres per minute for 7 minutes, so 42 litres leave.\n\n"
    "That leaves 58 litres in the tank."
)


@pytest.fixture(params=[MARKED_COT, NUMBERED_COT, PROSE_COT, PARAGRAPH_COT], ids=
                ["marked", "numbered", "prose", "paragraphs"])
def any_cot(request):
    """Every supported CoT shape, for round-trip invariants."""
    return request.param
