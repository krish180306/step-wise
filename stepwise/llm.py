"""Provider-agnostic LLM access with a disk cache.

Groq is the default backend (free tier, four production chat models, all of
which support tool calling). Providers are registered by name so a model row
in :mod:`stepwise.config` is all it takes to route somewhere else.

Every completion goes through :class:`DiskCache`, keyed by the caller's
structured key — normally ``(question_id, step_idx, perturbation_type,
model)``. Reruns of an evaluation therefore cost nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .config import CACHE_DIR, ModelSpec, api_key, resolve_model


class LLMError(RuntimeError):
    """Raised when a provider cannot be reached or refuses the request."""


@dataclass
class LLMResponse:
    """A single completion, whether it came from the API or the cache."""

    text: str
    model: str
    cached: bool = False
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "tool_calls": self.tool_calls,
        }

    @classmethod
    def from_dict(cls, payload: dict, *, cached: bool = True) -> "LLMResponse":
        return cls(
            text=payload.get("text", ""),
            model=payload.get("model", ""),
            cached=cached,
            finish_reason=payload.get("finish_reason", ""),
            usage=payload.get("usage", {}) or {},
            tool_calls=payload.get("tool_calls", []) or [],
        )


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------


def make_cache_key(**parts: Any) -> str:
    """Build a stable, human-readable cache key from keyword parts.

    Keys are sorted so argument order never changes the key, and ``None``
    values are dropped so an absent field reads the same as an omitted one.
    """
    items = [(k, v) for k, v in sorted(parts.items()) if v is not None]
    return "|".join(f"{k}={v}" for k, v in items)


class DiskCache:
    """A sharded JSON cache on disk, safe for threaded pipeline workers."""

    def __init__(self, root: Path | str = CACHE_DIR, *, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        # Shard on the first two hex chars: keeps directories small enough for
        # a filesystem to list quickly after tens of thousands of calls.
        shard = self.root / digest[:2]
        return shard / f"{digest}.json"

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A half-written entry is not worth a crash; treat it as a miss.
            self.misses += 1
            return None
        self.hits += 1
        return payload.get("value")

    def put(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        record = {"key": key, "stored_at": time.time(), "value": value}
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a killed run never leaves a torn file.
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


class Provider(Protocol):
    """What a backend must implement to serve as a target model."""

    name: str

    def complete(
        self,
        messages: Sequence[dict],
        spec: ModelSpec,
        *,
        temperature: float,
        max_tokens: int,
        tools: Sequence[dict] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        ...


class _OpenAICompatibleProvider:
    """Shared implementation for the OpenAI-shaped chat completions APIs."""

    name = "openai-compatible"
    env_var = "OPENAI_API_KEY"

    def __init__(self, client: Any = None):
        self._client = client

    def _make_client(self) -> Any:  # pragma: no cover - provider specific
        raise NotImplementedError

    @property
    def client(self) -> Any:
        if self._client is None:
            if not api_key(self.name):
                raise LLMError(
                    f"{self.env_var} is not set. Export it, or put it in a .env file "
                    f"next to main.py."
                )
            self._client = self._make_client()
        return self._client

    def complete(
        self,
        messages: Sequence[dict],
        spec: ModelSpec,
        *,
        temperature: float,
        max_tokens: int,
        tools: Sequence[dict] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": spec.api_id,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = tool_choice or "auto"
        if spec.extra:
            # Provider-specific switches, e.g. Qwen's reasoning_effort, ride in
            # extra_body so the SDK passes them through untouched.
            kwargs["extra_body"] = dict(spec.extra)

        started = time.time()
        try:
            completion = _with_retries(lambda: self.client.chat.completions.create(**kwargs))
        except LLMError as exc:
            if "model called a tool" in str(exc) and not tools:
                raise LLMError(
                    f"{spec.api_id} tried to call one of its built-in tools, which this "
                    f"pipeline does not provide, and the provider rejected the request. "
                    f"It is deterministic, so retrying will not help: gpt-oss-20b does it "
                    f"on every arithmetic question tested (2026-09-11). Use a different "
                    f"model for arithmetic datasets."
                ) from exc
            raise
        choice = completion.choices[0]
        message = choice.message
        calls = []
        for call in getattr(message, "tool_calls", None) or []:
            calls.append(
                {
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                }
            )
        usage = {}
        if getattr(completion, "usage", None):
            usage = {
                "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(completion.usage, "completion_tokens", 0),
            }
        return LLMResponse(
            text=strip_thinking(message.content or ""),
            model=spec.api_id,
            finish_reason=choice.finish_reason or "",
            usage=usage,
            tool_calls=calls,
            latency_s=round(time.time() - started, 3),
        )


class GroqProvider(_OpenAICompatibleProvider):
    name = "groq"
    env_var = "GROQ_API_KEY"

    def _make_client(self) -> Any:
        from groq import Groq

        return Groq(api_key=api_key("groq"))


class OpenAIProvider(_OpenAICompatibleProvider):
    name = "openai"
    env_var = "OPENAI_API_KEY"

    def _make_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(api_key=api_key("openai"))


class AnthropicProvider:
    """Anthropic's Messages API, shaped to look like the others."""

    name = "anthropic"
    env_var = "ANTHROPIC_API_KEY"

    def __init__(self, client: Any = None):
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            if not api_key("anthropic"):
                raise LLMError("ANTHROPIC_API_KEY is not set.")
            from anthropic import Anthropic

            self._client = Anthropic(api_key=api_key("anthropic"))
        return self._client

    def complete(
        self,
        messages: Sequence[dict],
        spec: ModelSpec,
        *,
        temperature: float,
        max_tokens: int,
        tools: Sequence[dict] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        kwargs: dict[str, Any] = {
            "model": spec.api_id,
            "messages": convo,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            # Anthropic's tool schema differs; translate the OpenAI shape.
            kwargs["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters", {}),
                }
                for t in tools
            ]

        started = time.time()
        message = _with_retries(lambda: self.client.messages.create(**kwargs))
        text_parts, calls = [], []
        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    {"id": block.id, "name": block.name, "arguments": json.dumps(block.input)}
                )
        return LLMResponse(
            text="".join(text_parts).strip(),
            model=spec.api_id,
            finish_reason=message.stop_reason or "",
            usage={
                "prompt_tokens": message.usage.input_tokens,
                "completion_tokens": message.usage.output_tokens,
            },
            tool_calls=calls,
            latency_s=round(time.time() - started, 3),
        )


PROVIDERS: dict[str, Callable[[], Provider]] = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove a reasoning model's <think> scratchpad from its visible output.

    Qwen is asked not to think at all (``config._NO_THINKING``); this is the
    backstop for any model that thinks anyway. An *unclosed* <think> means the
    model ran out of tokens mid-scratchpad and never wrote an answer, so
    everything from the tag onward is dropped. An empty answer is recorded as
    unscored; a number scraped from half-finished thinking would be scored as
    though the model had meant it.
    """
    text = _THINK_BLOCK_RE.sub("", text)
    opened = text.lower().find("<think>")
    if opened != -1:
        text = text[:opened]
    return text.strip()


def _with_retries(fn: Callable[[], Any], *, attempts: int = 5, base_delay: float = 2.0) -> Any:
    """Call ``fn``, backing off on rate limits and transient server errors.

    Groq's free tier meters requests per minute, so a run of any size will hit
    429s; the backoff is what keeps a long evaluation unattended.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # provider SDKs raise their own types
            last = exc
            status = getattr(exc, "status_code", None)
            retryable = status in (408, 409, 429, 500, 502, 503, 504) or status is None
            name = type(exc).__name__.lower()
            if "ratelimit" in name or "timeout" in name or "connection" in name:
                retryable = True
            if "output_parse_failed" in str(exc):
                # A 400, but not a bad request: gpt-oss sometimes writes its
                # private reasoning as plain text mid tool call, and Groq can't
                # parse the reply. Generation is stochastic, so the same
                # request usually succeeds on a second try. Seen live on
                # 2026-09-18 in the report agent.
                retryable = True
            if "authentication" in name or "permission" in name or status in (401, 403):
                raise LLMError(str(exc)) from exc
            if not retryable or attempt == attempts - 1:
                break
            delay = _retry_after(exc) or base_delay * (2**attempt) + random.uniform(0, 0.75)
            time.sleep(min(delay, 60.0))
    # Report the attempts actually made: a 404 fails on the first try, and
    # "after 5 attempts" would send someone looking for a flaky network.
    raise LLMError(f"LLM call failed after {attempt + 1} attempt(s): {last}") from last


def _retry_after(exc: Exception) -> float | None:
    """Honour a provider's Retry-After header when it sends one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
        value = headers.get(key) if hasattr(headers, "get") else None
        if value:
            try:
                return float(str(value).rstrip("s"))
            except ValueError:
                continue
    return None


# --------------------------------------------------------------------------
# Client facade
# --------------------------------------------------------------------------


class LLMClient:
    """The single entry point the rest of the codebase calls.

    ``providers`` can be overridden wholesale in tests, which is how the unit
    suite runs without touching the network.
    """

    def __init__(
        self,
        *,
        cache: DiskCache | None = None,
        providers: dict[str, Provider] | None = None,
        use_cache: bool = True,
    ):
        self.cache = cache if cache is not None else DiskCache(enabled=use_cache)
        self._overrides = providers or {}
        self._instances: dict[str, Provider] = {}
        self.calls = 0

    def provider_for(self, spec: ModelSpec) -> Provider:
        if spec.provider in self._overrides:
            return self._overrides[spec.provider]
        if spec.provider not in self._instances:
            factory = PROVIDERS.get(spec.provider)
            if factory is None:
                raise LLMError(
                    f"unknown provider {spec.provider!r}; "
                    f"known providers: {sorted(PROVIDERS)}"
                )
            self._instances[spec.provider] = factory()
        return self._instances[spec.provider]

    def complete(
        self,
        model: str | ModelSpec,
        *,
        system: str | None = None,
        user: str | None = None,
        messages: Sequence[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache_key: str | None = None,
        tools: Sequence[dict] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        """Run one completion, reading and writing the disk cache.

        Pass ``cache_key`` to control the cache identity explicitly — the
        pipeline uses ``(question_id, step_idx, perturbation_type, model)`` so
        a rerun with a different prompt template still reuses prior answers
        for the same experimental cell.
        """
        spec = model if isinstance(model, ModelSpec) else resolve_model(model)
        if messages is None:
            messages = _build_messages(system, user)
        temperature = spec.temperature if temperature is None else temperature
        max_tokens = spec.max_tokens if max_tokens is None else max_tokens

        key = cache_key or make_cache_key(
            model=spec.api_id,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt=hashlib.sha256(
                json.dumps(list(messages), sort_keys=True).encode("utf-8")
            ).hexdigest()[:24],
        )
        hit = self.cache.get(key)
        if hit is not None:
            return LLMResponse.from_dict(hit, cached=True)

        provider = self.provider_for(spec)
        response = provider.complete(
            messages,
            spec,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )
        self.calls += 1
        # Tool-calling turns are conversation state, not reusable answers.
        if not tools:
            self.cache.put(key, response.to_dict())
        return response

    @property
    def stats(self) -> dict[str, int]:
        return {"api_calls": self.calls, **self.cache.stats}


def _build_messages(system: str | None, user: str | None) -> list[dict]:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user or ""})
    return messages


def load_dotenv(path: Path | str | None = None) -> None:
    """Read a ``.env`` file into ``os.environ`` without adding a dependency.

    Existing environment variables win, so an explicit export always beats the
    file.
    """
    env_path = Path(path) if path else Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        os.environ.setdefault(key, value)
