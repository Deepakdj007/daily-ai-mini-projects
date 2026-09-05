"""The only module that spends money, and the only one that talks to Groq.

Inputs:  messages, an optional strict JSON schema, and a cache key
Outputs: a Completion carrying text, finish reason, usage and rate-limit headers

Three rules this file exists to enforce.

`complete()` never raises. A 429, a timeout, an oversized request and a schema
validation failure all come back as a Completion with a named finish_reason. If
it raised, every call site would need a try/except and a sweep that hit its
rate limit would look like a sweep that found nothing.

The SDK's retry loop is off. It retries without telling us tokens were spent,
which desyncs the local ledger exactly when the ledger matters.

Only message.content is read. On Groq, gpt-oss keeps its reasoning in a
separate field; grading a reasoning trace that enumerates candidate answers
would score every arm several points too high.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from groq import AsyncGroq

from src import cache, config

# (temperature, strict). A retry at temperature 0 redraws the identical sample,
# so each attempt must change the conditions. The last drops strict validation
# and keeps a best-effort answer rather than losing the call entirely.
_ATTEMPTS: tuple[tuple[float, bool], ...] = ((0.0, True), (0.4, True), (0.4, False))


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts for one call. A replay reports zeros and cached_tokens None."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int | None = None
    latency_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "latency_s": round(self.latency_s, 3),
        }


@dataclass(frozen=True, slots=True)
class Completion:
    """One model response, or one named way of not getting a model response."""

    text: str = ""
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    error: str = ""
    replayed: bool = False
    headers: Mapping[str, str] = field(default_factory=dict)
    reasoning_present: bool = False

    @property
    def ok(self) -> bool:
        """True when the model actually produced a complete answer."""
        return self.finish_reason == "stop" and not self.error and bool(self.text)


_client: AsyncGroq | None = None
_semaphore: asyncio.Semaphore | None = None
_calls = {"live": 0, "replayed": 0, "failed": 0}


def client() -> AsyncGroq:
    """The shared Groq client, with SDK retries disabled."""
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=config.require_api_key(), max_retries=0)
    return _client


def _gate() -> asyncio.Semaphore:
    """Concurrency gate, created lazily so it binds to the running loop."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
    return _semaphore


def stats() -> dict[str, int]:
    """Call counts, so a test can assert an arm made zero live calls."""
    return dict(_calls)


def reset_stats() -> None:
    """Zero the counters between eval arms."""
    for name in _calls:
        _calls[name] = 0


def _retry_after(headers: Mapping[str, str], attempt: int) -> float:
    """Honour the server's hint, with jitter so parallel workers do not resync."""
    hint = headers.get("retry-after") or headers.get("Retry-After")
    try:
        base = float(hint) if hint else 2.0 * (attempt + 1)
    except ValueError:
        base = 2.0 * (attempt + 1)
    return min(base, 60.0) + random.uniform(0.0, 1.5)


def _usage_from(raw: Any, latency: float) -> Usage:
    """Pull token counts off a response, tolerating a missing details block."""
    if raw is None:
        return Usage(latency_s=latency)
    details = getattr(raw, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details else None
    return Usage(
        prompt_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
        cached_tokens=int(cached) if cached is not None else None,
        latency_s=latency,
    )


async def complete(
    messages: Sequence[Mapping[str, str]],
    *,
    model: str = "",
    cache_key: str = "",
    schema: dict | None = None,
    schema_name: str = "response",
    max_completion_tokens: int = 0,
    use_cache: bool = True,
) -> Completion:
    """Call the model once. Returns a Completion for every outcome, good or bad.

    When `schema` is given it is sent as strict json_schema constrained
    decoding, and the attempt ladder above handles the model returning prose.
    """
    model = model or config.CHAT_MODEL
    budget = max_completion_tokens or config.MAX_COMPLETION_TOKENS["answer"]

    if use_cache and cache_key:
        hit = cache.get(cache_key)
        if hit is not None:
            _calls["replayed"] += 1
            return Completion(
                text=hit.get("text", ""),
                finish_reason=hit.get("finish_reason", "stop"),
                usage=Usage(**(hit.get("usage") or {})),
                error=hit.get("error", ""),
                replayed=True,
                reasoning_present=hit.get("reasoning_present", False),
            )

    last_error = ""
    headers: Mapping[str, str] = {}
    for attempt in range(config.MAX_RETRIES + 1):
        temperature, strict = _ATTEMPTS[min(attempt, len(_ATTEMPTS) - 1)]
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature if schema else config.TEMPERATURE,
            "max_completion_tokens": budget,
            "reasoning_effort": config.REASONING_EFFORT,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": strict, "schema": schema},
            }

        async with _gate():
            started = time.perf_counter()
            try:
                raw = await client().chat.completions.with_raw_response.create(**payload)
                headers = dict(raw.headers)
                parsed = await raw.parse()
                latency = time.perf_counter() - started
            except Exception as exc:  # noqa: BLE001 - every failure is a result here
                last_error = f"{type(exc).__name__}: {exc}"
                headers = dict(getattr(getattr(exc, "response", None), "headers", {}) or {})
                status = getattr(exc, "status_code", None)
                retryable = status == 429 or "json_validate_failed" in last_error
                if retryable and attempt < config.MAX_RETRIES:
                    if status == 429:
                        await asyncio.sleep(_retry_after(headers, attempt))
                    continue
                _calls["failed"] += 1
                reason = "rate_limited" if status == 429 else "api_error"
                return Completion(finish_reason=reason, error=last_error, headers=headers)

        choice = parsed.choices[0]
        text = (choice.message.content or "").strip()
        usage = _usage_from(getattr(parsed, "usage", None), latency)
        cache.record_spend(model, usage.prompt_tokens + usage.completion_tokens)
        _calls["live"] += 1

        # An empty body with a 'length' finish means reasoning ate the budget.
        # Retrying at the same budget cannot help, so say so plainly.
        if not text and choice.finish_reason == "length":
            return Completion(
                finish_reason="length",
                usage=usage,
                headers=headers,
                reasoning_present=True,
                error=f"empty content at max_completion_tokens={budget}; raise the budget",
            )

        result = Completion(
            text=text,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            headers=headers,
            reasoning_present=bool(getattr(choice.message, "reasoning", None)),
        )
        if use_cache and cache_key and result.ok:
            cache.put(cache_key, {
                "text": result.text,
                "finish_reason": result.finish_reason,
                "usage": usage.as_dict(),
                "reasoning_present": result.reasoning_present,
            })
        await asyncio.sleep(config.REQUEST_SPACING_S)
        return result

    _calls["failed"] += 1
    return Completion(finish_reason="rate_limited", error=last_error, headers=headers)


async def complete_json(
    messages: Sequence[Mapping[str, str]],
    *,
    schema: dict,
    schema_name: str,
    model: str = "",
    cache_key: str = "",
    max_completion_tokens: int = 0,
) -> tuple[dict | None, Completion]:
    """Strict-schema call that returns parsed JSON, or (None, why-not)."""
    completion = await complete(
        messages,
        model=model,
        cache_key=cache_key,
        schema=schema,
        schema_name=schema_name,
        max_completion_tokens=max_completion_tokens,
    )
    if not completion.ok:
        return None, completion
    try:
        return json.loads(completion.text), completion
    except json.JSONDecodeError as exc:
        return None, Completion(
            text=completion.text,
            finish_reason="api_error",
            usage=completion.usage,
            error=f"unparseable JSON: {exc}",
            replayed=completion.replayed,
        )


if __name__ == "__main__":

    async def _smoke() -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["city", "confident"],
            "properties": {"city": {"type": "string"}, "confident": {"type": "boolean"}},
        }
        k = cache.key("__smoke_llm", {"v": config.PROMPT_VERSION})
        data, comp = await complete_json(
            [{"role": "user", "content": "The user lives in Pune. Return city and whether you are confident."}],
            schema=schema, schema_name="probe", model=config.EXTRACT_MODEL, cache_key=k,
            max_completion_tokens=config.MAX_COMPLETION_TOKENS["classify"],
        )
        assert data is not None, f"strict schema call failed: {comp.error}"
        assert "pune" in str(data.get("city", "")).lower(), data
        print(f"live   -> {data} prompt={comp.usage.prompt_tokens} completion={comp.usage.completion_tokens}")
        print(f"        tpm remaining header: {comp.headers.get('x-ratelimit-remaining-tokens')}")

        again, comp2 = await complete_json(
            [{"role": "user", "content": "The user lives in Pune. Return city and whether you are confident."}],
            schema=schema, schema_name="probe", model=config.EXTRACT_MODEL, cache_key=k,
        )
        assert comp2.replayed and again == data, "the second identical call must replay"
        print(f"replay -> {again} (0 tokens)")

        broken = await complete(
            [{"role": "user", "content": "hi"}], model="openai/does-not-exist", use_cache=False
        )
        assert not broken.ok and broken.finish_reason == "api_error"
        print(f"failure-> {broken.finish_reason}: {broken.error[:70]}")
        print(f"calls   {stats()}")

    asyncio.run(_smoke())
