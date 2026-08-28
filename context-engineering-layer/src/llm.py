"""The only module that spends money, and the only one that talks to Groq.

Inputs:  a rendered message payload plus a cache key
Outputs: a Completion carrying text, finish reason, usage and rate-limit headers

Three rules this file exists to enforce.

`complete()` NEVER RAISES. An oversized request, a 429, a timeout and a
context-length error all come back as a Completion with a finish_reason. If it
raised, the harness would need try/except at every call site and the unlayered
arm would score as a 0% recall failure instead of what it actually is: a
request that structurally cannot run.

Oversized is detected BEFORE dispatch, from the local token count. Sending a
request larger than the per-minute bucket cannot succeed at any time, so
retrying it only burns the daily request allowance.

The SDK's own retry loop is off. It retries without telling us tokens were
spent, which desyncs the local ledger exactly when the ledger matters.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from groq import AsyncGroq

from src import cache, config
from src.types import Usage


@dataclass(frozen=True, slots=True)
class Completion:
    """One model response, or one named way of not getting a model response."""

    text: str = ""
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    error: str = ""
    replayed: bool = False
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the model actually produced a complete answer."""
        return self.finish_reason == "stop" and not self.error


_client: AsyncGroq | None = None
_semaphore: asyncio.Semaphore | None = None


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


def _retry_after(headers: Mapping[str, str], attempt: int) -> float:
    """Honour the server's retry hint, with jitter so workers do not resync."""
    hint = headers.get("retry-after") or headers.get("Retry-After")
    try:
        base = float(hint) if hint else 2.0 * (attempt + 1)
    except ValueError:
        base = 2.0 * (attempt + 1)
    return min(base, 60.0) + random.uniform(0.0, 1.5)


def would_be_oversized(prompt_tokens: int) -> bool:
    """True when prompt + reserved completion cannot fit the per-minute bucket.

    Groq reserves prompt plus max_completion_tokens against the ceiling before
    it runs anything, so this is arithmetic, not a guess.
    """
    return prompt_tokens + config.MAX_COMPLETION_TOKENS > config.TOKENS_PER_MINUTE


async def complete(
    messages: Sequence[Mapping[str, str]],
    *,
    model: str = "",
    cache_key: str = "",
    local_prompt_tokens: int = 0,
    max_completion_tokens: int = 0,
    use_cache: bool = True,
) -> Completion:
    """Call the model once. Returns a Completion for every outcome, good or bad."""
    model = model or config.CHAT_MODEL
    budget = max_completion_tokens or config.MAX_COMPLETION_TOKENS

    if use_cache and cache_key:
        hit = cache.get(cache_key)
        if hit is not None:
            return Completion(
                text=hit.get("text", ""),
                finish_reason=hit.get("finish_reason", "stop"),
                usage=Usage(**hit.get("usage", {})),
                error=hit.get("error", ""),
                replayed=True,
            )

    if local_prompt_tokens and would_be_oversized(local_prompt_tokens):
        return Completion(
            finish_reason="oversized",
            error=(
                f"{local_prompt_tokens:,} prompt + {budget} completion exceeds the "
                f"{config.TOKENS_PER_MINUTE:,} TPM bucket - rejected, not queued"
            ),
            usage=Usage(prompt_tokens=local_prompt_tokens),
        )

    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": config.TEMPERATURE,
        "max_completion_tokens": budget,
        "reasoning_effort": config.REASONING_EFFORT,
    }

    last_error = ""
    headers: Mapping[str, str] = {}
    for attempt in range(config.MAX_RETRIES + 1):
        async with _gate():
            started = time.perf_counter()
            try:
                raw = await client().chat.completions.with_raw_response.create(**payload)
                headers = dict(raw.headers)
                parsed = await raw.parse()
                latency = time.perf_counter() - started
            except Exception as exc:  # noqa: BLE001 - every failure is a result here
                latency = time.perf_counter() - started
                last_error = f"{type(exc).__name__}: {exc}"
                status = getattr(exc, "status_code", None)
                headers = dict(getattr(getattr(exc, "response", None), "headers", {}) or {})
                if status == 429 and attempt < config.MAX_RETRIES:
                    await asyncio.sleep(_retry_after(headers, attempt))
                    continue
                reason = "rate_limited" if status == 429 else "api_error"
                cmp = Completion(finish_reason=reason, error=last_error, headers=headers)
                if reason == "api_error" and status is not None and status < 500:
                    return cmp  # a 4xx that is not a 429 will fail identically forever
                if attempt >= config.MAX_RETRIES:
                    return cmp
                await asyncio.sleep(_retry_after(headers, attempt))
                continue

        choice = parsed.choices[0]
        # message.content only. On Groq, gpt-oss keeps its reasoning in a
        # separate field, and grading a reasoning trace that enumerates
        # candidate answers would score every arm several points too high.
        text = (choice.message.content or "").strip()
        usage = _usage_from(getattr(parsed, "usage", None), latency)
        cache.record_spend(model, usage.uncached_tokens + usage.completion_tokens)

        result = Completion(
            text=text,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            headers=headers,
        )
        if use_cache and cache_key:
            cache.put(cache_key, {
                "text": result.text,
                "finish_reason": result.finish_reason,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "cached_tokens": usage.cached_tokens,
                    "latency_s": usage.latency_s,
                },
            })
        await asyncio.sleep(config.REQUEST_SPACING_S)
        return result

    return Completion(finish_reason="rate_limited", error=last_error, headers=headers)


if __name__ == "__main__":

    async def _smoke() -> None:
        assert would_be_oversized(14_000), "a 14k prompt must be caught before dispatch"
        assert not would_be_oversized(3_000)

        big = await complete(
            [{"role": "user", "content": "hi"}], local_prompt_tokens=14_000, use_cache=False
        )
        assert big.finish_reason == "oversized" and not big.ok
        print(f"oversized guard: {big.error}")

        key = cache.key(config.CHAT_MODEL, {"smoke": "v1"})
        first = await complete(
            [{"role": "user", "content": "Reply with exactly: ok"}], cache_key=key
        )
        print(f"live call  -> {first.finish_reason} {first.text[:30]!r} "
              f"prompt={first.usage.prompt_tokens} "
              f"completion={first.usage.completion_tokens} "
              f"cached={first.usage.cached_tokens} "
              f"{first.usage.latency_s:.2f}s")
        second = await complete(
            [{"role": "user", "content": "Reply with exactly: ok"}], cache_key=key
        )
        assert second.replayed, "the second identical call must replay from sqlite"
        assert second.usage.cached_tokens is None, "a replay reports n/a"
        print(f"replay     -> {second.finish_reason} {second.text[:30]!r} (0 tokens)")

        tok, req = cache.spent_today()
        print(f"ledger today: {tok:,} tok / {req} requests")
        print(f"tpm remaining header: {first.headers.get('x-ratelimit-remaining-tokens')}")

    asyncio.run(_smoke())
