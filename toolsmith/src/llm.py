"""The only module that talks to Groq.

Inputs:  messages and a tools array
Outputs: a Completion carrying text, tool calls, and how the call ended

`complete()` never raises. A 429, a malformed tool call and a dead model all
come back as a Completion with a named finish_reason, because a caller that
has to wrap every call in try/except ends up swallowing the interesting cases.

Two failures here are worth knowing about before you meet them. Groq rejects a
tool call the model botched with a hard 400 under four different wordings, and
retrying at temperature 0 redraws the identical sample - so a retry has to
change the temperature to have any chance. And gpt-oss will sometimes return a
completely empty message mid-conversation, which is not an answer and not an
error anyone can act on, so it climbs the same ladder.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from groq import AsyncGroq

from src import config

# (temperature, ...) for successive attempts at a malformed generation.
_TEMPERATURES = (0.0, 0.4, 0.7)
_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
_DURATION = re.compile(r"([\d.]+)\s*(ms|s|m|h)")   # "ms" before "m", or 650ms is 650 minutes

# Groq reports a botched tool call under all of these.
_MALFORMED = ("json_validate_failed", "tool_use_failed", "parsing failed",
              "tool call arguments", "tool call validation")


@dataclass(frozen=True, slots=True)
class Completion:
    """One model response, or one named way of not getting one."""

    text: str = ""
    finish_reason: str = "stop"
    tool_calls: tuple[dict, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """A turn that is nothing but tool calls has no text and is still fine."""
        return not self.error and bool(self.text or self.tool_calls)


_client: AsyncGroq | None = None


def client() -> AsyncGroq:
    """The shared Groq client, with SDK retries off so we can see every failure."""
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=config.require_api_key(), max_retries=0)
    return _client


def _duration(value: str | None) -> float:
    """Parse Groq's reset hints: '7.66s', '1m30s', '650ms'."""
    parts = _DURATION.findall(str(value or ""))
    return sum(float(amount) * _UNITS[unit] for amount, unit in parts)


def _pace(headers: Mapping[str, str], prompt_tokens: int) -> float:
    """Wait only when the NEXT request would not fit the per-minute bucket.

    Two things make this less obvious than it looks. The bucket is charged
    max_completion_tokens when the request arrives, not what the answer
    actually costs - so pacing on tokens spent walks straight into 429s. And it
    is charged the prompt as well, which in an agent loop grows every round as
    tool results pile up. So the estimate for the next call is this call's
    prompt plus the reservation, and the last round of a long task can need
    five times what the first one did.
    """
    need = prompt_tokens + config.MAX_COMPLETION_TOKENS
    try:
        left = int(headers.get("x-ratelimit-remaining-tokens", ""))
    except ValueError:
        return max(config.REQUEST_SPACING_S, 60.0 * need / config.TOKENS_PER_MINUTE)
    if left < need:
        return max(config.REQUEST_SPACING_S,
                   min(_duration(headers.get("x-ratelimit-reset-tokens")) + 1.0, 65.0))
    return config.REQUEST_SPACING_S


def assistant_message(completion: Completion) -> dict:
    """The assistant turn to append, tool calls echoed back with their ids.

    Groq rejects a tool reply whose tool_call_id it has never seen, so the
    calls have to go back verbatim before their results can follow.
    """
    message: dict[str, Any] = {"role": "assistant", "content": completion.text or ""}
    if completion.tool_calls:
        message["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in completion.tool_calls]
    return message


async def complete(messages: Sequence[Mapping[str, Any]],
                   tools: Sequence[dict]) -> Completion:
    """Call the model once. Returns a Completion for every outcome, good or bad."""
    error, headers = "", {}
    for attempt in range(config.MAX_RETRIES + 1):
        payload = {
            "model": config.CHAT_MODEL,
            "messages": list(messages),
            "tools": list(tools),
            "temperature": _TEMPERATURES[min(attempt, len(_TEMPERATURES) - 1)],
            "max_completion_tokens": config.MAX_COMPLETION_TOKENS,
            "reasoning_effort": config.REASONING_EFFORT,
        }
        try:
            raw = await client().chat.completions.with_raw_response.create(**payload)
            headers = dict(raw.headers)
            parsed = await raw.parse()
        except Exception as exc:                # noqa: BLE001 - every failure is a result
            error = f"{type(exc).__name__}: {exc}"
            headers = dict(getattr(getattr(exc, "response", None), "headers", {}) or {})
            status = getattr(exc, "status_code", None)
            malformed = any(mark in error.lower() for mark in _MALFORMED)
            if status == 429 and "tokens per day" in error.lower():
                # A daily 429 will not clear for hours, whatever the message
                # says about trying again in two minutes. Retrying it just
                # burns the ladder and turns a spent budget into a wall of
                # failures with no explanation in them.
                return Completion(finish_reason="out of tokens", headers=headers,
                                  error="the free tier's daily token budget is "
                                        "spent. It resets at 00:00 UTC.")
            if attempt < config.MAX_RETRIES and (status == 429 or malformed):
                # A 429 is about the clock, so wait. A malformed generation is
                # about the sample, so the next attempt changes temperature.
                await asyncio.sleep(_retry_after(headers, attempt) if status == 429 else 0)
                continue
            return Completion(finish_reason="error", error=error, headers=headers)

        choice = parsed.choices[0]
        text = (choice.message.content or "").strip()
        calls = tuple({"id": c.id, "name": c.function.name,
                       "arguments": c.function.arguments or "{}"}
                      for c in (choice.message.tool_calls or []))
        usage = getattr(parsed, "usage", None)
        result = Completion(
            text=text, finish_reason=choice.finish_reason or "stop", tool_calls=calls,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            headers=headers)

        if not text and not calls:
            # A degenerate sample: reasoning, then nothing. Not an answer, and
            # at temperature 0 the identical draw would repeat.
            if attempt < config.MAX_RETRIES:
                continue
            return Completion(finish_reason="empty", headers=headers,
                              error="the model returned an empty message")
        await asyncio.sleep(_pace(headers, result.prompt_tokens))
        return result

    return Completion(finish_reason="error", error=error or "gave up", headers=headers)


def _retry_after(headers: Mapping[str, str], attempt: int) -> float:
    """How long to wait after a 429, with jitter so retries do not resync.

    The fallback climbs in fives rather than twos. A per-minute bucket can need
    the best part of a minute to refill, and a ladder that waits two, four, six
    and eight seconds gives up after twenty - which reads as a hard failure
    when all that was needed was to wait.
    """
    for key in ("retry-after", "x-ratelimit-reset-tokens"):
        if hint := _duration(headers.get(key)):
            return min(hint + 1.0, 65.0) + random.uniform(0.0, 1.5)
    raw = headers.get("retry-after", "")
    try:                                  # a bare number of seconds
        return min(float(raw) + 1.0, 65.0) + random.uniform(0.0, 1.5)
    except ValueError:
        return 5.0 * (attempt + 1) + random.uniform(0.0, 1.5)


if __name__ == "__main__":
    assert _duration("7.66s") == 7.66 and _duration("1m30s") == 90.0
    assert abs(_duration("650ms") - 0.65) < 1e-9
    assert _pace({"x-ratelimit-remaining-tokens": "7000"}, 500) == config.REQUEST_SPACING_S
    # The prompt counts too: 6000 left is not enough for a 5000-token
    # prompt plus the reservation, even though it dwarfs the reservation.
    assert _pace({"x-ratelimit-remaining-tokens": "6000",
                  "x-ratelimit-reset-tokens": "7.66s"}, 5000) > 7.0
    assert _pace({"x-ratelimit-remaining-tokens": "50",
                  "x-ratelimit-reset-tokens": "7.66s"}, 100) > 7.0

    async def _smoke() -> None:
        """One tool call, round-tripped, against the live API."""
        tool = {"type": "function", "function": {
            "name": "square", "description": "Square an integer.",
            "parameters": {"type": "object", "additionalProperties": False,
                           "required": ["n"], "properties": {"n": {"type": "integer"}}}}}
        started = time.perf_counter()
        got = await complete(
            [{"role": "user", "content": "What is 4021 squared? Use the tool."}], [tool])
        assert got.ok, got.error
        assert got.tool_calls and got.tool_calls[0]["name"] == "square", got
        print(f"tool call -> {got.tool_calls[0]['arguments']} "
              f"in {time.perf_counter() - started:.1f}s, "
              f"{got.completion_tokens} completion tokens")
        assert assistant_message(got)["tool_calls"][0]["id"] == got.tool_calls[0]["id"]
        print("OK - llm")

    asyncio.run(_smoke())
