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

Tool calls come back flattened to plain dicts rather than SDK objects. The
agent loop has to put them back into the message list on the next turn, cache
them, and write them to a results file, and an SDK object does none of those
three without a conversion somewhere.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from groq import AsyncGroq

from src import cache, config

# (temperature, strict). A retry at temperature 0 redraws the identical sample,
# so a SCHEMA failure must change the conditions to have any chance. The last
# rung drops strict validation and keeps a best-effort answer rather than
# losing the call entirely.
#
# This ladder is climbed only for schema failures. A 429 is a statement about
# the rate limit and says nothing about the sample, so retrying one at a
# different temperature moves the run off temperature 0 for a reason that has
# nothing to do with the model. See FINDINGS.md.
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
    attempt: int = 0
    tool_calls: tuple[dict, ...] = ()
    """Flattened to {"id", "name", "arguments"} with arguments left as the raw
    JSON string the model emitted. Parsing it here would throw away the
    malformed case, and a model that emits broken arguments is a thing the
    agent loop must be able to hand back as tool text rather than crash on."""
    """Which rung of the SCHEMA ladder produced this, not how many times the
    call was sent. Anything above 0 ran at a non-zero temperature, and a
    validity gate counts how many scored rows came from one - an eval that
    quietly drifts off temperature 0 is not repeatable. Rate-limit retries do
    not count, because they re-send the identical payload."""

    @property
    def ok(self) -> bool:
        """True when the model actually produced a complete turn.

        A turn that is nothing but tool calls has empty content and a
        finish_reason of "tool_calls". Judging it by text alone would read
        every tool call in the run as a failed call.
        """
        if self.error:
            return False
        if self.finish_reason == "tool_calls":
            return bool(self.tool_calls)
        return self.finish_reason == "stop" and bool(self.text or self.tool_calls)


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


_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
# "ms" has to be tried before "m", or 650ms reads as 650 minutes.
_DURATION = re.compile(r"([\d.]+)\s*(ms|s|m|h)")


def _duration(value: str | None) -> float:
    """Parse Groq's reset hints: '7.66s', '1m30s', '650ms', or a bare number."""
    if not value:
        return 0.0
    parts = _DURATION.findall(str(value))
    if parts:
        return sum(float(amount) * _UNITS[unit] for amount, unit in parts)
    try:
        return float(value)
    except ValueError:
        return 0.0


def _pace(headers: Mapping[str, str], budget: int) -> float:
    """How long to wait before the next call, from the limit Groq just reported.

    The per-minute bucket is charged max_completion_tokens at request time, not
    the tokens actually produced, so a run that paces on what it SPENT will
    still walk into 429s. Groq publishes what is left and when it resets, and
    waiting for the reset only when the next request would not fit is both
    exact and faster than a fixed sleep long enough to always be safe.
    """
    remaining = headers.get("x-ratelimit-remaining-tokens")
    try:
        left = int(remaining) if remaining is not None else None
    except ValueError:
        left = None
    if left is not None and left < budget:
        reset = _duration(headers.get("x-ratelimit-reset-tokens"))
        return max(config.REQUEST_SPACING_S, min(reset + 1.0, 65.0))
    if left is None:
        # No header to go on: fall back to spacing that keeps the reservation
        # rate under the published ceiling.
        return max(config.REQUEST_SPACING_S,
                   60.0 * budget / max(config.TOKENS_PER_MINUTE, 1))
    return config.REQUEST_SPACING_S


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


def _tool_calls_from(message: Any) -> tuple[dict, ...]:
    """Flatten the SDK's tool_calls into plain dicts, arguments left as text."""
    raw = getattr(message, "tool_calls", None) or []
    return tuple(
        {
            "id": call.id,
            "name": call.function.name,
            "arguments": call.function.arguments or "{}",
        }
        for call in raw
    )


def assistant_message(completion: Completion) -> dict:
    """The assistant turn to append to the message list, tool calls included.

    The tool replies that follow have nothing to attach to unless the calls are
    echoed back verbatim with their ids, and Groq rejects a tool message whose
    tool_call_id it has never seen.
    """
    message: dict[str, Any] = {"role": "assistant", "content": completion.text or ""}
    if completion.tool_calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for call in completion.tool_calls
        ]
    return message


async def complete(
    messages: Sequence[Mapping[str, str]],
    *,
    model: str = "",
    cache_key: str = "",
    schema: dict | None = None,
    schema_name: str = "response",
    tools: Sequence[dict] | None = None,
    max_completion_tokens: int = 0,
    use_cache: bool = True,
) -> Completion:
    """Call the model once. Returns a Completion for every outcome, good or bad.

    When `schema` is given it is sent as strict json_schema constrained
    decoding, and the attempt ladder above handles the model returning prose.
    When `tools` is given the reply may carry tool_calls instead of text.
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
                attempt=int(hit.get("attempt", 0)),
                tool_calls=tuple(hit.get("tool_calls") or ()),
            )

    last_error = ""
    headers: Mapping[str, str] = {}
    schema_attempt = 0   # only a schema failure climbs the ladder
    for attempt in range(config.MAX_RETRIES + 1):
        temperature, strict = _ATTEMPTS[min(schema_attempt, len(_ATTEMPTS) - 1)]
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            # The ladder's temperature applies to tool calls as well as to
            # schemas, because both fail the same way: the model emitted
            # something malformed, and the identical sample would be redrawn.
            # Every row that came from a rung above the first is counted, and a
            # validity gate voids a run that drifted off temperature 0 too often.
            "temperature": temperature if (schema or tools) else config.TEMPERATURE,
            "max_completion_tokens": budget,
        }
        # reasoning_effort is a gpt-oss parameter. The guard classifier is not
        # a reasoning model and rejects it.
        if model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = config.REASONING_EFFORT
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": strict, "schema": schema},
            }
        if tools:
            payload["tools"] = list(tools)

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
                # A 400 saying the tool call would not parse is a sampling
                # failure, not a request that can never work: the model emitted
                # something that is not JSON. Retrying at the identical
                # temperature would redraw the identical sample, so it climbs
                # the same ladder a schema failure does.
                # Groq reports this failure under at least four different
                # messages - "Failed to parse tool call arguments as JSON",
                # "Tool call validation failed", "Parsing failed. The model
                # generated output that could not be parsed", and
                # json_validate_failed - so the error CODE is what gets matched
                # and the wordings are only a backstop.
                schema_failure = any(mark in last_error.lower() for mark in (
                    "json_validate_failed", "tool_use_failed", "parsing failed",
                    "tool call arguments", "tool call validation"))
                # A tokens-per-day 429 will not clear for hours, so retrying it
                # just burns the ladder and turns a budget stop into a row of
                # failures. It is also the ONLY place Groq exposes the real TPD
                # counter - there is no remaining-TPD header - so the message
                # gets parsed rather than swallowed.
                exhausted = status == 429 and "tokens per day" in last_error.lower()
                if exhausted:
                    _calls["failed"] += 1
                    return Completion(finish_reason="tpd_exhausted", error=last_error,
                                      headers=headers)
                retryable = status == 429 or schema_failure
                if retryable and attempt < config.MAX_RETRIES:
                    if status == 429:
                        # Same payload, same temperature. Wait and ask again.
                        await asyncio.sleep(_retry_after(headers, attempt))
                    else:
                        schema_attempt += 1
                    continue
                _calls["failed"] += 1
                reason = "rate_limited" if status == 429 else "api_error"
                return Completion(finish_reason=reason, error=last_error, headers=headers)

        choice = parsed.choices[0]
        text = (choice.message.content or "").strip()
        calls = _tool_calls_from(choice.message)
        usage = _usage_from(getattr(parsed, "usage", None), latency)
        cache.record_spend(model, usage.prompt_tokens + usage.completion_tokens)
        _calls["live"] += 1

        # A 'stop' finish with no content and no tool calls is a degenerate
        # sample: the model produced a reasoning trace and then nothing. It is
        # not an answer and not an error anyone can act on, and at temperature 0
        # the identical draw would repeat, so it climbs the ladder like any
        # other malformed generation. Measured on gpt-oss mid-conversation,
        # after a tool result.
        if not text and not calls and choice.finish_reason == "stop":
            if schema_attempt < len(_ATTEMPTS) - 1 and attempt < config.MAX_RETRIES:
                schema_attempt += 1
                continue
            return Completion(finish_reason="empty", usage=usage, headers=headers,
                              reasoning_present=True, attempt=schema_attempt,
                              error="the model returned an empty message")

        # An empty body with a 'length' finish means reasoning ate the budget.
        # Retrying at the same budget cannot help, so say so plainly.
        if not text and not calls and choice.finish_reason == "length":
            return Completion(
                finish_reason="length",
                usage=usage,
                headers=headers,
                reasoning_present=True,
                attempt=schema_attempt,
                error=f"empty content at max_completion_tokens={budget}; raise the budget",
            )

        result = Completion(
            text=text,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            headers=headers,
            reasoning_present=bool(getattr(choice.message, "reasoning", None)),
            attempt=schema_attempt,
            tool_calls=calls,
        )
        if use_cache and cache_key and result.ok:
            cache.put(cache_key, {
                "text": result.text,
                "finish_reason": result.finish_reason,
                "usage": usage.as_dict(),
                "reasoning_present": result.reasoning_present,
                "attempt": result.attempt,
                "tool_calls": list(result.tool_calls),
            })
        await asyncio.sleep(_pace(headers, budget))
        return result

    _calls["failed"] += 1
    return Completion(finish_reason="rate_limited", error=last_error, headers=headers)


def tpd_used(error: str) -> tuple[int, int]:
    """(used, limit) parsed out of a tokens-per-day 429 body, or (0, 0).

    Groq publishes six x-ratelimit-* headers and none of them is TPD. This
    error string is the only exposed view of the counter the daily budget
    actually runs out against.
    """
    match = re.search(r"tokens per day \(TPD\): Limit (\d+), Used (\d+)", error)
    return (int(match.group(2)), int(match.group(1))) if match else (0, 0)


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

    assert _duration("7.66s") == 7.66
    assert _duration("1m30s") == 90.0
    assert abs(_duration("650ms") - 0.65) < 1e-9, _duration("650ms")
    assert _duration("2") == 2.0 and _duration(None) == 0.0 and _duration("") == 0.0
    assert _pace({"x-ratelimit-remaining-tokens": "50",
                  "x-ratelimit-reset-tokens": "7.66s"}, 1600) > 7.0
    assert _pace({"x-ratelimit-remaining-tokens": "7000"}, 1600) == config.REQUEST_SPACING_S
    assert _pace({}, 1600) >= 60.0 * 1600 / config.TOKENS_PER_MINUTE

    async def _smoke() -> None:
        """Prove four things: strict JSON works, tool calls come back, the
        cache replays, and a bad model id is a result rather than a crash."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {"answer": {"type": ["string", "null"]}},
        }
        prompt = "The monthly wallet cap is Rs 47,300. What is the monthly wallet cap?"
        k = cache.key("__smoke_llm", {"v": config.PROMPT_VERSION, "p": prompt})
        data, comp = await complete_json(
            [{"role": "user", "content": prompt}],
            schema=schema, schema_name="claim", cache_key=k,
            max_completion_tokens=config.MAX_COMPLETION_TOKENS["answer"],
        )
        assert data is not None, f"strict schema call failed: {comp.error}"
        assert "47,300" in str(data.get("answer")) or "47300" in str(data.get("answer")), data
        print(f"json     -> {data}")
        print(f"            prompt={comp.usage.prompt_tokens} "
              f"completion={comp.usage.completion_tokens} attempt={comp.attempt}")

        again, comp2 = await complete_json(
            [{"role": "user", "content": prompt}],
            schema=schema, schema_name="claim", cache_key=k,
        )
        assert comp2.replayed and again == data, "the second identical call must replay"
        print(f"replay   -> {again} (0 tokens)")

        tool = {
            "type": "function",
            "function": {
                "name": "square",
                "description": "Return the square of an integer.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["n"],
                    "properties": {"n": {"type": "integer"}},
                },
            },
        }
        tooled = await complete(
            [{"role": "user", "content": "What is 4021 squared? Use the tool."}],
            tools=[tool],
            cache_key=cache.key("__smoke_llm", {"v": config.PROMPT_VERSION, "t": "square"}),
            max_completion_tokens=config.MAX_COMPLETION_TOKENS["answer"],
        )
        assert tooled.ok, f"tool call failed: {tooled.error}"
        assert tooled.tool_calls, f"expected a tool call, got text: {tooled.text!r}"
        call = tooled.tool_calls[0]
        assert call["name"] == "square", call
        assert json.loads(call["arguments"])["n"] == 4021, call
        print(f"tools    -> {call['name']}({call['arguments']}) "
              f"finish={tooled.finish_reason}")

        echo = assistant_message(tooled)
        assert echo["tool_calls"][0]["id"] == call["id"], echo
        assert echo["tool_calls"][0]["function"]["name"] == "square", echo
        print(f"echo     -> {len(echo['tool_calls'])} call(s) round-trip with their ids")

        broken = await complete(
            [{"role": "user", "content": "hi"}], model="openai/does-not-exist", use_cache=False
        )
        assert not broken.ok and broken.finish_reason == "api_error"
        print(f"failure  -> {broken.finish_reason}: {broken.error[:70]}")
        print(f"calls     {stats()}")

    asyncio.run(_smoke())
