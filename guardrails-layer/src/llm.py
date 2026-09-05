"""Every Groq call the project makes: the assistant, the classifier, the judge.

Inputs:  message lists / raw text
Outputs: assistant text, a Prompt Guard probability, a policy verdict

Three different models sit behind one client. Keeping them in one file makes
the cost asymmetry visible: the classifier answers in milliseconds for a
handful of tokens, the judge thinks first and is rate-limited far harder.
"""

from __future__ import annotations

import asyncio
import random

from groq import AsyncGroq
from groq import APIStatusError

from src import cache
from src.config import CHAT_MODEL, GROQ_API_KEY, GUARD_MODEL, POLICY_MODEL
from src.config import MAX_RETRY_TOKENS, assistant_tokens

_client: AsyncGroq | None = None

MAX_RETRIES = 8


def client() -> AsyncGroq:
    """Return a lazily created AsyncGroq client."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is empty. Copy .env.example to .env.")
        _client = AsyncGroq(api_key=GROQ_API_KEY)
    return _client


def _retry_after(exc: APIStatusError) -> float | None:
    """Seconds Groq asked us to wait, if it said.

    Groq sends a `retry-after` header on a 429. Honouring it beats guessing,
    because the reset window is a token budget refilling, not a fixed penalty.
    """
    try:
        value = exc.response.headers.get("retry-after")
        return float(value) + 1 if value else None
    except (AttributeError, TypeError, ValueError):
        return None


async def _complete(model: str, messages: list[dict], **kwargs) -> str:
    """Call chat.completions with caching and backoff on rate limits."""
    payload = {"messages": messages, **kwargs}
    hit = cache.get(model, payload)
    if hit is not None:
        return hit

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client().chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            break
        except APIStatusError as exc:
            # 429 on the free tier is a per-minute token budget, not a wall.
            # A whole ladder is hundreds of calls, so the backoff has to
            # outlast a full reset window - an earlier version gave up after
            # 36 seconds and killed a 20-minute run near the end of it.
            if exc.status_code != 429 or attempt == MAX_RETRIES - 1:
                raise
            # Jitter matters more than the base delay. Without it, every
            # worker that hit the same limit wakes at the same moment, spends
            # the refilled budget in one burst, and 429s again in lockstep.
            wait = _retry_after(exc) or 10 * (attempt + 1)
            await asyncio.sleep(wait + random.uniform(0, 4))
    else:  # pragma: no cover - the loop always breaks or raises
        raise RuntimeError("unreachable")

    choice = resp.choices[0]
    text = choice.message.content or ""

    # A model that writes its reasoning into `content` can burn the whole
    # budget thinking and return nothing usable. Downstream that looks like a
    # refusal, and it once blocked 11 of 14 legitimate messages before anyone
    # noticed the cause was a token limit rather than an attack.
    if choice.finish_reason == "length" and "max_completion_tokens" in kwargs:
        roomier = dict(kwargs)
        # Capped, because a retry budget above the per-minute token ceiling
        # can never be served - it would 429 on every attempt forever.
        roomier["max_completion_tokens"] = min(
            kwargs["max_completion_tokens"] * 2, MAX_RETRY_TOKENS
        )
        try:
            resp = await client().chat.completions.create(
                model=model, messages=messages, **roomier
            )
            text = resp.choices[0].message.content or text
        except APIStatusError:
            # Keep the truncated answer rather than losing the whole run.
            pass

    cache.put(model, payload, text)
    return text


async def assistant(system: str, user: str, model: str | None = None) -> str:
    """Run the assistant we are defending.

    `model` is a parameter rather than a constant because one of the
    experiments swaps it: the same system prompt defends very differently
    depending on which model is reading it.
    """
    chat_model = model or CHAT_MODEL
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return await _complete(
        chat_model,
        messages,
        # Zero, because this is a measurement harness. At 0.2 the same ladder
        # scored 10 leaks on one run and 14 on the next from the same 20
        # attacks, which is enough noise to swamp the gap between two rungs.
        #
        # It narrows the variance rather than removing it: the same request at
        # temperature 0 still leaked in 1 of 6 samples. Hosted inference is
        # not bit-identical, so one ladder run is one draw, not the answer.
        temperature=0.0,
        max_completion_tokens=assistant_tokens(chat_model),
    )


async def guard_score(text: str) -> float:
    """Score one chunk of text with Prompt Guard 2.

    The model replies with a probability rendered as a string - "0.9995..." -
    in message.content. There is no label field and no logprobs to read.
    """
    raw = await _complete(GUARD_MODEL, [{"role": "user", "content": text}])
    try:
        return float(raw.strip())
    except ValueError:
        # A non-numeric reply means the shape changed; treat it as suspicious
        # rather than silently scoring it safe.
        return 1.0


async def policy_verdict(policy: str, text: str) -> tuple[str, str]:
    """Judge text against a written policy. Returns (verdict, reasoning).

    gpt-oss-safeguard thinks before it answers, so max_completion_tokens has
    to cover the reasoning as well as the one word we want back. Too low and
    content comes back empty.
    """
    messages = [
        {"role": "system", "content": policy},
        {"role": "user", "content": text},
    ]
    raw = await _complete(
        POLICY_MODEL, messages, temperature=0.0, max_completion_tokens=1600
    )
    word = raw.strip().upper()
    if "BLOCK" in word:
        return "BLOCK", raw
    if "ALLOW" in word:
        return "ALLOW", raw
    # Empty or truncated output fails closed.
    return "BLOCK", raw or "(empty response)"
