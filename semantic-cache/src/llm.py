"""One call to Gemini, with the tokens, cost and time it took.

Inputs: a customer question and the policy text to answer from.
Outputs: an Answer with the reply text, token counts, dollar cost and latency.
"""

import asyncio
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors, types

from src.config import (
    GEMINI_API_KEY, MIN_INTERVAL_S, MODEL, PRICE_IN_PER_M, PRICE_OUT_PER_M,
)

_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

INSTRUCTIONS = (
    "You are the support agent for Kumkum Home. Answer the customer in one or two "
    "short sentences, using only the policy below.\n\n"
)


@dataclass
class Answer:
    """A reply from the model and what it cost."""

    text: str
    in_tokens: int
    out_tokens: int
    cost: float
    latency_ms: float


def price(in_tokens: int, out_tokens: int) -> float:
    """Return the paid-tier dollar cost of one call."""
    return (in_tokens * PRICE_IN_PER_M + out_tokens * PRICE_OUT_PER_M) / 1_000_000


_last_call = 0.0


async def _pace() -> None:
    """Wait so calls start at least MIN_INTERVAL_S apart (free tier: 15 per minute)."""
    global _last_call
    wait = _last_call + MIN_INTERVAL_S - time.monotonic()
    if wait > 0:
        await asyncio.sleep(wait)
    _last_call = time.monotonic()


async def _generate(question: str, policy: str) -> tuple[types.GenerateContentResponse, float]:
    """Send one request; retry on 429/503. Returns the response and its latency in ms.

    Only the successful attempt is timed, so waiting on the rate limit never
    counts as model latency.
    """
    if _client is None:
        raise ValueError("GEMINI_API_KEY is not set - add it to .env")
    config = types.GenerateContentConfig(
        system_instruction=INSTRUCTIONS + policy,
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
    )
    for attempt in range(4):
        await _pace()
        start = time.perf_counter()
        try:
            response = await _client.aio.models.generate_content(
                model=MODEL, contents=question, config=config
            )
            return response, (time.perf_counter() - start) * 1000
        except errors.APIError as err:
            if err.code not in (429, 503) or attempt == 3:
                raise
            await asyncio.sleep(15)


async def answer(question: str, policy: str) -> Answer:
    """Ask the model one question and measure tokens, cost and latency."""
    response, latency_ms = await _generate(question, policy)

    usage = response.usage_metadata
    in_tokens = usage.prompt_token_count or 0
    # Thinking tokens are billed at the output price, so count them as output.
    out_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    return Answer(
        text=(response.text or "").strip(),
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        cost=price(in_tokens, out_tokens),
        latency_ms=latency_ms,
    )
