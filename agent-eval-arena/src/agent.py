"""The agent under test: a tool-calling loop over Groq, traced via Langfuse.

Inputs: an AsyncOpenAI client (pointed at Groq), an AgentConfig, and a question string.
Outputs: a dict with the final answer, accumulated token usage, cost, and wall-clock latency.
Only the model changes between configs — the loop itself is identical for every run.
"""

import asyncio
import json
import os
import time
from typing import Any

from openai import BadRequestError, RateLimitError

from langfuse.openai import AsyncOpenAI

from src.config import GROQ_BASE_URL, AgentConfig, calculate_cost_usd
from src.tools import CALCULATOR_TOOL_SCHEMA, run_calculator

SYSTEM_PROMPT = (
    "You are a precise assistant. Use the calculator tool for any arithmetic. "
    "Reply with only the final answer, as short as possible, no explanation."
)

MAX_TOOL_ROUNDS = 5
MAX_RATE_LIMIT_RETRIES = 5


def build_groq_client() -> AsyncOpenAI:
    """Create a Langfuse-traced AsyncOpenAI client pointed at Groq's OpenAI-compatible endpoint."""
    return AsyncOpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL)


async def _execute_tool_calls(tool_calls: list[Any]) -> list[dict[str, str]]:
    """Run every requested calculator call and format the results as tool messages."""
    tool_messages = []
    for tool_call in tool_calls:
        arguments = json.loads(tool_call.function.arguments)
        result = run_calculator(arguments.get("expression", ""))
        tool_messages.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": result}
        )
    return tool_messages


def _retry_after_seconds(exc: RateLimitError) -> float | None:
    """Read Groq's suggested wait time from the 429 response headers, if present."""
    header_value = exc.response.headers.get("retry-after") if exc.response else None
    try:
        return float(header_value) if header_value is not None else None
    except ValueError:
        return None


async def create_completion_with_retry(client: AsyncOpenAI, **kwargs: Any) -> Any:
    """Call chat.completions.create, backing off on Groq's free-tier 429 rate limit."""
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            return await client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            if attempt == MAX_RATE_LIMIT_RETRIES - 1:
                raise
            wait_s = _retry_after_seconds(exc) or (2**attempt)
            await asyncio.sleep(wait_s)


async def run_agent(client: AsyncOpenAI, config: AgentConfig, question: str) -> dict[str, Any]:
    """Answer one question with a tool-calling loop, tracking usage and latency.

    A model that hallucinates a tool name Groq never declared gets a hard 400 from
    the API mid-generation. That is scored as a wrong answer, not dropped from the
    experiment, so every config is judged across the same denominator of items.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt_tokens = 0
    completion_tokens = 0
    answer = "error: tool-call loop did not converge"
    start = time.perf_counter()

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = await create_completion_with_retry(
                client,
                model=config.model,
                temperature=config.temperature,
                messages=messages,
                tools=[CALCULATOR_TOOL_SCHEMA],
            )
            if response.usage is not None:
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens

            message = response.choices[0].message
            if not message.tool_calls:
                answer = message.content or ""
                break

            messages.append(message.model_dump(exclude_none=True))
            messages.extend(await _execute_tool_calls(message.tool_calls))
    except BadRequestError as exc:
        answer = f"error: model produced an invalid tool call ({exc.message})"

    latency_s = time.perf_counter() - start
    cost_usd = calculate_cost_usd(config.model, prompt_tokens, completion_tokens)

    return {
        "answer": answer.strip(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost_usd,
        "latency_s": latency_s,
    }
