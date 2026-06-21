"""Shared async LLM helpers: a rate-limit-aware invoker plus thin wrappers.

Groq's free tier caps tokens-per-minute, so a burst of parallel agents can hit a
429. `ainvoke_with_backoff` waits the few seconds Groq suggests and retries, so a
transient rate limit self-heals instead of failing the run. Groq/Llama returns
`.content` as a plain string, so no block flattening is needed.
"""

from __future__ import annotations

import asyncio
import re

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

_RETRY_HINT = re.compile(r"try again in ([\d.]+)s")


def _is_rate_limit(exc: Exception) -> bool:
    """True when an exception is a Groq token/request rate-limit (429)."""
    text = str(exc).lower()
    return "rate_limit" in text or "429" in text or "tokens per minute" in text


def _suggested_wait(exc: Exception, default: float = 8.0) -> float:
    """Seconds to wait before retry, parsed from Groq's message when present."""
    match = _RETRY_HINT.search(str(exc))
    return float(match.group(1)) + 0.5 if match else default


async def ainvoke_with_backoff(model: ChatGroq, messages: list,
                               retries: int = 2) -> BaseMessage:
    """Invoke a model, retrying transient 429s after the suggested wait."""
    for attempt in range(retries + 1):
        try:
            return await model.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001
            if attempt == retries or not _is_rate_limit(exc):
                raise
            await asyncio.sleep(min(_suggested_wait(exc), 15.0))
    raise RuntimeError("unreachable")  # pragma: no cover


async def run_llm(model: ChatGroq, system: str, user: str) -> str:
    """Call a ChatGroq model with a system+user prompt and return its text."""
    reply = await ainvoke_with_backoff(model, [("system", system), ("human", user)])
    content = reply.content
    if isinstance(content, list):  # defensive: some providers return blocks
        content = " ".join(str(part) for part in content)
    return str(content).strip()


async def safe_analyse(model: ChatGroq, system: str, user: str) -> tuple[str, bool]:
    """Run the LLM but never raise: return (text, ok) so a worker can degrade."""
    try:
        return await run_llm(model, system, user), True
    except Exception as exc:  # noqa: BLE001 — one worker's failure isn't fatal
        return f"analysis unavailable (LLM error: {exc})", False
