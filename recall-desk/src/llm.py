"""Groq client and the structured-output path every tier depends on.

Each memory tier asks the model for a typed decision rather than prose, so the
write path never parses free text.

Getting reliable JSON out of `gpt-oss-120b` took two fixes, both of which show up
as hard failures rather than degraded output:

1. langchain's default `method="function_calling"` makes the model invent a tool
   named after one of the schema's own fields:

       400 - attempted to call tool 'escalate' which was not in request.tools

   `method="json_schema"` sends the schema as a `response_format`, so there is no
   tool to hallucinate.

2. `json_schema` is still not a guarantee. Roughly one call in twenty comes back
   as prose with no JSON at all, and Groq rejects it:

       400 - Failed to generate JSON. Please adjust your prompt.
       failed_generation: 'Could you share the request ID? To rotate, go to...'

   The groq SDK does not retry 400s, because a 400 usually means the request is
   wrong — here the request is fine and the sample was bad. So retries live here,
   with a nudge appended and the temperature raised a little so the retry is not
   a rerun of the same sample.
"""

from __future__ import annotations

from typing import Any, TypeVar

from groq import BadRequestError
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from pydantic import BaseModel

from src.config import GROQ_MODEL, require_api_key

T = TypeVar("T", bound=BaseModel)

ATTEMPTS = 3
_NUDGE = ("human", "Reply with the JSON object only. No text outside it.")


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Build a ChatGroq client.

    Args:
        temperature: 0.0 for the write path so memory decisions are stable;
            the answer path uses a little more.

    Returns:
        A configured ChatGroq instance.
    """
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=require_api_key(),
        temperature=temperature,
        max_retries=3,  # covers 429s and 5xx; not the 400 described above
    )


def structured(schema: type[T], temperature: float = 0.0, include_raw: bool = False):
    """Return a runnable that replies with an instance of `schema`.

    Args:
        schema: the Pydantic model the model must fill in.
        temperature: sampling temperature.
        include_raw: when true, `.invoke()` returns a dict with "parsed" and
            "raw" instead of the bare model, so token usage can be read off the
            raw response.

    Returns:
        A runnable whose `.invoke()` returns a `schema` instance, or a dict with
        "parsed"/"raw" when `include_raw` is set.
    """
    return get_llm(temperature).with_structured_output(
        schema, method="json_schema", include_raw=include_raw
    )


def invoke(
    schema: type[T],
    messages: list,
    temperature: float = 0.0,
    include_raw: bool = False,
) -> Any:
    """Call the model for a typed result, retrying bad JSON samples.

    Args:
        schema: the Pydantic model to fill.
        messages: chat messages, as (role, content) tuples or message objects.
        temperature: starting temperature; nudged up on each retry.
        include_raw: return the {"parsed", "raw"} dict instead of the model.

    Returns:
        A `schema` instance, or the raw dict when `include_raw` is set.

    Raises:
        RuntimeError: if every attempt failed to produce valid JSON.
    """
    problem = "unknown"
    for attempt in range(ATTEMPTS):
        runnable = structured(schema, temperature + 0.1 * attempt, include_raw=include_raw)
        payload = messages if attempt == 0 else [*messages, _NUDGE]
        try:
            result = runnable.invoke(payload)
        except BadRequestError as exc:
            if "json_validate_failed" not in str(exc):
                raise
            problem = "model returned prose instead of JSON"
            continue

        # With include_raw, a failed parse is reported in-band as parsed=None
        # rather than raised. Left unchecked it becomes an AttributeError three
        # frames away from the cause.
        if include_raw and result.get("parsed") is None:
            problem = f"parse failed: {result.get('parsing_error')}"
            continue
        return result

    raise RuntimeError(f"no valid JSON after {ATTEMPTS} attempts ({problem})")


def ask(schema: type[T], system: str, user: str, temperature: float = 0.0) -> T:
    """One-shot structured call.

    Args:
        schema: the Pydantic model to fill.
        system: system instruction.
        user: user content.
        temperature: sampling temperature.

    Returns:
        A validated `schema` instance.
    """
    return invoke(schema, [("system", system), ("human", user)], temperature)
