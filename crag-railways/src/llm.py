"""Two thin wrappers over AsyncGroq: one returns parsed JSON, one returns plain text.

Inputs:  a system prompt, a user prompt, and (for JSON) a schema
Outputs: a dict that matches the schema, or the model's text reply
"""

import json

from groq import AsyncGroq

from src.config import ANSWER_MAX_TOKENS, GRADE_MAX_TOKENS, MODEL, require_api_key

_client: AsyncGroq | None = None


def client() -> AsyncGroq:
    """Create the Groq client once, after .env has been loaded."""
    global _client
    if _client is None:
        # The free tier allows 8,000 tokens a minute; one demo uses most of that, so
        # retry 429s a few more times than the SDK's default of 2.
        _client = AsyncGroq(api_key=require_api_key(), max_retries=5)
    return _client


async def ask_json(system: str, user: str, name: str, schema: dict) -> dict:
    """Ask for a reply that must match the schema. Strict mode makes Groq enforce it."""
    response = await client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        },
        temperature=0,
        max_completion_tokens=GRADE_MAX_TOKENS,
    )
    return json.loads(response.choices[0].message.content)


async def ask_text(system: str, user: str) -> str:
    """Ask for a free-text reply."""
    response = await client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
        max_completion_tokens=ANSWER_MAX_TOKENS,
    )
    return (response.choices[0].message.content or "").strip()
