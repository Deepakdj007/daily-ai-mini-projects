"""Every knob the research team has, in one place.

Holds the API key, the model, the size of the team, and the token budgets. The
budgets are not a style preference — Groq's free tier reserves `max_tokens`
against the per-minute bucket *before* the call runs, so they are arithmetic.

Inputs:  environment variables from .env
Outputs: constants, require_keys(), make_llm(), structured()
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel

import os

# Windows `uv run` does not inherit .env, so load it before anything reads a key.
load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# llama-3.3-70b-versatile was shut down on 2026-08-16. gpt-oss-120b is the
# replacement Groq names, and it is one of only two models that support strict
# structured output, which this project relies on.
MODEL: str = "openai/gpt-oss-120b"
REASONING_EFFORT: str = "low"

# Two researchers, not three. Both run in the same superstep, so both reserve
# their max_tokens against the 8,000-token-per-minute bucket at the same time.
# A third pushes one run past the bucket and earns a 429 on every attempt.
NUM_RESEARCHERS: int = 2
SEARCH_RESULTS: int = 5
SNIPPET_CHARS: int = 300

MAX_TOKENS_LEAD: int = 900
MAX_TOKENS_RESEARCHER: int = 900
MAX_TOKENS_WRITER: int = 1500

# never: nobody may ask.  lead: only the lead may ask.  team: researchers too.
ASK_POLICIES: tuple[str, ...] = ("never", "lead", "team")
DEFAULT_POLICY: str = "team"

ROOT: Path = Path(__file__).resolve().parent.parent
DB_PATH: Path = ROOT / "checkpoints.sqlite"
OUTPUT_DIR: Path = ROOT / "output"
BRIEFS_PATH: Path = ROOT / "data" / "briefs.json"


def require_keys() -> None:
    """Fail with a useful message instead of a stack trace three calls deep."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste a free "
            "key from https://console.groq.com/keys"
        )


def make_llm(max_tokens: int) -> ChatGroq:
    """Build the Groq chat model with a specific output budget.

    Args:
        max_tokens: cap on the response, reserved against the per-minute bucket.

    Returns:
        A configured ChatGroq client.

    max_retries is the entire pacing mechanism. The Groq SDK honours the
    `retry-after` header when it is 60 seconds or less, which is exactly what a
    per-minute rate limit sends back. A hand-rolled sleep would race between the
    two researcher threads; this does not.
    """
    require_keys()
    return ChatGroq(
        model=MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.0,
        max_tokens=max_tokens,
        reasoning_effort=REASONING_EFFORT,
        max_retries=6,
    )


def structured(schema: type[BaseModel], max_tokens: int):
    """Return a model that replies as an instance of `schema`, or fails loudly.

    Args:
        schema: the pydantic model the reply must match.
        max_tokens: output budget for the call.

    Returns:
        A runnable whose .invoke(prompt) returns a `schema` instance.

    method="json_schema" is required. The default path asks the model to call a
    tool, and gpt-oss is a reasoning model that writes prose instead, which Groq
    rejects with `400 tool_use_failed`. strict=True then turns on constrained
    decoding, so the reply cannot be malformed JSON in the first place.
    """
    return make_llm(max_tokens).with_structured_output(
        schema, method="json_schema", strict=True
    )
