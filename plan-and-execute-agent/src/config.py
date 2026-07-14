"""Runtime configuration: API key, model, loop caps, and the LLM factory.

Inputs:  GROQ_API_KEY from the environment (.env on Windows).
Outputs: a configured ChatGroq client via make_llm(); module-level constants.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env BEFORE reading any variable. On Windows `uv run` does not inherit the
# shell's environment, so this call is mandatory and must come first.
load_dotenv()

# --- API key
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# --- Model. openai/gpt-oss-120b is a free-tier Groq production model that stays
# available past the llama-3.x shutdown. It is a reasoning model, so structured
# output MUST go through method="json_schema" (see make_structured below) — the
# default tool-calling path returns prose and fails with a 400 tool_use_failed.
MODEL: str = "openai/gpt-oss-120b"

# --- Loop control.
MAX_STEPS: int = 6        # replan ceiling: hard stop on total executed steps
MAX_TOOL_ITERS: int = 8   # per-step ReAct cap: iterations before a forced answer


def require_keys() -> None:
    """Fail fast with a clear message if the Groq key is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste a free "
            "key from https://console.groq.com/keys"
        )


def make_llm(temperature: float = 0.0) -> ChatGroq:
    """Build a Groq chat model. Free-tier models need retries for rate limits."""
    return ChatGroq(
        model=MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_retries=4,
    )
