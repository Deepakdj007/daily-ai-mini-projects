"""Runtime configuration: API key, model, file paths, and the LLM factory.

Inputs:  GROQ_API_KEY from the environment (.env on Windows).
Outputs: a configured ChatGroq client via make_llm(); module-level paths and constants.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env BEFORE reading any variable. On Windows `uv run` does not inherit
# the shell's environment, so this call is mandatory and must come first.
load_dotenv()

# --- API key
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# --- Model (free tier, no credit card)
MODEL: str = "llama-3.3-70b-versatile"

# --- File paths
DB_PATH: Path = Path("checkpoints.sqlite")          # durable LangGraph checkpoints
OUTPUT_DIR: Path = Path("output")
AUDIT_LOG: Path = OUTPUT_DIR / "audit_log.jsonl"    # append-only compliance trail
LEDGER: Path = OUTPUT_DIR / "ledger.json"           # record of executed actions

# --- Loop control: caps the validator-bounce and human-edit re-draft loops so
# neither can spin forever. Mirrors reflexion-self-correcting's MAX_REVISIONS.
MAX_REVISIONS: int = 3


def require_keys() -> None:
    """Fail fast with a clear message if the Groq key is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste a free "
            "key from https://console.groq.com/keys"
        )


def make_llm(temperature: float = 0.3) -> ChatGroq:
    """Build a Groq chat model. Free-tier models need retries for rate limits."""
    return ChatGroq(
        model=MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_retries=4,
    )
