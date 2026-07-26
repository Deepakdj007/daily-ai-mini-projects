"""Runtime configuration: API key, model, paths, poll interval, and the Groq client.

Inputs:  GROQ_API_KEY from the environment (.env on Windows).
Outputs: a configured AsyncGroq client via make_client(); module-level path/const values.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import AsyncGroq

# Load .env BEFORE reading any variable. On Windows `uv run` does not inherit the
# shell's environment, so this call is mandatory and must come first.
load_dotenv()

# --- API key
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# --- Model. openai/gpt-oss-120b is a free-tier Groq production model with a 131k
# context window and strict json_schema structured outputs. We avoid llama-3.3-70b:
# it is scheduled to shut down on 2026-08-16.
MODEL: str = "openai/gpt-oss-120b"

# --- Paths. The DB is the source of truth; FAQ.md and CHANGELOG.md are rendered from it.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DOCS_DIR: Path = ROOT_DIR / "docs"
DB_PATH: Path = ROOT_DIR / "faq_autopilot.db"
FAQ_PATH: Path = ROOT_DIR / "FAQ.md"
CHANGELOG_PATH: Path = ROOT_DIR / "CHANGELOG.md"

# --- How often the watch loop scans the docs folder for drift, in seconds.
POLL_INTERVAL_SECONDS: float = 5.0


def require_keys() -> None:
    """Fail fast with a clear message if the Groq key is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste a free "
            "key from https://console.groq.com/keys"
        )


def make_client() -> AsyncGroq:
    """Build an async Groq client. Free-tier models need retries for rate limits."""
    return AsyncGroq(api_key=GROQ_API_KEY, max_retries=4)
