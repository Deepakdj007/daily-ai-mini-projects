"""Runtime configuration: API keys, model, paths, timezone, and the Groq client.

Inputs:  TELEGRAM_BOT_TOKEN, GROQ_API_KEY, OWNER_CHAT_ID from the environment (.env on Windows).
Outputs: a configured ChatGroq via make_llm(); module-level path/const values used by every other module.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env BEFORE reading any variable. On Windows `uv run` does not inherit the
# shell's environment, so this call is mandatory and must come first.
load_dotenv()

# --- API keys
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# --- Owner gate. 0 means unset — /start is the only handler exempt from the gate,
# so the reader can learn their chat id and paste it into .env.
OWNER_CHAT_ID: int = int(os.getenv("OWNER_CHAT_ID") or "0")

# --- Model. openai/gpt-oss-120b is a free-tier Groq production model with a 131k
# context window and native tool calling. We avoid llama-3.3-70b-versatile: it is
# scheduled to shut down on 2026-08-16.
MODEL: str = "openai/gpt-oss-120b"

# --- Paths
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DB_PATH: Path = ROOT_DIR / "assistant.db"        # notes + reminders (stdlib sqlite3)
MEM_PATH: Path = ROOT_DIR / "memory.db"          # LangGraph checkpoints (conversation memory)
CREDENTIALS_PATH: Path = ROOT_DIR / "credentials.json"  # optional Google OAuth client secret
TOKEN_PATH: Path = ROOT_DIR / "token.json"       # optional cached Google OAuth token

# --- Timezone. PTB's JobQueue/APScheduler default to UTC, and readers of this
# series are in India, so every time-shaped thing (the morning briefing, reminder
# math, "what time is it") uses this instead of UTC.
TIMEZONE: str = "Asia/Kolkata"

# --- Morning briefing
BRIEFING_HOUR: int = 7
BRIEFING_TOPICS: list[str] = [
    "top world news today",
    "top AI and technology news today",
]

# --- Tool output limits. Some free-tier Groq orgs cap tokens-per-minute as low
# as 8000 — far below the model's 131k context window, and easy to blow through
# with even one long tool result. Every tool output is truncated before it
# reaches the model, and agent.py's ContextEditingMiddleware clears old tool
# results out of history so a long conversation doesn't creep back over the cap.
MAX_TOOL_CHARS: int = 3000
SEARCH_MAX_RESULTS: int = 5

# --- Telegram file limits
MAX_DOCUMENT_BYTES: int = 20 * 1024 * 1024  # Bot API hard cap on file downloads


def require_keys() -> None:
    """Fail fast with a clear message if a required secret is missing."""
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Copy .env.example to .env and paste "
            "the token @BotFather gave you on https://t.me/BotFather"
        )
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste a free "
            "key from https://console.groq.com/keys"
        )


def make_llm() -> ChatGroq:
    """Build the Groq chat model used by the agent. temperature=0 keeps tool
    routing deterministic; retries absorb the free tier's occasional 429."""
    return ChatGroq(model=MODEL, api_key=GROQ_API_KEY, temperature=0, max_retries=3)
