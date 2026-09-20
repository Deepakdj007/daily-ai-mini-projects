"""Every knob the agent has, in one place.

Inputs:  environment variables from .env
Outputs: constants, paths, require_api_key()

The token budgets are arithmetic, not taste. Groq charges
max_completion_tokens against the per-minute bucket when the request arrives,
not when the answer comes back, so six rounds at 2,400 reserve 14,400 tokens
against a limit of 8,000 and the task dies on a 429 it never actually spent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
GENERATED_DIR = ROOT_DIR / "generated"
TASKS_PATH = DATA_DIR / "tasks.json"

# On Windows `uv run` does not inherit the shell's environment, so this must
# come before anything reads a key.
load_dotenv(ROOT_DIR / ".env")

for _stream in (sys.stdout, sys.stderr):
    try:                    # Windows consoles are cp1252 and tools return
        _stream.reconfigure(encoding="utf-8", errors="replace")   # rupee signs
    except (AttributeError, ValueError):
        pass

# llama-3.3-70b was shut down on 2026-08-16. gpt-oss-120b is the replacement.
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
TEMPERATURE = 0.0
REASONING_EFFORT = "low"

MAX_COMPLETION_TOKENS = 1600
# Ten, not six. Three rejected attempts plus the round that registers the tool
# plus the answer is five before a single call, and the model tends to call its
# tool once per item rather than passing the whole list - so a tight budget
# cuts off exactly the run that worked hardest to get there.
MAX_ROUNDS = 10
MAX_REPAIR_ATTEMPTS = 3
MAX_RESULT_CHARS = 1500
MAX_RETRIES = 4

TOKENS_PER_MINUTE = 8_000       # Groq free tier
REQUEST_SPACING_S = 2.0

SANDBOX_TIMEOUT_S = 10.0
SANDBOX_MAX_OUTPUT = 256 * 1024
SANDBOX_POLL_S = 0.02
MAX_SOURCE_CHARS = 8000

# What a generated tool may import. Default-deny: a module is here because a
# tool plausibly needs it, not because it looks harmless. Deliberately absent,
# and worth naming: os, sys, subprocess, socket, pathlib, importlib, builtins -
# and `operator`, the near-miss, because operator.attrgetter("__class__") is a
# string-built attribute access with no getattr call anywhere in it.
ALLOWED_IMPORTS: frozenset[str] = frozenset({
    "math", "datetime", "re", "decimal", "json", "itertools",
    "functools", "collections", "string", "calendar", "typing",
})


def ensure_dirs() -> None:
    """Create the directories the agent writes into."""
    for path in (DATA_DIR, GENERATED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def require_api_key() -> str:
    """Fail loudly and early, with the URL that fixes it."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste a free "
            "key from https://console.groq.com/keys")
    return key


if __name__ == "__main__":
    print(f"model      {CHAT_MODEL}")
    print(f"budget     {MAX_COMPLETION_TOKENS} completion tokens per round, "
          f"{MAX_ROUNDS} rounds, {MAX_REPAIR_ATTEMPTS} repairs")
    print(f"sandbox    {SANDBOX_TIMEOUT_S}s, {SANDBOX_MAX_OUTPUT:,} bytes")
    print(f"imports    {', '.join(sorted(ALLOWED_IMPORTS))}")
    print(f"api key    {'set' if os.getenv('GROQ_API_KEY') else 'MISSING'}")
