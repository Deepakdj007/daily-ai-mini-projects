"""Every knob the agent has, in one place.

Inputs:  environment variables from .env
Outputs: constants, paths, require_api_key()

Two of the numbers below are arithmetic rather than taste.

MAX_COMPLETION_TOKENS["write"] is large because a write_tool call carries a
whole Python module in its arguments, and Groq reserves max_completion_tokens
against the per-minute bucket before the call runs. Set it too low and the
model's source is truncated mid-function, which arrives as a syntax error the
model cannot see the cause of.

MAX_REPAIR_ATTEMPTS is the main cost lever in the project. Every repair
re-sends the previous source in the assistant message and again in the prompt
of every later round of that task, so the third repair is the most expensive
token in the run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
GENERATED_DIR = ROOT_DIR / "generated"

# On Windows `uv run` does not inherit the shell's environment, so this call is
# mandatory and must come before any os.getenv below.
load_dotenv(ROOT_DIR / ".env")


def _force_utf8_console() -> None:
    """Windows consoles die on the punctuation models emit (U+202F, rupee signs)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_force_utf8_console()

# --- Model ---
# llama-3.3-70b-versatile was shut down on 2026-08-16. gpt-oss-120b is the
# replacement, and it is one of the two Groq models that support strict
# structured output, which write_tool depends on.
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
TEMPERATURE = 0.0
REASONING_EFFORT = "low"
PROMPT_VERSION = "1"

# Groq reserves this against the per-minute bucket BEFORE the call runs, so
# these are arithmetic rather than taste: six rounds at 2400 reserve 14,400
# tokens against a limit of 8,000, and the task dies on a 429 it never spent.
# Measured write_tool calls land between 550 and 950 completion tokens, so 1600
# is roughly double the observed need and still lets five rounds fit a minute.
MAX_COMPLETION_TOKENS = {
    "write": 1600,   # a round where write_tool or run_python is on the table
    "answer": 1400,  # a round where it is not
}
# Why "answer" is not small: an arm with no way to write a tool tries to do the
# checksum in its head, and that reasoning ran past 500 tokens on every single
# task - so every baseline row came back empty and was scored as an API error
# rather than as a wrong answer. A budget that turns "the model got it wrong"
# into "the run failed" destroys the floor the whole ladder is measured against.

# --- The agent loop ---
MAX_ROUNDS = 6
MAX_REPAIR_ATTEMPTS = 3
MAX_RESULT_CHARS = 1500

# --- The sandbox ---
SANDBOX_TIMEOUT_S = 10.0
SANDBOX_MAX_OUTPUT = 256 * 1024
SANDBOX_POLL_S = 0.02
MAX_SOURCE_CHARS = 8000

# What a generated tool may import. Default-deny: a module is here because one
# of the six families needs it, not because it looks harmless.
#
#   math, datetime, calendar   the date and numeric families are these modules
#   decimal                    money must not be float
#   re                         semver parsing; its risk is catastrophic
#                              backtracking, which is a CPU spin the timeout catches
#   string                     base-36 is digits + ascii_uppercase
#   json                       models write `import json` by reflex and its
#                              dangerous surface needs a file object it cannot get
#   itertools, functools       cmp_to_key is how semver precedence is spelled
#   collections                note: namedtuple execs internally. The source it
#                              execs is built by collections, not by the tool.
#   typing                     so a type hint in a signature does not fail
#
# Deliberately absent, and worth naming: os, sys, subprocess, socket, pathlib,
# importlib, builtins, random, time, statistics, fractions - and `operator`,
# the near-miss, because operator.attrgetter("__class__") is a string-built
# attribute access with no getattr call, which is the one bypass class the
# guard works hardest to close.
#
# This list is data, not doctrine. If a legitimate tool needs something, add it
# here and re-run the red team.
ALLOWED_IMPORTS: frozenset[str] = frozenset({
    "math", "datetime", "re", "decimal", "json", "itertools",
    "functools", "collections", "string", "calendar", "typing",
})

# The date the fixed `today()` tool reports. Frozen so the date families give
# the same answer in March as in September; a corpus whose gold answers move
# with the wall clock cannot be re-run.
FROZEN_TODAY = "2026-09-15"

# --- Rate limits (Groq free tier, verified 2026-09-17) ---
TOKENS_PER_MINUTE = 8_000
TOKENS_PER_DAY = 200_000
REQUESTS_PER_MINUTE = 30
REQUESTS_PER_DAY = 1_000
# One task can spend most of a minute's allowance on its own, so there is no
# concurrency to be had here: a second worker only buys 429s.
MAX_CONCURRENCY = 1
REQUEST_SPACING_S = 2.0
MAX_RETRIES = 4
SQLITE_TIMEOUT = 30.0
DAILY_STOP_AT = 0.95

# --- Paths ---
TASKS_PATH = DATA_DIR / "tasks.json"
CACHE_PATH = OUTPUT_DIR / "cache.sqlite"
LEDGER_PATH = OUTPUT_DIR / "ledger.sqlite"
LEADERBOARD_PATH = OUTPUT_DIR / "leaderboard.md"


def results_path(model: str = "") -> Path:
    """Where one model's results live. Slashes are not path separators here."""
    return OUTPUT_DIR / f"results-{(model or CHAT_MODEL).replace('/', '-')}.json"


def ensure_dirs() -> None:
    """Create the directories the run writes into."""
    for path in (DATA_DIR, OUTPUT_DIR, GENERATED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def require_api_key() -> str:
    """Fail loudly and early, with the URL that fixes it."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste a free "
            "key from https://console.groq.com/keys"
        )
    return key


if __name__ == "__main__":
    print(f"model          {CHAT_MODEL}")
    print(f"budgets        {MAX_COMPLETION_TOKENS}")
    print(f"rounds         {MAX_ROUNDS} max, {MAX_REPAIR_ATTEMPTS} repairs per task")
    print(f"sandbox        {SANDBOX_TIMEOUT_S}s, {SANDBOX_MAX_OUTPUT:,} bytes of output")
    print(f"frozen today   {FROZEN_TODAY}")
    print(f"generated/     {GENERATED_DIR}")
    print(f"results        {results_path()}")
    print(f"api key        {'set' if os.getenv('GROQ_API_KEY') else 'MISSING'}")
