"""Central configuration for the SLM+LLM router.

Inputs:  environment variables from .env (GROQ_API_KEY, optional overrides).
Outputs: model IDs, tier settings, filesystem paths, and the verifier
         strictness grid that generates the deferral curve.

Everything tunable lives here so the experiment has exactly one place where a
knob can be turned. That matters for a measurement project: a threshold hidden
inside a module is a threshold nobody audits.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# On Windows `uv run` does not inherit the shell's environment, so this call is
# mandatory and must come before any os.getenv below.
load_dotenv()


def _force_utf8_console() -> None:
    """Stop Windows cp1252 consoles crashing on model output.

    Model answers routinely contain curly quotes, en-dashes and narrow
    no-break spaces. On a default Windows console those raise
    UnicodeEncodeError mid-run and destroy a long benchmark pass.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


_force_utf8_console()

# --- Paths -------------------------------------------------------------------
# Derived from this file, never from the current working directory, so the
# project behaves identically no matter where it is launched from.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
QUERIES_PATH = DATA_DIR / "queries.json"
CACHE_PATH = OUTPUT_DIR / "cache.sqlite"
LEADERBOARD_PATH = OUTPUT_DIR / "leaderboard.md"
CURVE_PATH = OUTPUT_DIR / "deferral_curve.png"
RESULTS_PATH = OUTPUT_DIR / "results.json"

OUTPUT_DIR.mkdir(exist_ok=True)

# --- Tiers -------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# The big model. Verified 2026-08-15 against console.groq.com/docs/models:
# production status, $0.15 in / $0.60 out per 1M, 131K context, and absent from
# the deprecation list at any date.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# SLM_PROVIDER selects the cheap tier.
#   "ollama" -> local model, $0 marginal API cost. This is what makes a ~90%
#               saving arithmetically reachable at all.
#   "groq"   -> openai/gpt-oss-20b. Needs no local install, but it is only 2x
#               cheaper than the 120b, so the savings ceiling collapses to ~50%.
#               Kept as a first-class mode because that ceiling IS the lesson:
#               the price gap between your tiers is what makes routing worth it.
SLM_PROVIDER = os.getenv("SLM_PROVIDER", "ollama").strip().lower()

# qwen3.5:4b - 3.4GB Q4_K_M, 4.66B params, 256K context, Apache-2.0.
# Lighter tags for smaller machines: qwen3.5:2b-q4_K_M (1.9GB), qwen3.5:0.8b
# (1.0GB). Expect the savings number to FALL on those - a weaker small model
# escalates more often, and watching that happen is the point.
SLM_OLLAMA_MODEL = os.getenv("SLM_OLLAMA_MODEL", "qwen3.5:4b")
SLM_GROQ_MODEL = os.getenv("SLM_GROQ_MODEL", "openai/gpt-oss-20b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST") or None

# --- Generation --------------------------------------------------------------
# Short cap because every eval item is authored to want a short answer. On a
# CPU-only laptop this is the difference between 30s and 8s per query, and it
# keeps the cost ledger tight.
SLM_MAX_TOKENS = int(os.getenv("SLM_MAX_TOKENS", "160"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "320"))
TEMPERATURE = 0.0

# Self-consistency sampling for the verifier. k=3 means the SLM answers three
# times; disagreement is the escalation signal. Local, so this costs wall-clock
# rather than dollars - which is itself a finding worth reporting.
SELF_CONSISTENCY_K = int(os.getenv("SELF_CONSISTENCY_K", "3"))
SELF_CONSISTENCY_TEMP = 0.7

# --- Verifier strictness grid ------------------------------------------------
# Each level escalates on a superset of the level below, so escalation rate is
# monotonic in strictness. Sweeping this grid is what draws the deferral curve;
# no trained model is involved.
#   0  never escalate                      -> the all_slm floor
#   1  escalate on empty/unparseable output
#   2  + escalate when the answer fails its item's format constraint
#   3  + escalate on hedging language
#   4  + escalate when self-consistency samples disagree
#   5  always escalate                     -> the baseline ceiling
STRICTNESS_LEVELS = [0, 1, 2, 3, 4, 5]

# The operating point quoted in the README headline. Declared here, in advance,
# rather than chosen after seeing the results - see README's pre-registered
# criterion. Changing this after a run and re-quoting is exactly the failure
# this constant exists to prevent.
HEADLINE_STRICTNESS = int(os.getenv("HEADLINE_STRICTNESS", "4"))

# --- Prompting ---------------------------------------------------------------
# Every item is graded by exact match against a short gold answer, so both tiers
# are told to end with a clean anchor line. The grader keys on this.
ANSWER_PREFIX = "Answer:"
SYSTEM_PROMPT = (
    "You are a concise assistant. Think briefly, then give the shortest "
    "correct answer.\n"
    f"You MUST end your reply with a final line of the form '{ANSWER_PREFIX} <answer>'.\n"
    "The answer must be the bare value only - no units unless asked, no "
    "explanation, no punctuation at the end."
)

# Bumping this string invalidates every cached response. The cache key includes
# it because prompts WILL change mid-build, and stale entries silently poison
# results in a way that looks like a real effect.
PROMPT_VERSION = "v1"

# --- Rate limits -------------------------------------------------------------
# Groq free tier, verified 2026-08-15: 30 RPM, 8k TPM, 1,000 RPD, 200,000 TPD.
# TPD is counted ACROSS THE WHOLE ACCOUNT, not per key - a benchmark loop can
# lock you out of every other project for 24 hours.
MAX_CONCURRENCY = 2
REQUEST_SPACING_S = 2.1


def require_keys() -> None:
    """Fail fast with an actionable message when the Groq key is missing."""
    if not GROQ_API_KEY:
        raise SystemExit(
            "GROQ_API_KEY is not set.\n"
            "Create a .env file next to pyproject.toml containing:\n"
            "    GROQ_API_KEY=gsk_...\n"
            "Get a free key at https://console.groq.com/keys"
        )


def slm_label() -> str:
    """Human-readable name of the active small tier, for tables and logs."""
    if SLM_PROVIDER == "ollama":
        return f"{SLM_OLLAMA_MODEL} (local)"
    return f"{SLM_GROQ_MODEL} (groq)"


if __name__ == "__main__":
    print(f"root          : {ROOT_DIR}")
    print(f"llm tier      : {LLM_MODEL}")
    print(f"slm tier      : {slm_label()}")
    print(f"slm provider  : {SLM_PROVIDER}")
    print(f"strictness    : {STRICTNESS_LEVELS} (headline={HEADLINE_STRICTNESS})")
    print(f"groq key      : {'set' if GROQ_API_KEY else 'MISSING'}")
