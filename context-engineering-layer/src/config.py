"""Central configuration: models, budgets, rate-limit arithmetic, and paths.

Inputs:  environment variables from .env
Outputs: module-level constants imported by the experiment's modules

This file imports nothing from src/. That is deliberate and load-bearing:
src/layers/*, context.py, tokens.py and pipeline.py must not depend on it
either, so the layer can be copied into somebody else's agent without dragging
130 lines of experiment thresholds along with it. Every constant that moves a
measured number lives here, so the leaderboard can point at one file when it
reports what it ran with.
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
SCENARIO_PATH = DATA_DIR / "scenario.json"
SUMMARY_FIXTURE_PATH = DATA_DIR / "summary_fixture.json"
CACHE_PATH = OUTPUT_DIR / "cache.sqlite"
LEDGER_PATH = OUTPUT_DIR / "ledger.sqlite"
LEADERBOARD_PATH = OUTPUT_DIR / "leaderboard.md"
COST_PLOT_PATH = OUTPUT_DIR / "context_cost.png"
RECALL_PLOT_PATH = OUTPUT_DIR / "recall_by_zone.png"

OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def results_path(model: str) -> Path:
    """Per-model results file, so a swap run never overwrites the main one."""
    return OUTPUT_DIR / f"results-{model.replace('/', '-')}.json"


# --- Models ------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Verified 2026-08-28 against console.groq.com/docs/models and
# /docs/deprecations: both are production status and absent from the
# deprecation schedule at any date. $0.15/$0.60 and $0.075/$0.30 per 1M.
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
SWAP_MODEL = os.getenv("SWAP_MODEL", "openai/gpt-oss-20b")

# USD per 1M tokens, as (input, output). Cached input is half the input rate.
PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}

# --- Rate limits -------------------------------------------------------------
# Groq free tier, verified 2026-08-28. TPD is counted ACROSS THE WHOLE ACCOUNT,
# not per key, so switching keys does not reset it. Whether TPM is per-model or
# shared is re-checked by `main.py cache-probe`; the pacing does not assume it.
# There is no header for remaining TPD - x-ratelimit-remaining-tokens is TPM
# and x-ratelimit-limit-requests is RPD - so daily spend is tracked locally.
TOKENS_PER_MINUTE = 8_000
TOKENS_PER_DAY = 200_000
REQUESTS_PER_DAY = 1_000
REQUESTS_PER_MINUTE = 30

# --- Generation --------------------------------------------------------------
# MAX_COMPLETION_TOKENS is rate-limit arithmetic, not a style setting. Groq
# reserves prompt + completion against the per-minute ceiling BEFORE it runs
# anything, so this number is subtracted from every request's headroom whether
# the model uses it or not. 256 covers a one-line answer with room for the
# reasoning gpt-oss does first. `cache-probe` measures the actual completion
# length and the truncation rate; if truncation exceeds TRUNCATION_CEILING,
# raise this to 512 and cut the probe count rather than scoring truncations.
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "256"))
REASONING_EFFORT = "low"
TEMPERATURE = 0.0
TRUNCATION_CEILING = 0.02

# The effective prompt ceiling. A request larger than this is rejected outright
# rather than queued, and that wall - not the model's 131,072-token context
# window - is what this whole project is about.
EFFECTIVE_PROMPT_CEILING = TOKENS_PER_MINUTE - MAX_COMPLETION_TOKENS

# --- What the pre-flight measured, 2026-08-28 --------------------------------
# Groq's docs say cached tokens do not count toward rate limits. Measured on
# this account they do, and the cache barely engages in the first place: one
# hit in ten calls against an identical 1,289-token prefix fired back to back
# inside fifty seconds, well above the documented 128-1024 minimum. On the one
# call that did hit - 1,280 of 1,289 tokens served from cache - refill-
# corrected spend was 1,359 tokens against 1,318 predicted if charged in full
# and 29 if exempt. The hit bought a price discount and no rate-limit relief.
#
# So a warm probe costs a full prefix, not the ~130 tokens the budget was
# planned around, and the full profile would need 3.3 days rather than one.
# The pre-registered fallback fires: profile `lite` today, the rest tomorrow.
# Re-run `main.py cache-probe` before trusting any of this on another account.
CACHED_TOKENS_ARE_RATE_LIMIT_EXEMPT = False

# --- The context budget ------------------------------------------------------
# What every layer assembles to, and the only number that drives cost: the
# transcript can be as long as we like because the one arm that would send all
# of it is rejected before dispatch, for free. 900 tokens is a 15.6x
# compression against the 14,000-token transcript - severe, and exactly the
# regime a free-tier agent actually lives in.
#
# At this budget the naive window holds 4 raw turns and capping holds 9. That
# is the quotable core of the project, and it is why the budget is not lowered
# further: at 700 the contrast collapses to 3 turns against 6.
CONTEXT_BUDGET_TOKENS = int(os.getenv("CONTEXT_BUDGET_TOKENS", "900"))

# Transcript shape. Many small turns rather than few large ones: the position
# axis needs turns to spread facts across, and the retrieval pool needs to be
# big enough that BM25 is doing real work instead of a twenty-item lookup.
TRANSCRIPT_TURNS = 100
# Tool output dominates the turn on purpose. That is what real agent traffic
# looks like, and it is what sets how much capping can buy: a 140-token turn
# caps to 70, so capping exactly doubles how many turns fit the same budget.
TURN_USER_TOKENS = 15
TURN_ASSISTANT_TOKENS = 20
TURN_TOOL_TOKENS = 105

# --- Layer knobs -------------------------------------------------------------
# Tool outputs are what actually kill agents, so the cap targets them. Head and
# tail are both kept because the useful framing of a JSON blob lives at both
# ends; everything between is elided behind a marker the model can cite.
CAP_MAX_MESSAGE_TOKENS = 35
CAP_HEAD_TOKENS = 22
CAP_TAIL_TOKENS = 8

PIN_MAX_TOKENS = 40

# The retrieved block re-injects the UNCAPPED original turn. Re-injecting a
# capped one would hand back a fact whose middle `cap` had already elided, and
# 40% of the planted facts live in exactly that middle - which would hold the
# full stack's ceiling near 60% for a reason having nothing to do with
# retrieval quality. The block is a FIXED reservation whether or not
# de-duplication drops a hit: returning freed tokens to the window would change
# turns_kept per probe, moving a block before the prefix divergence point and
# destroying the cache stability the suffix placement exists to buy.
RETRIEVE_K = 2
RETRIEVE_BLOCK_TOKENS = 280
RETRIEVE_MIN_SCORE = 0.0

SUMMARY_BUDGET_TOKENS = 80
SUMMARY_TRIGGER_RATIO = 1.0

WINDOW_MIN_TURNS = 1

# --- Pacing ------------------------------------------------------------------
# Concurrency 3+ makes workers wake together after a 429, spend the refilled
# budget at once and re-429 in lockstep until retries run out. The SDK's own
# retry loop is disabled (max_retries=0) because it retries without telling the
# harness tokens were spent, which desyncs the local ledger exactly when it
# matters most.
MAX_CONCURRENCY = 2
MAX_RETRIES = 4
REQUEST_SPACING_S = 1.0

# --- Cache-key versions ------------------------------------------------------
# Bumping either invalidates every cached response, so a prompt or assembler
# edit can never silently shift a score.
PROMPT_VERSION = "v1"
ASSEMBLER_VERSION = "v4"  # v4: block costing includes per-message framing  # v3: realistic filler prose - repeated-word padding made the model decline to read the context at all

# tiktoken's o200k_harmony counts raw text; it does not see the Harmony chat
# template's role headers and per-message framing. `cache-probe` measures the
# ratio of our count to Groq's reported prompt_tokens and writes it here.
TOKENIZER_NAME = "o200k_harmony"
# Measured 2026-08-28: Groq reported 1,289 prompt tokens where the local count
# said 1,205. Applied to every budget assertion in the project.
TOKENIZER_FUDGE = float(os.getenv("TOKENIZER_FUDGE", "1.07"))

# --- Grading -----------------------------------------------------------------
ANSWER_PREFIX = "Answer:"
UNKNOWN_TOKEN = "UNKNOWN"

# Byte-identical in every arm. Only the history block varies between arms - if
# the instruction moved too, the null arm would be measuring task framing as
# well as guessability. Permitting UNKNOWN is what separates a model that
# forgot from one that confabulated, and it is what makes the never-planted
# negative probes gradeable at all.
ANSWER_INSTRUCTION = (
    "Answer from the conversation above. Reply with exactly one line: "
    f"{ANSWER_PREFIX} <value>. If the conversation does not contain the "
    f"answer, reply {ANSWER_PREFIX} {UNKNOWN_TOKEN}."
)


def require_api_key() -> str:
    """Return the Groq key, or exit with the exact fix."""
    if not GROQ_API_KEY:
        raise SystemExit(
            "GROQ_API_KEY is not set.\n"
            f"  Put it in {ROOT_DIR / '.env'} as:\n"
            "      GROQ_API_KEY=gsk_...\n"
            "  Free key: https://console.groq.com/keys"
        )
    return GROQ_API_KEY


if __name__ == "__main__":
    print(f"root                     {ROOT_DIR}")
    print(f"chat model               {CHAT_MODEL}")
    print(f"swap model               {SWAP_MODEL}")
    print(f"context budget           {CONTEXT_BUDGET_TOKENS:,} tok")
    print(f"max completion           {MAX_COMPLETION_TOKENS:,} tok")
    print(
        f"effective prompt ceiling {EFFECTIVE_PROMPT_CEILING:,} tok"
        f"   (= {TOKENS_PER_MINUTE:,} TPM - {MAX_COMPLETION_TOKENS} completion)"
    )
    print(
        f"retrieve                 k={RETRIEVE_K}, "
        f"block {RETRIEVE_BLOCK_TOKENS} tok, uncapped"
    )
    print(f"transcript               {TRANSCRIPT_TURNS} turns")
    print(f"tokenizer                {TOKENIZER_NAME}, fudge {TOKENIZER_FUDGE}")
    print(f"cached tokens exempt     {CACHED_TOKENS_ARE_RATE_LIMIT_EXEMPT} (measured)")
    print(f"api key                  {'set' if GROQ_API_KEY else 'MISSING'}")
