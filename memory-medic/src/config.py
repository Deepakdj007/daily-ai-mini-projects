"""Every knob in one place: paths, models, thresholds, rate limits, versions.

Inputs:  environment variables, optionally from .env
Outputs: module-level constants the rest of the project reads

Nothing here does I/O beyond reading .env and creating the output directory.
The half-lives and confidence thresholds are the two blocks worth arguing with;
both are stated with the reasoning that produced them so a reader can disagree
on the numbers without having to reverse-engineer the intent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SOURCES_DIR = DATA_DIR / "sources"
OUTPUT_DIR = ROOT_DIR / "output"

# On Windows `uv run` does not inherit the shell's environment, so this call is
# mandatory and must come before any os.getenv below.
load_dotenv(ROOT_DIR / ".env")


def _force_utf8_console() -> None:
    """Make stdout survive model output.

    Models emit U+2019 and U+202F. A Windows console defaults to cp1252, which
    cannot encode either, and the crash lands far from the text that caused it.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_force_utf8_console()
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Paths ------------------------------------------------------------------
MEMORY_DB = OUTPUT_DIR / "memory.db"
CHECKPOINT_DB = OUTPUT_DIR / "checkpoints.db"
CACHE_PATH = OUTPUT_DIR / "cache.sqlite"
LEDGER_PATH = OUTPUT_DIR / "ledger.sqlite"
LEADERBOARD_PATH = OUTPUT_DIR / "leaderboard.md"

# --- Models -----------------------------------------------------------------
# Verified 2026-09-05 against console.groq.com/docs/deprecations: both gpt-oss
# models are absent from the deprecation list, and both support strict
# json_schema constrained decoding (probed live, see the guide's Common Errors).
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "openai/gpt-oss-20b")
SWAP_MODEL = os.getenv("SWAP_MODEL", "openai/gpt-oss-20b")

# --- Generation -------------------------------------------------------------
TEMPERATURE = 0.0
REASONING_EFFORT = "low"
# 2048, not 512: gpt-oss spends completion tokens reasoning before it emits the
# JSON. Starved, it returns a 400 json_validate_failed with an EMPTY
# failed_generation, which reads like a schema bug and is not one.
MAX_COMPLETION_TOKENS = {"extract": 2048, "classify": 1024, "answer": 512, "judge": 256}

# --- Rate limits (free tier, verified 2026-09-05) ---------------------------
TOKENS_PER_MINUTE = 8_000
TOKENS_PER_DAY = 200_000
REQUESTS_PER_MINUTE = 30
REQUESTS_PER_DAY = 1_000
# Groq publishes no remaining-TPD header, so daily spend is tracked locally.
# Whether the 200k is per-model or account-wide is an assumption until a 429
# says otherwise; `preflight` logs the body of the first one.
TPD_PER_MODEL = True
MAX_CONCURRENCY = 2
REQUEST_SPACING_S = 1.0
MAX_RETRIES = 4

# --- Store ------------------------------------------------------------------
# 9999-12-31. One sentinel instead of NULL because vec0 metadata columns support
# only = != < <= > >=, so "still valid" has to be comparable. Using it in both
# tables keeps the point-in-time predicate byte-identical.
OPEN_END = 253402300799
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIMS = 384
RECALL_K = 8
SQLITE_TIMEOUT = 30.0

# How long a fact of each class stays believable without fresh evidence.
# stable: effectively never ages, but stays in the sweep's field of view so the
#   class is not a hidden exemption.
# slow:   people change jobs and cities on a two-to-three year horizon, so six
#   months gives two or three checks per tenure without nagging.
# fast:   roughly how long "my current project" stays the same project.
# scheduled facts do not decay, they end - governed by valid_to, not by age.
HALF_LIFE_DAYS = {"stable": 3650, "slow": 180, "fast": 30}
# Freshness below this renders as "may be out of date" at read time. 0.5 is
# exactly one half-life, which is deliberately the same threshold the sweep
# uses to flag a fact durably. Two different definitions of stale would put the
# prompt and the sweep in open disagreement, and at 0.7 every fact past half of
# its half-life was hedged - a warning on everything is a warning on nothing.
FRESH_FLOOR = 0.5

# --- Repair routing ---------------------------------------------------------
# Anything that retires a row needs 0.80. Anything that only adds a row needs
# 0.60, because the cost of being wrong is a duplicate rather than a deletion.
AUTO_APPLY_MIN_CONFIDENCE = 0.80
QUALIFY_MIN_CONFIDENCE = 0.60
CONFIRM_MIN_CONFIDENCE = 0.60
# A contradiction on a stable fact (name, birthday, allergy) is more often an
# extraction error than a change in the world, so it always asks a human.
STABLE_ALWAYS_PARKS = True
# Cosine floor for calling two differently-scoped facts the same fact. High
# enough that "async for design reviews" and "sync for incident calls" - which
# share few words - never pair.
SCOPE_SIM_THRESHOLD = 0.92
EVIDENCE_MAX_CHARS = 1500
MAX_REVISIONS = 3

# --- Behaviour switches (the eval overrides these per arm) ------------------
WRITE_POLICY = os.getenv("WRITE_POLICY", "bitemporal")
GATE_ENABLED = os.getenv("GATE_ENABLED", "1") == "1"
SWEEP_ENABLED = os.getenv("SWEEP_ENABLED", "1") == "1"
CLOCK_AT = os.getenv("CLOCK_AT", "")  # ISO date; empty = system clock
SWEEP_INTERVAL_S = float(os.getenv("SWEEP_INTERVAL_S", "30"))

# --- Prompt versions (any edit here retires every cached response) ----------
PROMPT_VERSION = "v1"
EXTRACT_SCHEMA_VERSION = "v1"
RELATION_SCHEMA_VERSION = "v1"
JUDGE_PROMPT_VERSION = "ku-v1"
STORE_VERSION = "v1"


def results_path(model: str) -> Path:
    """Per-model results file, so a swap run never overwrites the main one."""
    return OUTPUT_DIR / f"results-{model.replace('/', '-')}.json"


def require_api_key() -> str:
    """Return the Groq key, or explain exactly how to set it."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "GROQ_API_KEY is not set.\n"
            f"  1. Get a free key (no card): https://console.groq.com/keys\n"
            f"  2. Put it in {ROOT_DIR / '.env'} as:  GROQ_API_KEY=gsk_...\n"
        )
    return key


if __name__ == "__main__":
    print(f"root        {ROOT_DIR}")
    print(f"chat/answer {CHAT_MODEL}")
    print(f"extract     {EXTRACT_MODEL}")
    print(f"policy      {WRITE_POLICY}  gate={GATE_ENABLED}  sweep={SWEEP_ENABLED}")
    print(f"clock       {CLOCK_AT or 'system'}")
    print(f"half-lives  {HALF_LIFE_DAYS}")
    print(f"key         {'set' if os.getenv('GROQ_API_KEY') else 'MISSING'}")
