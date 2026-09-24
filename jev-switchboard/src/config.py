"""Settings for the switchboard: API keys, model IDs and routing thresholds.

Inputs:  .env in the project root (TYPESAFE_API_KEY, GROQ_API_KEY, optional gateway vars).
Outputs: constants imported by every other module.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # must run before any client reads its API key

# Jev: pinned to a version, because our thresholds are tuned against it.
# A gateway such as Vercel uses its own model name, set in .env.
JEV_MODEL = os.getenv("TYPESAFE_DEFAULT_MODEL", "jev-1.13.0")

# Groq: a small model for easy questions, a big one for judgment calls.
SMALL_LLM = "openai/gpt-oss-20b"
BIG_LLM = "openai/gpt-oss-120b"

# Routing thresholds. These encode our risk tolerance, not the model's.
MIN_INTENT_CONFIDENCE = 0.5   # below this, ask the customer to clarify
WANTS_HUMAN_CUTOFF = 0.7      # above this, hand over to a person
MAX_AUTO_COMPLEXITY = 1.0     # complaints scored above this go to a person
GUARD_HOLD_AT = 0.75          # measured: honest replies 0.36-0.53, invented facts 0.88+
GUARD_MIN_ANSWERED = 0.5      # below this, the reply missed the customer's question
