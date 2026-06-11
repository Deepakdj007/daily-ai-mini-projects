"""Configuration and constants for the smolagents code agent.

Loads the Groq API key from .env (mandatory on Windows, where `uv run` does not
inherit the .env automatically) and exposes the model name, the step limit, and
the list of imports the agent's generated Python is allowed to use.
"""

import os

from dotenv import load_dotenv

# Load .env before anything reads GROQ_API_KEY or builds a model client.
load_dotenv()

# Groq's free-tier model, addressed through LiteLLM's native "groq/" prefix.
# The prefix routes via LiteLLM's Groq provider directly — not the OpenAI-
# compatible endpoint, which mishandles smolagents' system messages.
MODEL = "groq/llama-3.3-70b-versatile"

# Hard cap on how many write-and-run cycles the agent may take for one problem.
# Each step is one model round-trip, so this bounds cost and stops runaway loops.
MAX_STEPS = 8

# Modules the agent's generated code is allowed to import. smolagents blocks
# every import by default; this allowlist is what lets the agent reach for the
# standard data and math toolkit while keeping os/sys/subprocess out of reach.
AUTHORIZED_IMPORTS = [
    "pandas",
    "numpy",
    "math",
    "statistics",
    "datetime",
    "json",
    "re",
    "collections",
    "itertools",
]


def get_api_key() -> str:
    """Return the Groq API key, or raise a clear error if it is missing."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste your "
            "key from https://console.groq.com/keys"
        )
    return key
