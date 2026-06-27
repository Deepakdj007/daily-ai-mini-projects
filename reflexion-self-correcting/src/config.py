"""Central configuration for the Reflexion self-correcting agent.

Loads the API key, names the three Gemini models used across the loop (one per
tier of the generator/critic split), sets the quality threshold and the hard
revision cap that guarantees the loop ends, and exposes a single LLM factory.

Inputs:  environment variable GEMINI_API_KEY (from .env).
Outputs: constants + make_llm() used by every node in the graph.
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load before any client is constructed (mandatory on Windows, where `uv run`
# does not inherit the shell's environment).
load_dotenv()

# --- API key --------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# --- Models (the three-tier cost/quality split) ---------------------------
# Generator writes the email. It needs the strongest free-tier writer, so it
# runs on flash, not flash-lite.
MODEL_GENERATOR: str = "gemini-3.5-flash"

# Critic grades every single loop, so it must be cheap and fast. flash-lite has
# the largest free allowance and is plenty for scoring against a fixed rubric.
MODEL_CRITIC: str = "gemini-3.1-flash-lite"

# Adjudicator runs once at the end for a hard-reasoning final verdict. The pro
# reasoning model is the ideal judge, but the pro tier is billing-only on the
# Gemini API (free-tier quota is 0), so the adjudicator tries it first and falls
# back to the strongest *free* model below if the pro call is blocked. Readers
# with billing get the real third tier; everyone else still gets a verdict.
# Swap in "gemini-3.5-pro" here once it reaches general availability.
MODEL_ADJUDICATOR: str = "gemini-3.1-pro-preview"
MODEL_ADJUDICATOR_FALLBACK: str = "gemini-3.5-flash"

# --- Loop control ---------------------------------------------------------
# The critic passes a draft once its overall score reaches this (out of 10).
# Set high (9) so the loop actually iterates a few times instead of passing the
# first draft — that's what makes the diminishing-returns curve worth plotting.
PASS_THRESHOLD: int = 9

# Hard cap on revisions. Without it, a strict critic that never says "good
# enough" would loop forever. The loop always stops at or before this.
MAX_REVISIONS: int = 5


def require_keys() -> None:
    """Fail fast with a clear message if the API key is missing."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and paste "
            "your key from https://aistudio.google.com/apikey"
        )


def make_llm(model: str, temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Build a Gemini chat model.

    Args:
        model: a Gemini model id (use one of the MODEL_* constants).
        temperature: lower for grading/reasoning, higher for prose.

    Returns:
        A configured ChatGoogleGenerativeAI client.
    """
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
        # Free-tier models occasionally return a transient 503 ("high demand").
        # Retry a few times with backoff before failing the run.
        max_retries=4,
    )
