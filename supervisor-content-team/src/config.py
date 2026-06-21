"""Central configuration for the supervisor content team.

Loads API keys from the environment, names the two Gemini models used across
the graph, sets the hard step cap that guarantees the supervisor loop ends,
and exposes a single factory for building chat LLMs.

Inputs:  environment variables GEMINI_API_KEY, TAVILY_API_KEY (from .env).
Outputs: constants + make_llm() used by every node in the graph.
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load before any client is constructed (mandatory on Windows, where `uv run`
# does not inherit the shell's environment).
load_dotenv()

# --- API keys -------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# --- Models ---------------------------------------------------------------
# The supervisor routes often and only needs to pick the next agent, so it
# always runs on the cheap, low-latency lite model.
MODEL_SUPERVISOR: str = "gemini-3.1-flash-lite"

# Worker model for the four specialists. The free tier caps gemini-3.5-flash at
# 20 requests/day and one full run uses about four of them (~5 runs/day), while
# gemini-3.1-flash-lite has a much higher free allowance. Default to flash-lite
# so the project runs free out of the box. Flip this to True for higher-quality
# prose if you have gemini-3.5-flash quota or billing enabled.
USE_HIGH_QUALITY_WORKER: bool = False
MODEL_WORKER: str = (
    "gemini-3.5-flash" if USE_HIGH_QUALITY_WORKER else "gemini-3.1-flash-lite"
)

# --- Safety ---------------------------------------------------------------
# Hard cap on how many times the supervisor may route. Without it, an LLM that
# keeps asking for "one more revision" would loop forever.
MAX_STEPS: int = 12

# Tavily search breadth for the researcher.
TAVILY_MAX_RESULTS: int = 5


def require_keys() -> None:
    """Fail fast with a clear message if either API key is missing."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and paste "
            "your key from https://aistudio.google.com/apikey"
        )
    if not TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Copy .env.example to .env and paste "
            "your key from https://app.tavily.com"
        )


def make_llm(model: str, temperature: float = 0.3):
    """Build a Gemini chat model.

    Args:
        model: a Gemini model id (use MODEL_SUPERVISOR or MODEL_WORKER).
        temperature: lower for routing/factual work, higher for prose.

    Returns:
        A configured ChatGoogleGenerativeAI client.
    """
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
    )
