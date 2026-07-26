"""Central configuration for the deep-research team.

Loads the Gemini key from the environment, names the models each agent uses,
sets the caps that keep the reflection loop finite, and exposes one factory
for building LLM clients.

Inputs:  environment variable GEMINI_API_KEY (from .env).
Outputs: constants + make_llm() used by every step in the workflow.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI

# Load .env BEFORE reading any variable. On Windows `uv run` does not inherit the
# shell's environment, so this call is mandatory and must come first.
load_dotenv()

# --- API key
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# --- Models. flash-lite has by far the largest free-tier allowance, and one full
# research run makes 8-14 model calls. Both roles default to it so the project
# runs free out of the box. Swap MODEL_WRITER to "gemini-3.5-flash" for better
# prose if you have the quota.
MODEL_FAST: str = "gemini-3.1-flash-lite"
MODEL_WRITER: str = "gemini-3.1-flash-lite"

# --- Research shape.
NUM_SUB_QUESTIONS: int = 4  # how many questions the planner splits the topic into
MAX_GAP_QUESTIONS: int = 3  # cap on extra questions the reflector may request
SEARCH_RESULTS: int = 5  # DuckDuckGo results fetched per sub-question

# How many researchers run at once. The @step decorator reads this at import
# time, so it is exposed as an env var to make the comparison easy to run:
#   RESEARCH_WORKERS=1 PYTHONPATH=. uv run python -m src.main "some topic"
RESEARCH_WORKERS: int = int(os.getenv("RESEARCH_WORKERS", "4"))

# --- Loop control. The reflector may send the team back out for more research,
# so it needs a hard ceiling. Round 1 is the initial plan; MAX_ROUNDS = 2 means
# at most one follow-up round before the writer runs no matter what.
MAX_ROUNDS: int = 2

# --- The workflow default timeout is 45 seconds, which a real research run blows
# straight through. This covers search latency plus every model call.
WORKFLOW_TIMEOUT: float = 600.0

# --- Paths.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = ROOT_DIR / "output"
REPORT_PATH: Path = OUTPUT_DIR / "report.md"


def require_keys() -> None:
    """Fail fast with a clear message if the Gemini key is missing."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and paste a free "
            "key from https://aistudio.google.com/apikey"
        )


def make_llm(model: str, temperature: float = 0.2) -> GoogleGenAI:
    """Build a Gemini client.

    GoogleGenAI reads GOOGLE_API_KEY from the environment by default. This project
    standardises on GEMINI_API_KEY like the rest of the series, so the key is
    always passed explicitly.

    Args:
        model: a Gemini model id (use MODEL_FAST or MODEL_WRITER).
        temperature: low for planning and fact extraction, higher for prose.

    Returns:
        A configured GoogleGenAI client with retries enabled for free-tier limits.
    """
    return GoogleGenAI(
        model=model,
        api_key=GEMINI_API_KEY,
        temperature=temperature,
        max_retries=3,
    )
