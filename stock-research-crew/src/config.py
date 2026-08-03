"""Central configuration for the stock research crew.

Loads the Gemini key, names the models each role uses, sets the free-tier
guard rails, and exposes one factory for building CrewAI LLM clients.

Inputs:  environment variables GEMINI_API_KEY, MODEL_ANALYST, MODEL_EDITOR,
         MAX_RPM (all but the key are optional, from .env).
Outputs: constants + make_llm() used by src/agents.py.
"""

import os
from pathlib import Path

from crewai import LLM
from dotenv import load_dotenv

# Load .env BEFORE reading any variable. On Windows `uv run` does not inherit the
# shell's environment, so this call is mandatory and must come first.
load_dotenv()

# CrewAI ships anonymous telemetry on by default. Set this before the first
# import of a Crew so nothing leaves the machine during a run.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

# On a machine that has never run CrewAI before, the first kickoff stops and asks
# "Would you like to view your execution traces? [y/N]" on stdin. In a CLI that is
# a 20-second stall in the middle of a run, so opt out up front.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

# --- API key
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# --- Models. flash-lite has the largest free-tier allowance and one full run makes
# roughly 10-15 model calls across five agents. Both roles default to it so the
# project runs free out of the box. If the analysts start fumbling their tool calls,
# raise MODEL_ANALYST to "gemini-3.5-flash" — that is the one knob worth turning.
MODEL_ANALYST: str = os.getenv("MODEL_ANALYST", "gemini-3.1-flash-lite")
MODEL_EDITOR: str = os.getenv("MODEL_EDITOR", "gemini-3.1-flash-lite")

# --- Free-tier guard rail. Four analysts fire at once, so the burst is what trips
# rate limits, not the total. CrewAI throttles every agent to this ceiling.
MAX_RPM: int = int(os.getenv("MAX_RPM", "12"))

# --- How much history the price agent looks at, and how many headlines the news
# agent reads. Both feed straight into token cost, so they stay small.
HISTORY_PERIOD: str = "1y"
NEWS_COUNT: int = 8

# --- Agent loop ceiling. Each analyst needs exactly one tool call, so anything
# above a handful means the model is thrashing and should be cut off.
MAX_ITER: int = 5

# --- Retries per agent. The free Gemini tier returns 503 "model is currently
# experiencing high demand" often enough that one retry is not enough. CrewAI
# retries the agent's whole step, which is why a run that hits one takes noticeably
# longer rather than failing.
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

# --- Paths.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = ROOT_DIR / "output"


def report_path(ticker: str) -> Path:
    """Where the finished note for one ticker lands."""
    return OUTPUT_DIR / f"{ticker.upper()}-report.md"


def require_keys() -> None:
    """Fail fast with a clear message if the Gemini key is missing."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and paste a free "
            "key from https://aistudio.google.com/apikey"
        )


def make_llm(model: str, temperature: float = 0.2) -> LLM:
    """Build a CrewAI LLM client pointed at the Gemini API.

    The "gemini/" prefix is what selects CrewAI's native google-genai provider.
    Without it CrewAI falls back to LiteLLM, which is no longer installable.
    The key is passed explicitly because CrewAI will otherwise accept either
    GOOGLE_API_KEY or GEMINI_API_KEY and the ambiguity is not worth debugging.

    Args:
        model: a bare Gemini model id, e.g. "gemini-3.1-flash-lite".
        temperature: low for analysis, higher for the editor's prose.

    Returns:
        A configured crewai.LLM.
    """
    return LLM(
        model=f"gemini/{model}",
        api_key=GEMINI_API_KEY,
        temperature=temperature,
    )
