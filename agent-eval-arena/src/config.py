"""Environment loading, paths, and the model configs competing on the leaderboard.

Inputs: .env file (GROQ_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST).
Outputs: CONFIGS (the agent configs under test), PRICING (per-model $/M token rates),
and JUDGE_MODEL (the fixed referee model).
"""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TEST_SET_PATH = DATA_DIR / "test_set.json"
LEADERBOARD_PATH = ROOT_DIR / "leaderboard.md"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

JUDGE_MODEL = "llama-3.3-70b-versatile"


@dataclass(frozen=True)
class AgentConfig:
    """One competing agent configuration: a name paired with a Groq model."""

    name: str
    model: str
    temperature: float = 0.0


CONFIGS: list[AgentConfig] = [
    AgentConfig(name="llama-3.3-70b", model="llama-3.3-70b-versatile"),
    AgentConfig(name="llama-3.1-8b-instant", model="llama-3.1-8b-instant"),
    AgentConfig(name="gpt-oss-20b", model="openai/gpt-oss-20b"),
]

# USD per 1M tokens, (input, output). Source: https://groq.com/pricing (2026-07-04).
PRICING: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-20b": (0.075, 0.30),
}


def calculate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute the real-money cost of a call from accumulated token counts."""
    input_rate, output_rate = PRICING[model]
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
