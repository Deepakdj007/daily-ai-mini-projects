"""Central configuration: env loading, model-per-task selection, API constants.

Inputs:  environment variables (.env) — GROQ_API_KEY, FOOTBALL_DATA_TOKEN,
         optional LANGSMITH_API_KEY.
Outputs: a `Settings` singleton plus `light_model()` / `heavy_model()` factories
         that hand back right-sized ChatGroq clients.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# load_dotenv must run before anything reads os.environ — on Windows `uv run`
# does not inject .env automatically.
load_dotenv()

# football-data.org REST API (free tier).
FOOTBALL_API_BASE = "https://api.football-data.org/v4"
WORLD_CUP_CODE = "WC"          # competition code for the FIFA World Cup
HTTP_TIMEOUT_SECONDS = 15.0    # per-request timeout for the async client
FREE_TIER_RPM = 10             # documented free-tier limit (requests / minute)

# Keyless football news RSS feeds (the always-on base for the news agent).
RSS_FEEDS: dict[str, str] = {
    "BBC": "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "Guardian": "https://www.theguardian.com/football/rss",
}

# TheSportsDB free, public demo key (no signup) — used for player bios.
SPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json"
SPORTSDB_KEY = "123"

# Right-sized models: a small fast model for routing, a large model for reasoning.
LIGHT_MODEL = "llama-3.1-8b-instant"
HEAVY_MODEL = "llama-3.3-70b-versatile"


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    groq_api_key: str | None
    football_token: str | None
    tavily_api_key: str | None
    langsmith_api_key: str | None

    @property
    def has_groq(self) -> bool:
        """True when a Groq key is present (required for any LLM call)."""
        return bool(self.groq_api_key)

    @property
    def has_football_token(self) -> bool:
        """True when a football-data.org token is present (required for live data)."""
        return bool(self.football_token)

    @property
    def has_tavily(self) -> bool:
        """True when a Tavily key is present (enables team-targeted news search)."""
        return bool(self.tavily_api_key)


def _enable_langsmith_if_configured(key: str | None) -> None:
    """Wire LangSmith tracing only when a key is supplied (otherwise stay silent)."""
    if not key:
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "worldcup-analyst")
    os.environ["LANGSMITH_API_KEY"] = key


def load_settings() -> Settings:
    """Read configuration from the environment and apply optional side effects."""
    settings = Settings(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        football_token=os.getenv("FOOTBALL_DATA_TOKEN"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
    )
    _enable_langsmith_if_configured(settings.langsmith_api_key)
    return settings


SETTINGS = load_settings()


def light_model() -> ChatGroq:
    """Small, fast model for routing and classification (cheap, low-latency)."""
    return ChatGroq(model=LIGHT_MODEL, api_key=SETTINGS.groq_api_key, temperature=0.0)


def agent_model() -> ChatGroq:
    """Model that drives the agents' tool loops.

    Uses the light (8b) model on purpose: tool selection + relaying facts is well
    within its ability, and it draws from a separate, far larger daily token
    budget than the 70b model — so a full briefing's many tool-loop calls stay
    free-tier-sustainable. The 70b model is reserved for the final synthesis.
    """
    return ChatGroq(model=LIGHT_MODEL, api_key=SETTINGS.groq_api_key, temperature=0.0)


def heavy_model() -> ChatGroq:
    """Large model for the final reader-facing synthesis (higher quality prose)."""
    return ChatGroq(model=HEAVY_MODEL, api_key=SETTINGS.groq_api_key, temperature=0.3)
