"""news_agent: a tool-using agent that reads team news for the next match.

It decides how to gather news — keyless RSS headlines and/or targeted Tavily
searches (it can phrase its own queries, e.g. for injuries or the opponent) — and
distills injuries, suspensions, momentum, and storylines.

Inputs:  AnalystState (uses team_name, opponent_name).
Outputs: partial state with one `findings` entry (and `missing` on failure).
"""

from __future__ import annotations

from app.agents.runner import run_agent_finding
from app.agents.tools import news_tools
from app.state import AnalystState

AGENT = "news_agent"
_TITLE = "News & Storylines"
_SYSTEM = (
    "You are a football news analyst. Use get_rss_headlines and search_news (you "
    "phrase the query) to gather recent news about the focus team and its next "
    "match. Distil injuries, suspensions, lineup/selection news, momentum, and "
    "storylines into 3-4 sentences. Ground every claim in what the tools return; "
    "never invent. If little is available, say so. No tool names in the answer."
)


async def news_node(state: AnalystState) -> dict:
    """Run the news agent over its tools and return a News & Storylines finding."""
    team = state.get("team_name") or "the focus team"
    opponent = state.get("opponent_name")
    matchup = f" ahead of facing {opponent}" if opponent else ""
    task = (f"Gather and summarise the latest news for {team}{matchup}. "
            "Use RSS first, then a targeted search for injuries or lineup news.")
    return await run_agent_finding(AGENT, _TITLE, news_tools(state), _SYSTEM, task)
