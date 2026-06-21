"""player_agent: a tool-using agent that names and scouts the key player.

It decides to pull the scorers list, find the focus team's scorer, and (when it
wants) enrich them with a TheSportsDB bio. If the focus team has no scorer yet it
says so plainly and never name-drops a rival.

Inputs:  AnalystState (uses team_name).
Outputs: partial state with one `findings` entry (and `missing` on failure).
"""

from __future__ import annotations

from app.agents.runner import run_agent_finding
from app.agents.tools import player_tools
from app.state import AnalystState

AGENT = "player_agent"
_TITLE = "Key Player"
_SYSTEM = (
    "You are a player scout. Use get_top_scorers to find the FOCUS TEAM's leading "
    "scorer, then optionally use get_player_profile to add their position/club. "
    "Write 2-3 sentences scouting that player's threat in the next match. If the "
    "focus team has NO scorer yet, say so plainly and do NOT name or scout any "
    "player from another team. No tool names in the answer; no invented stats."
)


async def player_node(state: AnalystState) -> dict:
    """Run the player agent over its tools and return a Key Player finding."""
    team = state.get("team_name") or "the focus team"
    task = (f"Identify and scout {team}'s key player for their next match. "
            "Find their top scorer first, then enrich the profile if useful.")
    return await run_agent_finding(AGENT, _TITLE, player_tools(state), _SYSTEM, task)
