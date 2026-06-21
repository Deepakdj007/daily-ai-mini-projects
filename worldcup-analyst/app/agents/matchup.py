"""matchup_agent: a tool-using agent that previews the fixture from form + table.

It decides which of its tools to call — the focus team's form, the opponent's
form, the group standings — and weaves them into a comparative read of the next
match.

Inputs:  AnalystState (uses team_name, opponent_name, team_id, opponent_id).
Outputs: partial state with one `findings` entry (and `missing` on failure).
"""

from __future__ import annotations

from app.agents.runner import run_agent_finding
from app.agents.tools import matchup_tools
from app.state import AnalystState, Finding

AGENT = "matchup_agent"
_TITLE = "Form & Matchup"
_SYSTEM = (
    "You are a World Cup matchup analyst. Use your tools to gather the focus "
    "team's recent form, the opponent's form, and the group standings, then write "
    "3-4 sentences comparing the two sides and what it means for the upcoming "
    "match. Call each tool you need before answering. Do not mention tool names or "
    "invent data; if a tool returns nothing, work with what you have."
)


async def matchup_node(state: AnalystState) -> dict:
    """Run the matchup agent over its tools and return a Form & Matchup finding."""
    team = state.get("team_name")
    if not team:
        return {"findings": [Finding(AGENT, _TITLE,
                                     "No focus team was identified.", False)],
                "missing": [AGENT]}
    opponent = state.get("opponent_name") or "their next opponent"
    task = (f"Preview {team}'s next World Cup match against {opponent}. "
            "Gather form and standings with your tools, then give the comparison.")
    return await run_agent_finding(AGENT, _TITLE, matchup_tools(state), _SYSTEM, task)
