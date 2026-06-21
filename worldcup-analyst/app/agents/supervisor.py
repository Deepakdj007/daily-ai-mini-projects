"""Supervisor / router node: classify the query and decide which workers to run.

On the first pass it uses the light model to (a) extract the focus team and
(b) pick the relevant worker jobs, then resolves the team's id and next fixture
ONCE and shares them via state so workers don't each re-resolve the team. On a
retry pass (findings already exist) it re-dispatches only the workers that
previously failed, and bumps the counter so the fan-out can loop at most once.

Inputs:  AnalystState (needs `query`; on retry also `findings` + `retries`).
Outputs: partial state — `jobs`, `team_name`, context fields, `retries`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import light_model
from app.data.client import FootballDataClient
from app.state import AnalystState, Finding, latest_findings

ALL_JOBS = ["matchup_agent", "player_agent", "news_agent"]


class _Route(BaseModel):
    """Structured router decision returned by the light model."""

    team_name: str | None = Field(
        default=None, description="Focus team mentioned in the query, else null"
    )
    jobs: list[str] = Field(
        default_factory=list, description="Subset of worker job names to run"
    )


_SYSTEM = (
    "You route a FIFA World Cup analyst. Two tasks:\n"
    "1. team_name: the country/team the user focuses on, or null. Examples: "
    "'Brazil's next match' -> 'Brazil'; 'How is Argentina doing?' -> 'Argentina'; "
    "'Who is the top scorer?' -> null.\n"
    "2. jobs: which specialist agents to run from this exact list:\n"
    f"   {ALL_JOBS}\n"
    "   - matchup_agent: form (focus + opponent) and group standings\n"
    "   - player_agent: the focus team's key player / scorer read\n"
    "   - news_agent: latest team news, injuries, and storylines\n"
    "Rule: for a general 'briefing' or 'next match' request, return ALL jobs. "
    "For a narrow question, return only the relevant ones. If a team is named, "
    "ALWAYS include matchup_agent and news_agent."
)

# Words that signal the user wants a full briefing -> fan out to every agent.
_BRIEFING_HINTS = ("brief", "next match", "preview", "everything", "overview")


def _failed_jobs(findings: list[Finding]) -> list[str]:
    """Worker names whose latest Finding was not ok (used to drive a retry)."""
    return [f.agent for f in latest_findings(findings) if not f.ok]


def _refine(query: str, jobs: list[str], team: str | None) -> tuple[list[str], str | None]:
    """Apply deterministic guardrails on top of the LLM's routing decision."""
    team = team or _naive_team(query)
    selected = [j for j in jobs if j in ALL_JOBS]
    lowered = query.lower()
    if any(hint in lowered for hint in _BRIEFING_HINTS):
        selected = list(ALL_JOBS)               # full briefing -> all workers
    if team:                                    # a named team always gets matchup + news
        for job in ("matchup_agent", "news_agent"):
            if job not in selected:
                selected.append(job)
    return (selected or list(ALL_JOBS)), team


async def _resolve_context(team: str) -> dict:
    """Resolve the team's id and next fixture once; degrade quietly on failure."""
    context: dict = {"team_id": None, "opponent_name": None,
                     "opponent_id": None, "next_match": None}
    async with FootballDataClient() as client:
        resolved = await client.resolve_team_id(team)
        if not resolved.ok or resolved.data.id is None:
            return context
        team_id = resolved.data.id
        context["team_id"] = team_id

        fixture = await client.next_match(team_id)
    if fixture.ok:
        match = fixture.data
        opponent = match.opponent_of(team_id)
        context["opponent_name"] = opponent.label() if opponent else None
        context["opponent_id"] = opponent.id if opponent else None
        context["next_match"] = match.fixture_line()
    return context


async def supervisor_node(state: AnalystState) -> dict:
    """Classify + resolve context (first pass) or pick failed jobs (retry pass)."""
    existing = state.get("findings") or []
    if existing:  # retry pass — only re-run what failed
        return {"jobs": _failed_jobs(existing), "retries": state.get("retries", 0) + 1}

    query = state["query"]
    router = light_model().with_structured_output(_Route)
    try:
        decision: _Route = await router.ainvoke(
            [("system", _SYSTEM), ("human", query)]
        )
        jobs, team = _refine(query, decision.jobs, decision.team_name)
    except Exception:  # noqa: BLE001 — never let routing crash the run
        jobs, team = list(ALL_JOBS), _naive_team(query)

    context = await _resolve_context(team) if team else {}
    return {"jobs": jobs, "team_name": team, "retries": 0, **context}


# Common stop-words to skip when guessing a team name from capitalization.
_STOP = {
    "give", "me", "a", "an", "the", "how", "who", "what", "is", "are", "tell",
    "show", "list", "get", "find", "display", "latest", "recent", "current",
    "please", "of", "on", "for", "next", "match", "do", "does", "can",
}


def _naive_team(query: str) -> str | None:
    """Heuristic team guess: first capitalized token that isn't a sentence start."""
    tokens = query.replace("?", " ").replace("'s", " ").split()
    for token in tokens:
        clean = token.strip(".,")
        if clean[:1].isupper() and len(clean) > 2 and clean.lower() not in _STOP:
            return clean
    return None
