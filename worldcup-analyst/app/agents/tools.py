"""LangChain tools that expose the data layer to the agents.

Tools are built per-run by small factories that close over the resolved context
(team_id, opponent) in `AnalystState`, so the model calls e.g. `get_team_form()`
with no arguments and the tool already knows which team. Every tool returns a
compact STRING (token-cheap) and degrades to a readable message, never raising.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from app.data.client import FootballDataClient
from app.data.models import GroupStanding, Match, Scorer
from app.data.news import NewsClient
from app.data.sportsdb import SportsDBClient
from app.state import AnalystState


def _result_letter(match: Match, team_id: int) -> str:
    """W / D / L for `team_id` in a finished match, else '-'."""
    fh, fa = match.score.full_time.home, match.score.full_time.away
    if fh is None or fa is None:
        return "-"
    scored, conceded = (fh, fa) if match.home_team.id == team_id else (fa, fh)
    return "W" if scored > conceded else "D" if scored == conceded else "L"


def _form_str(label: str, team_id: int, matches: list[Match]) -> str:
    """Compact 'Label form W D L | scoreline; scoreline' string."""
    letters = " ".join(_result_letter(m, team_id) for m in matches)
    lines = "; ".join(m.scoreline() for m in matches)
    return f"{label} form (newest first): {letters} | {lines}"


def _table_str(group: GroupStanding) -> str:
    """Compact one-line-per-row group table."""
    rows = "; ".join(
        f"{r.position}. {r.team.label()} {r.points}pts "
        f"(P{r.played_games} {r.won}-{r.draw}-{r.lost} GD{r.goal_difference})"
        for r in group.table
    )
    return f"Group {group.group or '?'}: {rows}"


def _scorers_str(scorers: list[Scorer], team: str | None) -> str:
    """Top scorers plus an explicit note on the focus team's scorers."""
    top = ", ".join(f"{s.player_name} ({s.team_name}) {s.goals}" for s in scorers[:8])
    focus = ""
    if team:
        mine = [s for s in scorers if s.team_name and team.lower() in s.team_name.lower()]
        focus = (f" | {team} scorers: "
                 + (", ".join(f"{s.player_name} {s.goals}" for s in mine) or "none yet"))
    return f"Top scorers: {top}{focus}"


def matchup_tools(state: AnalystState) -> list[BaseTool]:
    """Tools for the matchup agent: focus form, opponent form, group table."""
    team_id = state.get("team_id")
    opp_id = state.get("opponent_id")
    team = state.get("team_name") or "the focus team"
    opp = state.get("opponent_name") or "the opponent"

    @tool
    async def get_team_form() -> str:
        """Get the focus team's last-5 World Cup results and W/D/L form."""
        if team_id is None:
            return f"{team}'s team id is unknown; form unavailable."
        async with FootballDataClient() as c:
            r = await c.team_form(team_id, 5)
        if not r.ok:
            return f"form unavailable: {r.error}"
        if not r.data:
            return f"{team} has no finished World Cup matches yet."
        return _form_str(team, team_id, r.data)

    @tool
    async def get_opponent_form() -> str:
        """Get the next opponent's last-5 World Cup results and W/D/L form."""
        if opp_id is None:
            return "the next opponent is unknown; opponent form unavailable."
        async with FootballDataClient() as c:
            r = await c.team_form(opp_id, 5)
        if not r.ok or not r.data:
            return f"{opp} form unavailable."
        return _form_str(opp, opp_id, r.data)

    @tool
    async def get_group_standings() -> str:
        """Get the current World Cup group table containing the focus team."""
        async with FootballDataClient() as c:
            r = await c.standings()
        if not r.ok:
            return f"standings unavailable: {r.error}"
        if not r.data:
            return "group standings are not published yet."
        groups = r.data
        chosen = groups[0]
        for g in groups:
            if any(team.lower() in row.team.label().lower() for row in g.table):
                chosen = g
                break
        return _table_str(chosen)

    return [get_team_form, get_opponent_form, get_group_standings]


def player_tools(state: AnalystState) -> list[BaseTool]:
    """Tools for the player agent: top scorers and a player bio lookup."""
    team = state.get("team_name")

    @tool
    async def get_top_scorers() -> str:
        """Get the World Cup top scorers and the focus team's scorers (if any)."""
        async with FootballDataClient() as c:
            r = await c.top_scorers(100)
        if not r.ok:
            return f"scorers unavailable: {r.error}"
        if not r.data:
            return "no World Cup scorers are published yet."
        return _scorers_str(r.data, team)

    @tool
    async def get_player_profile(name: str) -> str:
        """Get a player's bio (position, club, nationality, age) by full name."""
        async with SportsDBClient() as c:
            r = await c.player_profile(name)
        return r.data.line() if r.ok else f"no profile for {name}: {r.error}"

    return [get_top_scorers, get_player_profile]


def news_tools(state: AnalystState) -> list[BaseTool]:
    """Tools for the news agent: keyless RSS headlines and Tavily search."""
    team = state.get("team_name") or "the team"
    opp = state.get("opponent_name")

    @tool
    async def get_rss_headlines() -> str:
        """Get recent BBC/Guardian football headlines mentioning the focus team."""
        async with NewsClient() as c:
            r = await c.fetch_rss(team, 6)
        if not r.ok or not r.data:
            return f"no RSS headlines naming {team} right now."
        return " | ".join(i.line() for i in r.data)

    @tool
    async def search_news(query: str) -> str:
        """Search fresh web news for a specific query (injuries, lineup, form)."""
        async with NewsClient() as c:
            r = await c.fetch_tavily(team, 6, opponent=opp, query=query)
        if not r.ok or not r.data:
            return f"news search returned nothing ({r.error or 'no results'})."
        return " | ".join(i.line() for i in r.data)

    return [get_rss_headlines, search_news]
