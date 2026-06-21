"""Pydantic schemas for football-data.org responses.

The API speaks camelCase; these models accept the raw camelCase keys via field
aliases and expose clean snake_case attributes. Every field that the free tier
may omit is Optional so validation never crashes on a thin payload.

Inputs:  raw JSON dicts from football-data.org.
Outputs: validated, typed objects the agents can reason over.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Camel(BaseModel):
    """Base that lets models be built from camelCase API keys or snake_case names."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class TeamRef(_Camel):
    """Minimal reference to a team as it appears nested inside matches/standings."""

    id: int | None = None
    name: str | None = None
    short_name: str | None = Field(default=None, alias="shortName")
    tla: str | None = None  # three-letter acronym, e.g. "BRA"

    def label(self) -> str:
        """Human-friendly team label, preferring the full name."""
        return self.name or self.short_name or self.tla or "Unknown"


class Score(_Camel):
    """Score for one phase of a match (home / away goals, may be null pre-match)."""

    home: int | None = None
    away: int | None = None


class MatchScore(_Camel):
    """The `score` block of a match: winner, full-time and half-time tallies."""

    winner: str | None = None  # HOME_TEAM | AWAY_TEAM | DRAW
    full_time: Score = Field(default_factory=Score, alias="fullTime")
    half_time: Score = Field(default_factory=Score, alias="halfTime")


class Match(_Camel):
    """A single World Cup match (scheduled, live, or finished)."""

    id: int | None = None
    utc_date: str | None = Field(default=None, alias="utcDate")
    status: str | None = None  # SCHEDULED | TIMED | IN_PLAY | FINISHED | ...
    matchday: int | None = None
    stage: str | None = None
    group: str | None = None
    venue: str | None = None
    home_team: TeamRef = Field(default_factory=TeamRef, alias="homeTeam")
    away_team: TeamRef = Field(default_factory=TeamRef, alias="awayTeam")
    score: MatchScore = Field(default_factory=MatchScore)

    def scoreline(self) -> str:
        """Render a compact 'BRA 2-1 SRB (FINISHED)' style line."""
        home, away = self.home_team.label(), self.away_team.label()
        fh, fa = self.score.full_time.home, self.score.full_time.away
        if fh is None or fa is None:
            return f"{home} vs {away} ({self.status or 'SCHEDULED'})"
        return f"{home} {fh}-{fa} {away} ({self.status or ''})".strip()

    def opponent_of(self, team_id: int) -> TeamRef | None:
        """Return the other team in the fixture relative to `team_id`."""
        if self.home_team.id == team_id:
            return self.away_team
        if self.away_team.id == team_id:
            return self.home_team
        return None

    def fixture_line(self) -> str:
        """Render an upcoming fixture as 'Home vs Away — 2026-06-21 [Venue]'."""
        day = (self.utc_date or "")[:10] or "date TBD"
        spot = f" [{self.venue}]" if self.venue else ""
        return f"{self.home_team.label()} vs {self.away_team.label()} — {day}{spot}"


class StandingRow(_Camel):
    """One row of a group table."""

    position: int | None = None
    team: TeamRef = Field(default_factory=TeamRef)
    played_games: int | None = Field(default=None, alias="playedGames")
    won: int | None = None
    draw: int | None = None
    lost: int | None = None
    points: int | None = None
    goals_for: int | None = Field(default=None, alias="goalsFor")
    goals_against: int | None = Field(default=None, alias="goalsAgainst")
    goal_difference: int | None = Field(default=None, alias="goalDifference")


class GroupStanding(_Camel):
    """A single group's table (type TOTAL), with its rows."""

    stage: str | None = None
    type: str | None = None  # TOTAL | HOME | AWAY
    group: str | None = None
    table: list[StandingRow] = Field(default_factory=list)


class Scorer(_Camel):
    """A top-scorer entry from the competition `scorers` endpoint.

    The free tier exposes goals/assists/penalties at the competition level; it
    does NOT expose per-player per-match statistics (a paid-tier feature).
    """

    player_name: str | None = None
    team_name: str | None = None
    goals: int | None = None
    assists: int | None = None
    penalties: int | None = None

    @classmethod
    def from_api(cls, raw: dict) -> "Scorer":
        """Flatten the nested {player, team, goals, ...} scorer payload."""
        player = raw.get("player") or {}
        team = raw.get("team") or {}
        return cls(
            player_name=player.get("name"),
            team_name=team.get("name"),
            goals=raw.get("goals"),
            assists=raw.get("assists"),
            penalties=raw.get("penalties"),
        )


class PlayerProfile(_Camel):
    """A player's bio from TheSportsDB (position / club / nationality / age)."""

    name: str | None = None
    position: str | None = None
    club: str | None = None
    nationality: str | None = None
    age: int | None = None

    def line(self) -> str:
        """Render a compact 'Name — Position, Club (Nationality, age 24)' line."""
        bits = [b for b in (self.position, self.club) if b]
        extra = ", ".join(b for b in (self.nationality,
                                      f"age {self.age}" if self.age else None) if b)
        tail = f" ({extra})" if extra else ""
        return f"{self.name or 'Unknown'} — {', '.join(bits)}{tail}".strip(" —")


class NewsItem(_Camel):
    """One news headline from an RSS feed or a Tavily search result."""

    title: str
    summary: str | None = None
    source: str | None = None      # e.g. "BBC", "Guardian", "Tavily"
    url: str | None = None
    published: str | None = None   # publish date as reported by the source

    def line(self) -> str:
        """Render a compact '[Source] Title — summary' line for the LLM prompt."""
        tag = f"[{self.source}] " if self.source else ""
        body = f" — {self.summary}" if self.summary else ""
        return f"{tag}{self.title}{body}"
