"""Async football-data.org client — the only place that touches this API.

Swap this module to change data sources; the agents depend only on its typed
return values. Every public method degrades gracefully: on a missing token,
timeout, rate limit (429), or bad payload it returns an `ApiResult` carrying an
error string instead of raising, so one failed call never crashes the graph.

Inputs:  competition / team identifiers.
Outputs: `ApiResult[...]` wrapping validated pydantic models.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import httpx

from app.config import (
    FOOTBALL_API_BASE,
    HTTP_TIMEOUT_SECONDS,
    SETTINGS,
    WORLD_CUP_CODE,
)
from app.data.models import GroupStanding, Match, Scorer, TeamRef
from app.data.results import ApiResult, explain_error, is_transient

# Module-level TTL cache shared across client instances. Parallel agents often
# request the same endpoint (e.g. standings, team resolution); caching keeps the
# free tier's 10 req/min limit from biting and speeds repeat runs.
_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL_SECONDS = 90.0


def _cache_key(path: str, params: dict | None) -> str:
    """Stable key from a path + its sorted params."""
    items = sorted((params or {}).items())
    return path + "?" + "&".join(f"{k}={v}" for k, v in items)


class FootballDataClient:
    """Thin async wrapper over the football-data.org v4 REST API."""

    def __init__(self) -> None:
        """Build the client; absence of a token is reported per-call, not here."""
        self._token = SETTINGS.football_token
        headers = {"X-Auth-Token": self._token} if self._token else {}
        self._http = httpx.AsyncClient(
            base_url=FOOTBALL_API_BASE,
            headers=headers,
            timeout=HTTP_TIMEOUT_SECONDS,
        )

    async def __aenter__(self) -> "FootballDataClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._http.aclose()

    async def _get(self, path: str, params: dict | None = None) -> dict:
        """GET a path (TTL-cached), raising on a missing token or HTTP error."""
        if not self._token:
            raise RuntimeError("FOOTBALL_DATA_TOKEN is not set")
        key = _cache_key(path, params)
        cached = _CACHE.get(key)
        if cached and time.monotonic() - cached[0] < _CACHE_TTL_SECONDS:
            return cached[1]
        resp = await self._http.get(path, params=params)
        resp.raise_for_status()
        data = resp.json()
        _CACHE[key] = (time.monotonic(), data)
        return data

    async def recent_matches(self, limit: int = 6) -> ApiResult[list[Match]]:
        """Most recent finished + in-play World Cup matches (live scores feed)."""
        try:
            payload = await self._get(
                f"/competitions/{WORLD_CUP_CODE}/matches",
                params={"status": "FINISHED"},
            )
            matches = [Match.model_validate(m) for m in payload.get("matches", [])]
            matches.sort(key=lambda m: m.utc_date or "", reverse=True)
            return ApiResult(data=matches[:limit])
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            return ApiResult(error=explain_error(exc), transient=is_transient(exc))

    async def resolve_team_id(self, name: str) -> ApiResult[TeamRef]:
        """Map a team name/TLA to its World Cup team id (case-insensitive)."""
        try:
            payload = await self._get(f"/competitions/{WORLD_CUP_CODE}/teams")
            needle = name.strip().lower()
            for raw in payload.get("teams", []):
                team = TeamRef.model_validate(raw)
                haystack = {
                    (team.name or "").lower(),
                    (team.short_name or "").lower(),
                    (team.tla or "").lower(),
                }
                if needle in haystack or any(needle in h for h in haystack if h):
                    return ApiResult(data=team)
            return ApiResult(error=f"team '{name}' not found in the WC team list")
        except Exception as exc:  # noqa: BLE001
            return ApiResult(error=explain_error(exc), transient=is_transient(exc))

    async def team_form(self, team_id: int, limit: int = 5) -> ApiResult[list[Match]]:
        """A team's last-N finished matches across the World Cup (form trend)."""
        try:
            payload = await self._get(
                f"/teams/{team_id}/matches",
                params={"status": "FINISHED", "competitions": WORLD_CUP_CODE,
                        "limit": limit},
            )
            matches = [Match.model_validate(m) for m in payload.get("matches", [])]
            matches.sort(key=lambda m: m.utc_date or "", reverse=True)
            return ApiResult(data=matches[:limit])
        except Exception as exc:  # noqa: BLE001
            return ApiResult(error=explain_error(exc), transient=is_transient(exc))

    async def standings(self) -> ApiResult[list[GroupStanding]]:
        """Current World Cup group tables (TOTAL type only)."""
        try:
            payload = await self._get(f"/competitions/{WORLD_CUP_CODE}/standings")
            groups = [
                GroupStanding.model_validate(s)
                for s in payload.get("standings", [])
                if s.get("type") == "TOTAL"
            ]
            return ApiResult(data=groups)
        except Exception as exc:  # noqa: BLE001
            return ApiResult(error=explain_error(exc), transient=is_transient(exc))

    async def top_scorers(self, limit: int = 10) -> ApiResult[list[Scorer]]:
        """Competition-level top scorers (free tier: no per-match player stats)."""
        try:
            payload = await self._get(
                f"/competitions/{WORLD_CUP_CODE}/scorers",
                params={"limit": limit},
            )
            scorers = [Scorer.from_api(s) for s in payload.get("scorers", [])]
            return ApiResult(data=scorers)
        except Exception as exc:  # noqa: BLE001
            return ApiResult(error=explain_error(exc), transient=is_transient(exc))

    async def next_match(self, team_id: int, horizon_days: int = 60
                         ) -> ApiResult[Match]:
        """The team's soonest upcoming World Cup fixture (the 'next match')."""
        try:
            today = date.today()
            payload = await self._get(
                f"/teams/{team_id}/matches",
                params={"competitions": WORLD_CUP_CODE,
                        "dateFrom": today.isoformat(),
                        "dateTo": (today + timedelta(days=horizon_days)).isoformat()},
            )
            upcoming = [
                m for m in (Match.model_validate(x) for x in payload.get("matches", []))
                if m.status not in {"FINISHED", "IN_PLAY", "PAUSED"}
            ]
            upcoming.sort(key=lambda m: m.utc_date or "")
            if not upcoming:
                return ApiResult(error="no upcoming World Cup fixture found")
            return ApiResult(data=upcoming[0])
        except Exception as exc:  # noqa: BLE001
            return ApiResult(error=explain_error(exc), transient=is_transient(exc))
