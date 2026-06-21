"""Async TheSportsDB client — free player bios (the only place that touches it).

Uses the public demo key (no signup). The free key returns up to a handful of
results per query; we use per-player search (`searchplayers.php`), which is
reliable, to enrich a known scorer with position / club / nationality.

Inputs:  a player name.
Outputs: `ApiResult[PlayerProfile]` — degrades to an error string, never raises.
"""

from __future__ import annotations

from datetime import date

import httpx

from app.config import HTTP_TIMEOUT_SECONDS, SPORTSDB_BASE, SPORTSDB_KEY
from app.data.models import PlayerProfile
from app.data.results import ApiResult, explain_error, is_transient


def _age_from(born: str | None) -> int | None:
    """Compute an age in years from a YYYY-MM-DD birth date, if parseable."""
    if not born or len(born) < 10:
        return None
    try:
        b = date.fromisoformat(born[:10])
    except ValueError:
        return None
    today = date.today()
    return today.year - b.year - ((today.month, today.day) < (b.month, b.day))


class SportsDBClient:
    """Thin async wrapper over TheSportsDB v1 free API."""

    def __init__(self) -> None:
        """Open one async HTTP client against the free-key base URL."""
        self._http = httpx.AsyncClient(
            base_url=f"{SPORTSDB_BASE}/{SPORTSDB_KEY}",
            timeout=HTTP_TIMEOUT_SECONDS,
        )

    async def __aenter__(self) -> "SportsDBClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._http.aclose()

    async def player_profile(self, name: str) -> ApiResult[PlayerProfile]:
        """Look up a player's bio by name (first/best match)."""
        try:
            resp = await self._http.get("/searchplayers.php", params={"p": name})
            resp.raise_for_status()
            players = resp.json().get("player") or []
            if not players:
                return ApiResult(error=f"no TheSportsDB profile for '{name}'")
            raw = players[0]
            return ApiResult(data=PlayerProfile(
                name=raw.get("strPlayer"),
                position=raw.get("strPosition"),
                club=raw.get("strTeam"),
                nationality=raw.get("strNationality"),
                age=_age_from(raw.get("dateBorn")),
            ))
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            return ApiResult(error=explain_error(exc), transient=is_transient(exc))
