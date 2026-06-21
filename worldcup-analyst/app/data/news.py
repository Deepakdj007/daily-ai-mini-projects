"""Async, swappable news client: BBC + Guardian RSS plus optional Tavily search.

The ONLY place that touches news sources. Every method degrades into an
`ApiResult` so a feed going down never crashes the graph. Tavily runs only when a
key is configured; otherwise the keyless RSS feeds carry the news agent alone.

Inputs:  a focus team name.
Outputs: `ApiResult[list[NewsItem]]` — deduped, newest-first headlines.
"""

from __future__ import annotations

import asyncio
import re

import feedparser
import httpx

from app.config import HTTP_TIMEOUT_SECONDS, RSS_FEEDS, SETTINGS
from app.data.models import NewsItem
from app.data.results import ApiResult, explain_error, is_transient

_TAG_RE = re.compile(r"<[^>]+>")
_USER_AGENT = "worldcup-analyst/1.0 (+news-agent)"


def _clean(text: str | None, limit: int = 240) -> str | None:
    """Strip HTML tags, collapse whitespace, and truncate a summary string."""
    if not text:
        return None
    stripped = _TAG_RE.sub(" ", text)
    collapsed = " ".join(stripped.split())
    return collapsed[:limit] if collapsed else None


def _mentions(item: NewsItem, team: str) -> bool:
    """True when the headline or summary names the focus team."""
    needle = team.lower()
    return needle in (item.title or "").lower() or needle in (item.summary or "").lower()


class NewsClient:
    """Fetches football news from RSS feeds and (optionally) Tavily search."""

    def __init__(self) -> None:
        """Open one async HTTP client shared across the RSS feed requests."""
        self._http = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS, headers={"User-Agent": _USER_AGENT}
        )

    async def __aenter__(self) -> "NewsClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._http.aclose()

    async def _one_feed(self, source: str, url: str) -> list[NewsItem]:
        """Fetch and parse a single RSS feed into NewsItems (newest-first)."""
        resp = await self._http.get(url)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)  # CPU-only parse on bytes
        return [
            NewsItem(
                title=(entry.get("title") or "").strip(),
                summary=_clean(entry.get("summary")),
                source=source,
                url=entry.get("link"),
                published=entry.get("published"),
            )
            for entry in parsed.entries
            if entry.get("title")
        ]

    async def fetch_rss(self, team: str, limit: int = 6) -> ApiResult[list[NewsItem]]:
        """Pull both feeds concurrently and keep items that name the team."""
        try:
            feeds = await asyncio.gather(
                *(self._one_feed(s, u) for s, u in RSS_FEEDS.items()),
                return_exceptions=True,
            )
            items: list[NewsItem] = []
            for feed in feeds:
                if isinstance(feed, list):
                    items.extend(feed)
            relevant = [i for i in items if _mentions(i, team)]
            return ApiResult(data=relevant[:limit])
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            return ApiResult(error=explain_error(exc), transient=is_transient(exc))

    async def fetch_tavily(self, team: str, limit: int = 6,
                           opponent: str | None = None,
                           query: str | None = None) -> ApiResult[list[NewsItem]]:
        """Team-targeted fresh news via Tavily; skipped when no key is set.

        Pass `query` to search a specific phrase; otherwise a default team/opponent
        query is built.
        """
        if not SETTINGS.has_tavily:
            return ApiResult(error="Tavily not configured")
        try:
            from tavily import AsyncTavilyClient  # local import: optional dependency

            client = AsyncTavilyClient(api_key=SETTINGS.tavily_api_key)
            if not query:
                matchup = f" vs {opponent}" if opponent else ""
                query = (f"{team}{matchup} FIFA World Cup 2026 team news injury "
                         "lineup form")
            resp = await client.search(query, max_results=limit, topic="news")
            items = [
                NewsItem(
                    title=(r.get("title") or "").strip(),
                    summary=_clean(r.get("content")),
                    source="Tavily",
                    url=r.get("url"),
                )
                for r in resp.get("results", [])
                if r.get("title")
            ]
            return ApiResult(data=items)
        except Exception as exc:  # noqa: BLE001
            return ApiResult(error=explain_error(exc), transient=is_transient(exc))

    async def gather_news(self, team: str, limit: int = 6,
                          opponent: str | None = None) -> ApiResult[list[NewsItem]]:
        """Merge RSS + Tavily, dedupe by title, and cap — RSS alone is enough."""
        rss, tavily = await asyncio.gather(
            self.fetch_rss(team, limit),
            self.fetch_tavily(team, limit, opponent),
        )
        merged: list[NewsItem] = []
        if rss.ok:
            merged.extend(rss.data or [])
        if tavily.ok:
            merged.extend(tavily.data or [])

        seen: set[str] = set()
        deduped: list[NewsItem] = []
        for item in merged:
            key = (item.title or "").lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(item)

        if not deduped:
            # Surface the most informative reason for an empty result. Only a
            # transient RSS failure is worth a retry (Tavily is best-effort).
            reason = rss.error or tavily.error or f"no recent news naming {team}"
            return ApiResult(error=reason, transient=rss.transient)
        return ApiResult(data=deduped[:limit])
