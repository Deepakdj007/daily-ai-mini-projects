"""Keyless web search over DuckDuckGo.

This is the only tool the research team has, and it needs no API key or signup.

`ddgs` is a synchronous library with no async client, so the call is pushed onto
a worker thread. That keeps four concurrent researchers from blocking the single
event loop the workflow runs on.

Inputs:  a query string.
Outputs: a list of Source objects (title, url, snippet).

Smoke test:  PYTHONPATH=. uv run python -m src.search
"""

import asyncio
import time

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from src.events import Source

# Four researchers hitting DuckDuckGo at once will occasionally get throttled.
# Retry a few times with a growing pause rather than failing the whole run.
_MAX_ATTEMPTS: int = 3
_BACKOFF_SECONDS: float = 2.0


def _search_sync(query: str, k: int) -> list[Source]:
    """Run one blocking DuckDuckGo search, retrying on rate limits.

    Args:
        query: the search string.
        k: maximum number of results to return.

    Returns:
        Up to k Source objects. Returns an empty list if every attempt fails,
        so one dead search never kills the whole research run.
    """
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            raw = DDGS().text(query, max_results=k)
            return [
                Source(
                    title=item.get("title", ""),
                    url=item.get("href", ""),
                    snippet=item.get("body", ""),
                )
                for item in raw
            ]
        except (RatelimitException, TimeoutException, DDGSException):
            if attempt == _MAX_ATTEMPTS:
                return []
            time.sleep(_BACKOFF_SECONDS * attempt)
    return []


async def web_search(query: str, k: int = 5) -> list[Source]:
    """Search the web without blocking the event loop.

    Args:
        query: the search string.
        k: maximum number of results.

    Returns:
        A list of Source objects, possibly empty.
    """
    return await asyncio.to_thread(_search_sync, query, k)


def _demo() -> None:
    """Print a few live results so you can confirm search works on its own."""
    results = _search_sync("solid-state battery commercialization 2026", 5)
    print(f"{len(results)} results\n")
    for source in results:
        print(f"- {source.title}")
        print(f"  {source.url}\n")


if __name__ == "__main__":
    _demo()
