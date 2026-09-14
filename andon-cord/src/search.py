"""Keyless web search over DuckDuckGo.

This is the only tool the research team has, and it needs no API key or signup.

Inputs:  a query string.
Outputs: a list of {"title", "url", "snippet"} dicts, possibly empty.

Smoke test:  PYTHONPATH=. uv run python -m src.search
"""

import time

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from src.config import SEARCH_RESULTS, SNIPPET_CHARS

# Two researchers hit DuckDuckGo at the same instant, which occasionally earns a
# throttle. Retry with a growing pause rather than failing the whole run.
_MAX_ATTEMPTS: int = 3
_BACKOFF_SECONDS: float = 2.0


def web_search(query: str, k: int = SEARCH_RESULTS) -> list[dict]:
    """Search the web and return up to k results.

    Args:
        query: the search string.
        k: maximum number of results.

    Returns:
        A list of dicts with title, url and snippet. Returns an empty list if
        every attempt fails — a dead search must never raise, because an
        exception in one researcher fails the whole parallel superstep.
    """
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            raw = DDGS().text(query, max_results=k)
            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", "")[:SNIPPET_CHARS],
                }
                for item in raw
            ]
        except (RatelimitException, TimeoutException, DDGSException):
            if attempt == _MAX_ATTEMPTS:
                return []
            time.sleep(_BACKOFF_SECONDS * attempt)
        except Exception:
            # Any other transport failure is still not worth killing a run over.
            return []
    return []


def _demo() -> None:
    """Print live results so you can confirm search works before wiring agents."""
    results = web_search("UPI autopay mandate limit 2026")
    print(f"{len(results)} results\n")
    for result in results:
        print(f"- {result['title']}")
        print(f"  {result['url']}\n")


if __name__ == "__main__":
    _demo()
