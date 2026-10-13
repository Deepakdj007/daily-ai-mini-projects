"""Web fallback: rewrite the question into a search query, then search DuckDuckGo.

Inputs:  the user's question
Outputs: a search query string, and web snippets tagged with their source URL
"""

import asyncio
import time

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException

from src.config import WEB_RESULTS
from src.llm import ask_text

REWRITE_SYSTEM = """You work at an Indian Railways help desk. Rewrite the user's question as a
short web search query of 4 to 10 keywords. Keep names, places and numbers.
If the question is about trains, include "Indian Railways" or "IRCTC".
Reply with the query only, no quotes."""


async def rewrite_query(question: str) -> str:
    """Turn a chatty question into keywords a search engine handles well."""
    query = await ask_text(REWRITE_SYSTEM, question)
    return query.strip().strip('"') or question


def _search(query: str) -> list[dict]:
    """Run the search, retrying twice when DuckDuckGo rate-limits us."""
    for attempt in range(3):
        try:
            return DDGS().text(query, max_results=WEB_RESULTS)
        except RatelimitException:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return []


async def web_search(query: str) -> list[str]:
    """Return web snippets as '<title>: <body> [WEB:<url>]'. Returns [] if the search fails."""
    try:
        # ddgs has no async client, so run it in a worker thread.
        results = await asyncio.to_thread(_search, query)
    except DDGSException as error:
        print(f"  web search failed: {error}")
        return []
    return [f"{r['title']}: {r['body']} [WEB:{r['href']}]" for r in results]


if __name__ == "__main__":
    question = "How much check-in baggage can I carry on an IndiGo domestic flight?"
    query = asyncio.run(rewrite_query(question))
    print(f"query: {query}")
    for snippet in asyncio.run(web_search(query)):
        print(f"  {snippet[:160]}")
