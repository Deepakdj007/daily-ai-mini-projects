"""CLI entry point: take a query, run the graph, print the match-day briefing.

Run:  PYTHONPATH=. uv run python app/main.py "Give me a briefing on Brazil's next match"
Add  --verbose to print each agent's tool calls.

If no query is passed a sample one is used so the graph can be smoke-tested.
"""

from __future__ import annotations

import asyncio
import sys

from app.agents.runner import set_verbose
from app.config import SETTINGS
from app.graph import build_graph
from app.state import AnalystState

SAMPLE_QUERY = "Give me a briefing on Brazil's next match"


def _read_query(argv: list[str]) -> str:
    """Join CLI args (minus flags) into the query, or fall back to the sample."""
    words = [a for a in argv[1:] if not a.startswith("--")]
    return " ".join(words).strip() or SAMPLE_QUERY


def _preflight() -> None:
    """Print clear warnings for missing keys without aborting the run."""
    if not SETTINGS.has_groq:
        print("WARNING: GROQ_API_KEY is not set — LLM analysis will be skipped.\n")
    if not SETTINGS.has_football_token:
        print("WARNING: FOOTBALL_DATA_TOKEN is not set — live data will be "
              "unavailable; workers will degrade gracefully.\n")


async def run(query: str) -> str:
    """Execute the full graph end-to-end for one query and return the briefing.

    Prints the resolved next fixture (when found) before the briefing body.
    """
    graph = build_graph()
    initial: AnalystState = {"query": query, "retries": 0,
                             "findings": [], "missing": []}
    final = await graph.ainvoke(initial, config={"recursion_limit": 25})
    if final.get("next_match"):
        print(f"NEXT MATCH: {final['next_match']}\n")
    return final.get("briefing") or "(no briefing produced)"


def main() -> None:
    """Parse args, run the analyst, and print the briefing."""
    # Keep curly quotes / accented names readable in the Windows console.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    set_verbose("--verbose" in sys.argv)
    query = _read_query(sys.argv)
    _preflight()
    print(f"QUERY: {query}\n" + "=" * 60)
    briefing = asyncio.run(run(query))
    print(briefing)
    print("=" * 60)


if __name__ == "__main__":
    main()
