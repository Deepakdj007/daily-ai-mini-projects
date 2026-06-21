"""CLI entry point for the supervisor content team.

Takes a topic on the command line, runs the graph end to end, prints the agent
handoff log and the finished article + SEO package, and saves the article to
output/<slug>.md.

Run:
    PYTHONPATH=. uv run python -m src.main "How RAG improves LLM accuracy"
"""

import sys
from pathlib import Path

from google.genai.errors import ClientError

from src.config import require_keys
from src.graph import build_graph

# Gemini writes em-dashes and curly quotes; force UTF-8 so the Windows console
# prints them instead of mangling them into question marks.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_TOPIC = "How retrieval-augmented generation improves LLM accuracy"
OUTPUT_DIR = Path("output")


def run(topic: str) -> None:
    """Invoke the graph for one topic and report the results."""
    require_keys()
    graph = build_graph()

    print(f"\nTopic: {topic}\n" + "=" * 60)
    print("\nHANDOFF LOG")
    print("-" * 60)

    # Stream in "values" mode: each step yields the full accumulated state, so
    # we can print each handoff the moment it happens instead of waiting for the
    # whole pipeline to finish. recursion_limit is a final backstop; MAX_STEPS
    # caps supervisor turns well below it.
    final: dict = {}
    printed = 0
    for state in graph.stream(
        {"topic": topic}, {"recursion_limit": 50}, stream_mode="values"
    ):
        final = state
        history = state.get("history", [])
        for line in history[printed:]:
            print(f"  {line}", flush=True)
        printed = len(history)

    article = final.get("edited") or final.get("draft", "(no article produced)")
    seo = final.get("seo", {})

    print("\nSEO PACKAGE")
    print("-" * 60)
    print(f"  Title : {seo.get('title', '')}")
    print(f"  Meta  : {seo.get('meta_description', '')}")
    print(f"  Words : {', '.join(seo.get('keywords', []))}")
    print(f"  Slug  : {seo.get('slug', '')}")

    print("\nARTICLE")
    print("-" * 60)
    print(article)

    _save(article, seo)


def _save(article: str, seo: dict) -> None:
    """Write the finished article to output/<slug>.md with an SEO header."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    slug = seo.get("slug") or "article"
    path = OUTPUT_DIR / f"{slug}.md"
    header = (
        f"# {seo.get('title', '')}\n\n"
        f"> {seo.get('meta_description', '')}\n\n"
        f"**Keywords:** {', '.join(seo.get('keywords', []))}\n\n---\n\n"
    )
    path.write_text(header + article, encoding="utf-8")
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TOPIC
    try:
        run(topic)
    except ClientError as exc:
        # The free tier caps gemini-3.5-flash at 20 requests/day; one run uses
        # about four. Show a clean message instead of a raw traceback.
        if exc.code == 429:
            print(
                "\nGemini quota hit (HTTP 429). The free tier allows ~20 "
                "gemini-3.5-flash requests/day and each run uses about four.\n"
                "Wait for the daily reset, use a different API key, or set the "
                "worker model to gemini-3.1-flash-lite in src/config.py.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
