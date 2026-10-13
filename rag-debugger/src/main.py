"""Command line for the RAG debugger.

Inputs:  `debug "<question>" [--top-k N] [--min-sim F]` or `demo`
Outputs: the Rich trace view in the terminal, and one trace JSON per query

Run from the project root:
    PYTHONPATH=. uv run python -m src.main demo
"""

import argparse
import asyncio
import json

from src import config
from src.diagnose import judge_agreement
from src.pipeline import run
from src.report import console, render


async def debug(question: str, top_k: int, min_sim: float) -> None:
    """Trace one question."""
    render(await run(question, top_k, min_sim))


async def demo() -> None:
    """Trace the four prepared questions, one per failure mode, and check the judge against hand labels."""
    queries = json.loads(config.QUERIES_PATH.read_text(encoding="utf-8"))
    summary = []
    for item in queries:
        console.rule(f"[bold]{item['label']}[/]  expect {item['expect']}")
        trace = await run(item["query"])
        agreement = judge_agreement(trace.rows, item["answer_pids"])
        render(trace, agreement)
        summary.append((item, trace.diagnosis["headline"], agreement))

    console.rule("[bold]summary[/]")
    for item, headline, agreement in summary:
        mark = "[green]as expected[/]" if headline == item["expect"] else "[red]UNEXPECTED[/]"
        console.print(f"{item['label']:<14} {headline:<17} {mark}   judge {agreement['agree']}/{agreement['total']}")
    agree = sum(a["agree"] for _, _, a in summary)
    total = sum(a["total"] for _, _, a in summary)
    console.print(f"judge agreed with the hand labels on {agree}/{total} passages")


def main() -> None:
    """Parse arguments, check the key, and run the chosen command."""
    parser = argparse.ArgumentParser(description="See why retrieval failed.")
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("debug", help="trace one question")
    one.add_argument("question")
    one.add_argument("--top-k", type=int, default=config.TOP_K)
    one.add_argument("--min-sim", type=float, default=config.MIN_SIMILARITY)
    commands.add_parser("demo", help="trace the four prepared failure modes")
    args = parser.parse_args()

    config.require_api_key()
    try:
        if args.command == "debug":
            asyncio.run(debug(args.question, args.top_k, args.min_sim))
        else:
            asyncio.run(demo())
    except KeyboardInterrupt:
        console.print("stopped")


if __name__ == "__main__":
    main()
