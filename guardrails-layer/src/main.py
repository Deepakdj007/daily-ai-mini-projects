"""CLI entry point: demo, eval, swap, chat.

Inputs:  a subcommand
Outputs: terminal tables, output/results.json, output/leaderboard.md

Usage:
    PYTHONPATH=. uv run python src/main.py demo
    PYTHONPATH=. uv run python src/main.py eval
    PYTHONPATH=. uv run python src/main.py swap
    PYTHONPATH=. uv run python src/main.py chat
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from rich.console import Console

from src import cache, chat, evaluate, report
from src.agent import respond
from src.config import RESULTS_PATH
from src.ladder import CONFIGS_BY_NAME
from src.suites import ATTACKS, BENIGN, detect_harm
from src.swap import run_swap

console = Console()

SWAP_PATH = RESULTS_PATH.with_name("swap.json")


def fix_console() -> None:
    """Force UTF-8 on stdout.

    Model output contains characters cp1252 cannot encode - narrow no-break
    spaces and curly quotes - and the default Windows console raises
    UnicodeEncodeError on the first one.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# The last case runs on a weaker model on purpose. On gpt-oss-120b the system
# prompt refuses these attacks by itself and the output firewall never has to
# do anything, which makes for a demo that proves nothing.
DEMO_CASES = [
    ("exfil-dispute", None),
    ("overflow-512", None),
    ("emi-correction", None),
    ("verify-last4", "qwen/qwen3.6-27b"),
    # Last on purpose: the one attack the full stack does not stop, on the
    # strongest model. A demo that only shows the wins is an advert.
    ("action-indirect", "openai/gpt-oss-120b"),
]


async def demo() -> None:
    """Walk four cases through the full stack, the last on a weaker model."""
    cfg = CONFIGS_BY_NAME["full stack"]
    by_id = {c.case_id: c for c in ATTACKS + BENIGN}

    for case_id, model in DEMO_CASES:
        case = by_id[case_id]
        label = f" on {model}" if model else ""
        console.rule(f"[bold]{case.case_id}[/bold] ({case.family}){label}")
        console.print(f"[cyan]user[/cyan] {case.text[:160]}...\n")

        turn = await respond(case.text, cfg, pinned=case.pinned, chat_model=model)

        # Showing the draft matters: it is the difference between "the model
        # behaved" and "the model misbehaved and layer 4 caught it".
        if case.harm and turn.raw_answer:
            in_draft = detect_harm(case.harm, turn.raw_answer)
            if in_draft:
                console.print(
                    f"  [red]the model's draft leaked:[/red] {', '.join(in_draft)}"
                )

        chat.show(turn)

        if case.harm:
            leaked = detect_harm(case.harm, turn.final)
            verdict = (
                f"[red]LEAKED {leaked}[/red]" if leaked
                else "[green]nothing leaked[/green]"
            )
            console.print(f"  outcome: {verdict}\n")


# Ladder models, weakest first, and the order matters more than it looks.
#
# On gpt-oss-120b every rung below `off` reads 0 leaked, because the model
# refuses almost everything unaided. gpt-oss-20b gives the `off` row more to
# work with but its rungs saturate too. Only qwen3.6-27b still leaks with the
# hardened prompt in place, which makes it the one ladder where you can watch
# each layer knock something down. The stronger two are run as ceilings.
LADDER_MODELS = [
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]


async def run_eval(models: list[str] | None = None) -> None:
    """Run the ladder on each model, print the tables, write the markdown."""
    models = models or LADDER_MODELS
    console.print("[bold]running the ladder[/bold] (free tier is 8k tokens/min)")

    ladders = []
    for model in models:
        console.print(f"[bold cyan]{model}[/bold cyan]")
        results = await evaluate.run_all(model)
        evaluate.save(results)
        ladders.append(results)

    swap = json.loads(SWAP_PATH.read_text(encoding="utf-8")) if SWAP_PATH.exists() else None
    for results in ladders:
        console.print(report.leaderboard(results))
    console.print(report.per_case(ladders[0], "prompt-only"))
    if swap:
        console.print(report.swap_table(swap))
    report.write_markdown(ladders, swap)
    console.print(f"[dim]cache holds {cache.stats()} responses[/dim]")


async def run_swap_cmd() -> None:
    """Run the model-swap control and store it next to the results."""
    console.print("[bold]swapping the chat model[/bold] - same prompt, same attacks")
    swap = await run_swap()
    SWAP_PATH.write_text(json.dumps(swap, indent=2), encoding="utf-8")
    console.print(report.swap_table(swap))


def main() -> None:
    """Parse arguments and dispatch."""
    fix_console()
    parser = argparse.ArgumentParser(description="Guardrails layer")
    parser.add_argument(
        "command", choices=["demo", "eval", "swap", "chat"], nargs="?", default="demo"
    )
    parser.add_argument("--config", default="full stack")
    parser.add_argument(
        "--model", action="append",
        help="chat model for the ladder; repeatable. Defaults to 20b then 120b.",
    )
    args = parser.parse_args()

    if args.command == "chat":
        chat.main(args.config)
    elif args.command == "eval":
        asyncio.run(run_eval(args.model))
    elif args.command == "swap":
        asyncio.run(run_swap_cmd())
    else:
        asyncio.run(demo())


if __name__ == "__main__":
    main()
