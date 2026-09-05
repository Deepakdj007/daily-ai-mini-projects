"""Interactive demo: talk to the defended assistant and watch the layers fire.

Inputs:  typed messages on stdin
Outputs: the guarded answer plus a trace of what each layer decided

Run this before the evaluation. Watching a single turn go through the stack is
what makes the leaderboard readable afterwards.
"""

from __future__ import annotations

import asyncio

from rich.console import Console

from src.agent import respond
from src.ladder import CONFIGS_BY_NAME

console = Console()


def show(turn) -> None:
    """Print one turn's guard trace and final answer."""
    for note in turn.notes:
        console.print(f"  [dim]layer:[/dim] {note}")

    if turn.inp.quarantined:
        console.print(
            f"  [yellow]quarantined documents:[/yellow] "
            f"{', '.join(turn.inp.quarantined)}"
        )
    if turn.out and turn.out.pii_found:
        types = sorted({t for t, _ in turn.out.pii_found})
        console.print(f"  [yellow]pii redacted:[/yellow] {', '.join(types)}")
    if turn.blocked_by:
        console.print(f"  [red]blocked by:[/red] {turn.blocked_by}")

    console.print(f"\n[bold green]assistant[/bold green] {turn.final}\n")


async def loop(config_name: str = "full stack") -> None:
    """Read messages until the user quits."""
    cfg = CONFIGS_BY_NAME[config_name]
    console.print(f"[bold]PaySetu support[/bold] - guard config: {cfg.name}")
    console.print("[dim]blank line to quit[/dim]\n")

    while True:
        try:
            text = console.input("[bold cyan]you[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        turn = await respond(text, cfg)
        show(turn)


def main(config_name: str = "full stack") -> None:
    """Entry point for the interactive demo."""
    asyncio.run(loop(config_name))
