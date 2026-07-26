"""Entry point: run the research team on a topic and save the report.

Run it:
    PYTHONPATH=. uv run python -m src.main "the state of solid-state batteries in 2026"
    PYTHONPATH=. uv run python -m src.main            # uses DEFAULT_TOPIC

Inputs:  a topic on the command line, GEMINI_API_KEY in .env.
Outputs: live progress in the terminal + output/report.md
"""

import argparse
import asyncio
import sys
import time

from rich.console import Console
from workflows.errors import WorkflowTimeoutError

from src import config
from src.events import ProgressEvent
from src.workflow import DeepResearchWorkflow

DEFAULT_TOPIC = "the state of solid-state batteries in 2026"

console = Console()


def _parse_args() -> argparse.Namespace:
    """Read the topic from the command line."""
    parser = argparse.ArgumentParser(
        description="Research any topic with a team of parallel agents."
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=DEFAULT_TOPIC,
        help=f"what to research (default: {DEFAULT_TOPIC!r})",
    )
    return parser.parse_args()


def _save(report: str) -> None:
    """Write the report to output/report.md, creating the folder if needed."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_PATH.write_text(report, encoding="utf-8")


async def _run(topic: str) -> None:
    """Stream one research run to the console and save the result."""
    flow = DeepResearchWorkflow(timeout=config.WORKFLOW_TIMEOUT)

    console.rule(f"[bold]Deep Research — {topic}")
    started = time.perf_counter()

    handler = flow.run(topic=topic)
    # A handler's stream can only be consumed once, so read it here and await
    # the handler afterwards for the final result.
    async for event in handler.stream_events():
        if isinstance(event, ProgressEvent):
            console.print(f"[{event.style}]{event.agent:>12}[/] │ {event.msg}")

    report = str(await handler)
    elapsed = time.perf_counter() - started

    _save(report)
    console.rule("[bold green]Done")
    console.print(
        f"[green]✓[/] {elapsed:.1f}s with {config.RESEARCH_WORKERS} researcher(s) "
        f"— report saved to {config.REPORT_PATH}"
    )


async def _main() -> None:
    """Validate config, then run, turning a timeout into a readable message."""
    config.require_keys()
    args = _parse_args()
    try:
        await _run(args.topic)
    except WorkflowTimeoutError as exc:
        console.print(f"[red]Timed out:[/] {exc}")
        console.print(
            f"[dim]Raise WORKFLOW_TIMEOUT in src/config.py "
            f"(currently {config.WORKFLOW_TIMEOUT:.0f}s).[/]"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(_main())
