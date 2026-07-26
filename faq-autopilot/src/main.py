"""FAQ Autopilot — a long-horizon agent that watches source docs and keeps a FAQ current.

Run one pass (deterministic, terminates):
    PYTHONPATH=. uv run python -m src.main --once
Run the always-on watcher (Ctrl-C to stop):
    PYTHONPATH=. uv run python -m src.main --watch
Start fresh (wipe the DB and generated files first):
    PYTHONPATH=. uv run python -m src.main --reset --once

Each tick: detect which docs drifted, ask the agent for grounded FAQ edits, apply them
autonomously, log every action with its reason + citation, and re-render FAQ.md.
"""

import argparse
import asyncio
import sqlite3
import sys

from groq import AsyncGroq
from rich.console import Console

from src import config
from src.faq_agent import Operation, propose_operations
from src.render import render_faq, render_changelog
from src.state import (
    add_faq,
    connect,
    delete_snapshot,
    faqs_for_source,
    get_snapshot_hashes,
    log_action,
    mark_stale,
    update_faq,
    upsert_snapshot,
)
from src.watcher import DocEvent, current_hash, scan

# Windows consoles default to cp1252, which cannot encode the arrows and em-dashes
# the agent and Rich emit. Force UTF-8 so output never crashes mid-run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

console = Console()

_OP_STYLE = {"add": "green", "update": "yellow", "remove": "red", "stale": "red"}


def _apply_operations(conn: sqlite3.Connection, doc: DocEvent, ops: list[Operation]) -> int:
    """Apply the agent's edits for one doc to the DB and audit log. Returns action count."""
    count = 0
    for op in ops:
        cite = f"{doc.rel_path} → {op.source_anchor}"
        if op.op == "add":
            add_faq(conn, op.question, op.answer, doc.rel_path, op.source_anchor)
            detail = f'added "{op.question}" (cite: {cite}) — {op.reason}'
        elif op.op == "update" and op.faq_id > 0:
            update_faq(conn, op.faq_id, op.question, op.answer, op.source_anchor)
            detail = f'updated #{op.faq_id} "{op.question}" (cite: {cite}) — {op.reason}'
        elif op.op == "remove" and op.faq_id > 0:
            mark_stale(conn, op.faq_id)
            detail = f'removed #{op.faq_id} "{op.question}" — {op.reason}'
        else:
            continue  # malformed op (e.g. update with no id) — skip, don't crash
        log_action(conn, op.op.upper(), detail)
        console.print(f"  [{_OP_STYLE[op.op]}]{op.op.upper():7}[/] {detail}")
        count += 1
    return count


def _handle_removed(conn: sqlite3.Connection, doc: DocEvent) -> int:
    """A source doc vanished: retire every FAQ entry grounded in it."""
    count = 0
    for row in faqs_for_source(conn, doc.rel_path):
        mark_stale(conn, row["id"])
        detail = f'source `{doc.rel_path}` deleted — retired #{row["id"]} "{row["question"]}"'
        log_action(conn, "STALE", detail)
        console.print(f"  [red]STALE  [/] {detail}")
        count += 1
    return count


async def tick(conn: sqlite3.Connection, client: AsyncGroq) -> int:
    """One monitoring pass: detect drift, edit the FAQ, re-render. Returns action count."""
    events = scan(config.DOCS_DIR, get_snapshot_hashes(conn))
    if not events:
        return 0

    changed = [e for e in events if e.kind in ("added", "modified")]
    removed = [e for e in events if e.kind == "removed"]
    labels = ", ".join(f"{e.kind}:{e.rel_path}" for e in events)
    console.print(f"[bold cyan]drift detected[/] → {labels}")

    # One agent call per changed doc, run concurrently.
    batches = await asyncio.gather(
        *(propose_operations(client, e, faqs_for_source(conn, e.rel_path)) for e in changed)
    )

    actions = 0
    for event, ops in zip(changed, batches):
        actions += _apply_operations(conn, event, ops)
        upsert_snapshot(conn, event.rel_path, current_hash(event.content), event.content)
    for event in removed:
        actions += _handle_removed(conn, event)
        delete_snapshot(conn, event.rel_path)

    render_faq(conn, config.FAQ_PATH)
    render_changelog(conn, config.CHANGELOG_PATH)
    return actions


async def run_once(conn: sqlite3.Connection, client: AsyncGroq) -> None:
    """Do a single pass and report the outcome."""
    console.rule("[bold]FAQ Autopilot — single pass")
    actions = await tick(conn, client)
    if actions:
        console.print(f"[green]✓ {actions} change(s) applied.[/] See FAQ.md and CHANGELOG.md.")
    else:
        console.print("[dim]No drift detected. FAQ is already up to date.[/]")


async def run_watch(conn: sqlite3.Connection, client: AsyncGroq, interval: float) -> None:
    """Run forever: scan every `interval` seconds and act on any drift."""
    console.rule("[bold]FAQ Autopilot — watching")
    console.print(
        f"Watching [cyan]{config.DOCS_DIR}[/] every {interval:g}s. Edit a doc to see it react. "
        "Ctrl-C to stop.\n"
    )
    while True:
        actions = await tick(conn, client)
        if actions:
            console.print(f"[green]✓ {actions} change(s) applied.[/]\n")
        await asyncio.sleep(interval)


def _reset() -> None:
    """Delete the DB and generated artifacts so the next run starts clean."""
    for path in (config.DB_PATH, config.FAQ_PATH, config.CHANGELOG_PATH):
        path.unlink(missing_ok=True)
    console.print("[dim]Reset: cleared DB, FAQ.md, CHANGELOG.md.[/]")


def _parse_args() -> argparse.Namespace:
    """Define the CLI: --once vs --watch, plus --interval and --reset."""
    parser = argparse.ArgumentParser(description="Long-horizon FAQ maintenance agent.")
    parser.add_argument("--once", action="store_true", help="run a single pass and exit")
    parser.add_argument("--watch", action="store_true", help="run continuously (default)")
    parser.add_argument("--interval", type=float, default=config.POLL_INTERVAL_SECONDS,
                        help="seconds between scans in --watch mode")
    parser.add_argument("--reset", action="store_true", help="wipe state before running")
    return parser.parse_args()


async def _main() -> None:
    """Wire config, DB, and client, then dispatch to the chosen mode."""
    args = _parse_args()
    config.require_keys()
    if args.reset:
        _reset()

    conn = connect(config.DB_PATH)
    client = config.make_client()
    try:
        if args.once:
            await run_once(conn, client)
        else:
            await run_watch(conn, client, args.interval)
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/]")
