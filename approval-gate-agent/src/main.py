"""CLI entry point: propose an action (freeze) and review it later (resume).

The two commands run in separate processes on purpose. `propose` drafts the action,
hits the interrupt, and persists the frozen graph to checkpoints.sqlite before exiting.
`review` re-opens that SQLite file by thread_id and resumes from the exact checkpoint —
proving the approval survives a process restart.

    PYTHONPATH=. uv run python -m src.main propose "Email priya@acme.com ..."
    PYTHONPATH=. uv run python -m src.main review <thread_id>
    PYTHONPATH=. uv run python -m src.main list
"""

import argparse
import json
import sys
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from src.audit import record_event
from src.config import AUDIT_LOG, DB_PATH, require_keys
from src.graph import build_graph

# Windows consoles default to cp1252, which cannot encode the box-drawing and
# emoji characters we print. Force UTF-8 so output never crashes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _format_draft(action: dict, revision: int | None) -> str:
    """Render a planned action as a clean terminal preview."""
    rev = f" (revision {revision})" if revision else ""
    return (
        f"  type:    {action.get('action_type')}{rev}\n"
        f"  to:      {action.get('recipient')}\n"
        f"  subject: {action.get('subject')}\n"
        f"  body:    {action.get('body')}\n"
        f"  summary: {action.get('summary')}"
    )


def _config(thread_id: str) -> dict:
    """The config dict that ties every call to one checkpointed thread."""
    return {"configurable": {"thread_id": thread_id}}


def _audit_if_frozen(snapshot, thread_id: str) -> bool:
    """If the run froze at the approval gate, log it once. Returns True if frozen.

    Logging the freeze here (not inside the node) keeps the audit trail exact:
    one entry per actual freeze, with no duplicate when the node re-runs on resume.
    """
    if snapshot.next and "approval" in snapshot.next:
        action = snapshot.values.get("action", {})
        record_event(
            thread_id, "awaiting_approval", "agent",
            {"revision": snapshot.values.get("revision"), "summary": action.get("summary")},
        )
        return True
    return False


def propose(request: str) -> None:
    """Plan the action, run until the approval interrupt, then freeze and exit."""
    require_keys()
    thread_id = uuid4().hex[:8]
    with SqliteSaver.from_conn_string(str(DB_PATH)) as cp:
        graph = build_graph(cp)
        config = _config(thread_id)
        graph.invoke(
            {"request": request, "thread_id": thread_id, "revision": 0}, config
        )
        snapshot = graph.get_state(config)
        _audit_if_frozen(snapshot, thread_id)

    if not snapshot.next:
        # Validator bounced it past the revision limit before reaching a human.
        print(f"\nRun {thread_id} ended without approval: {snapshot.values.get('result')}")
        return

    print("\n── DRAFT AWAITING APPROVAL ─────────────────────────")
    print(_format_draft(snapshot.values["action"], snapshot.values.get("revision")))
    print("────────────────────────────────────────────────────")
    print(f"\nFrozen to {DB_PATH}. thread_id = {thread_id}")
    print("Review it (even after closing this terminal):")
    print(f"  PYTHONPATH=. uv run python -m src.main review {thread_id}")


def _prompt_human() -> dict:
    """Read one approve / edit / reject decision from the terminal."""
    while True:
        raw = input("\nDecision [approve | edit <note> | reject <reason>]: ").strip()
        parts = raw.split(maxsplit=1)
        if not parts:
            continue
        decision = parts[0].lower()
        note = parts[1] if len(parts) > 1 else ""
        if decision in {"approve", "edit", "reject"}:
            return {"decision": decision, "feedback": note}
        print("  Please type approve, edit, or reject.")


def review(thread_id: str) -> None:
    """Resume a frozen run: show the draft, take a decision, drive to completion."""
    require_keys()
    with SqliteSaver.from_conn_string(str(DB_PATH)) as cp:
        graph = build_graph(cp)
        config = _config(thread_id)
        snapshot = graph.get_state(config)

        if not snapshot.values:
            print(f"No run found for thread_id '{thread_id}'.")
            return
        if not snapshot.next:
            # Already resolved — safe no-op. This covers the double-review case.
            status = snapshot.values.get("status", "done")
            print(f"Run {thread_id} is already {status}: {snapshot.values.get('result')}")
            return

        # Loop: each edit re-drafts and interrupts again, so keep prompting until
        # the run reaches a terminal node (execute or cancel). If the reviewer
        # bails (Ctrl+C / Ctrl+D), the run stays frozen on disk for later.
        try:
            while snapshot.next:
                print("\n── DRAFT AWAITING APPROVAL ─────────────────────────")
                print(_format_draft(snapshot.values.get("action", {}), snapshot.values.get("revision")))
                print("────────────────────────────────────────────────────")
                graph.invoke(Command(resume=_prompt_human()), config)
                snapshot = graph.get_state(config)
                # An edit re-drafts and freezes again — log that new freeze once.
                _audit_if_frozen(snapshot, thread_id)
        except (EOFError, KeyboardInterrupt):
            print(f"\nNo decision made. Run {thread_id} is still frozen; resume it with:")
            print(f"  PYTHONPATH=. uv run python -m src.main review {thread_id}")
            return

        print(f"\nFinal status: {snapshot.values.get('status')}")


def list_pending() -> None:
    """List threads whose most recent audit event is still awaiting approval."""
    if not AUDIT_LOG.exists():
        print("No audit log yet.")
        return
    last: dict[str, str] = {}
    for line in AUDIT_LOG.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        last[entry["thread_id"]] = entry["event"]
    pending = [tid for tid, event in last.items() if event == "awaiting_approval"]
    if not pending:
        print("No pending approvals.")
        return
    print("Pending approvals:")
    for tid in pending:
        print(f"  {tid}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-in-the-loop approval gate agent.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_propose = sub.add_parser("propose", help="Draft an action and freeze for approval.")
    p_propose.add_argument("request", help="Natural-language description of the action.")

    p_review = sub.add_parser("review", help="Resume a frozen run by thread_id.")
    p_review.add_argument("thread_id", help="The thread_id printed by propose.")

    sub.add_parser("list", help="List threads awaiting approval.")

    args = parser.parse_args()
    if args.command == "propose":
        propose(args.request)
    elif args.command == "review":
        review(args.thread_id)
    elif args.command == "list":
        list_pending()


if __name__ == "__main__":
    sys.exit(main())
