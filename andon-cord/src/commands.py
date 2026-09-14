"""What each command actually does: run a brief, list parked runs, answer one.

Kept out of src/main.py so that file stays a thin command-line shell: argument
parsing and error messages, nothing else.

Inputs:  a brief, an ask policy, or a thread id
Outputs: a markdown report under output/, and terminal summaries
"""

import re
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver

from src import config
from src.cord import suppressed_questions
from src.graph import build_graph, parked_threads, thread_config
from src.human import TerminalHuman, drive


def _slug(text: str) -> str:
    """Turn a brief into a short, safe filename stem."""
    words = re.sub(r"[^a-z0-9\s-]", "", text.lower()).split()
    return "-".join(words[:6]) or "report"


def _save_report(snapshot, thread_id: str) -> None:
    """Write the finished report to disk, then say where it went.

    Writing happens here and not inside the writer node on purpose: a node that
    pauses re-runs from the top, so a file written inside one would be written
    twice.
    """
    report = snapshot.values.get("report", "")
    if not report:
        print("The run finished without a report.")
        return
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / f"{_slug(snapshot.values.get('brief', thread_id))}.md"
    path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {path}")


def _report_asks(snapshot) -> None:
    """Say what was asked, and what was thought of but never asked."""
    for clarification in snapshot.values.get("clarifications", []):
        print(f"  asked ({clarification['asked_by']}): {clarification['text']}")
        print(f"    -> {clarification['answer']}")
    for question in suppressed_questions(snapshot.values):
        print(f"  would have asked ({question['asked_by']}): {question['text']}")


def run(brief: str, policy: str, park: bool) -> None:
    """Start a new research run under one ask policy."""
    config.require_keys()
    thread_id = uuid4().hex[:8]
    initial = {"brief": brief, "ask_policy": policy}

    with SqliteSaver.from_conn_string(str(config.DB_PATH)) as checkpointer:
        graph = build_graph(checkpointer)
        cfg = thread_config(thread_id)
        try:
            snapshot = drive(graph, cfg, TerminalHuman(), initial=initial, park=park)
        except (EOFError, KeyboardInterrupt):
            print(f"\nNo answer given. Run {thread_id} is still parked.")
            print(f"  PYTHONPATH=. uv run python -m src.main answer {thread_id}")
            return

        if snapshot.interrupts:
            _print_parked(snapshot, thread_id)
            return

        print(f"\nInterpretation: {snapshot.values.get('interpretation', '')}")
        _report_asks(snapshot)
        _save_report(snapshot, thread_id)


def _print_parked(snapshot, thread_id: str) -> None:
    """Show a frozen run and how to come back to it."""
    payload = snapshot.interrupts[0].value
    print("\n== PARKED ========================================")
    print(f"Reading the brief as: {payload.get('interpretation', '')}")
    for question in payload["questions"]:
        print(f"  {question['asked_by']}: {question['text']}")
    print(f"\nFrozen to {config.DB_PATH}. thread_id = {thread_id}")
    print("Answer it later, even from a new terminal:")
    print(f"  PYTHONPATH=. uv run python -m src.main answer {thread_id}")


def inbox() -> None:
    """List every run that is waiting on an answer."""
    with SqliteSaver.from_conn_string(str(config.DB_PATH)) as checkpointer:
        graph = build_graph(checkpointer)
        parked = parked_threads(checkpointer, graph)
        if not parked:
            print("Nothing is waiting on you.")
            return
        print(f"{len(parked)} run(s) waiting on you:\n")
        for thread_id, payload in parked:
            print(f"  {thread_id}  ({len(payload['questions'])} question(s))")
            for question in payload["questions"]:
                print(f"      {question['asked_by']}: {question['text']}")
            print()


def answer(thread_id: str) -> None:
    """Resume a parked run in this process and drive it to the end."""
    config.require_keys()
    with SqliteSaver.from_conn_string(str(config.DB_PATH)) as checkpointer:
        graph = build_graph(checkpointer)
        cfg = thread_config(thread_id)
        snapshot = graph.get_state(cfg)

        if not snapshot.values:
            print(f"No run found for thread_id '{thread_id}'.")
            return
        if not snapshot.interrupts:
            print(f"Run {thread_id} is not waiting on anything.")
            _save_report(snapshot, thread_id)
            return

        try:
            snapshot = drive(graph, cfg, TerminalHuman())
        except (EOFError, KeyboardInterrupt):
            print(f"\nNo answer given. Run {thread_id} is still parked.")
            return

        print(f"\nInterpretation: {snapshot.values.get('interpretation', '')}")
        _report_asks(snapshot)
        _save_report(snapshot, thread_id)


