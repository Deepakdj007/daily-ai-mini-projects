"""Run the same briefs with the cord live and with the cord cut.

Both policies use byte-identical prompts. `never` does not tell the agents to
stop thinking of questions — it only refuses to put them to a human. So the
difference between the two rows is the gate, not a different instruction.

What the table measures: whether the team pulls the cord on briefs that hide
something and stays quiet on briefs that do not, and whether the answer reaches
the report. It does not judge how good the report is.

Inputs:  data/briefs.json
Outputs: a terminal table and output/compare.md
"""

import json
import re
import time

from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console

from src import config, report
from src.cord import suppressed_questions
from src.graph import build_graph, thread_config
from src.human import ScriptedHuman, drive

POLICIES: tuple[str, ...] = ("never", "team")


def load_briefs(use_all: bool) -> list[dict]:
    """Read the fixtures and refuse to run if a planted token is guessable.

    A token that already appears in the brief could reach the report without
    anyone being asked, which would make every number in the table meaningless.
    """
    data = json.loads(config.BRIEFS_PATH.read_text(encoding="utf-8"))
    briefs = [b for b in data["briefs"] if use_all or b["default"]]
    for brief in briefs:
        for token in brief["tokens"]:
            if token.lower() in brief["brief"].lower():
                raise AssertionError(
                    f"{brief['id']}: the token '{token}' is already in the brief, "
                    "so finding it in the report would prove nothing."
                )
        # Every brief needs something to say, including the ones nobody should
        # ask about. An empty reply is not an answer, so a false alarm there
        # would park the run instead of being recorded as the false alarm it is.
        if not brief["oracle_answer"].strip():
            raise AssertionError(f"{brief['id']}: oracle_answer is empty.")
    return briefs


def _strip_scope(report: str) -> str:
    """Drop the Scope line before looking for planted tokens.

    The Scope line quotes the human's answer verbatim, so leaving it in would
    let every answered run score full marks for nothing. Searching the body only
    asks the harder question: did the answer change the research and the prose?
    """
    return "\n".join(
        line for line in report.splitlines() if not line.startswith("> **Scope:**")
    )


def _tokens_found(report: str, tokens: list[str]) -> int:
    """Count planted tokens present in the report body, whole words only."""
    body = _strip_scope(report)
    return sum(
        1
        for token in tokens
        if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", body, re.IGNORECASE)
    )


def run_one(brief: dict, policy: str) -> dict:
    """Run one brief under one policy and record what happened.

    Uses an in-memory checkpointer: nothing here needs to survive the process,
    and it keeps the comparison out of the real checkpoint file.
    """
    human = ScriptedHuman(brief["oracle_answer"])
    started = time.time()
    with InMemorySaver() as checkpointer:
        graph = build_graph(checkpointer)
        cfg = thread_config(f"cmp-{brief['id']}-{policy}")
        snapshot = drive(
            graph, cfg, human,
            initial={"brief": brief["brief"], "ask_policy": policy},
        )
    values = snapshot.values
    findings = values.get("findings", [])
    asked = [c["asked_by"] for c in values.get("clarifications", [])]
    wanted = [q["asked_by"] for q in suppressed_questions(values)]
    return {
        "id": brief["id"],
        "kind": brief["kind"],
        "policy": policy,
        "asked": asked,
        "wanted": wanted,
        "turns": human.turns,
        "questions": human.questions_asked,
        "tokens_found": _tokens_found(values.get("report", ""), brief["tokens"]),
        "tokens_total": len(brief["tokens"]),
        # A row built on collapsed researchers is not evidence about asking.
        "failures": [f["failed"] for f in findings if f.get("failed")],
        "dry": sum(1 for f in findings if not f.get("n_results")),
        "seconds": round(time.time() - started, 1),
    }


def run_comparison(use_all: bool = False) -> None:
    """Run every brief under every policy, print the table, save it."""
    config.require_keys()
    console = Console()
    briefs = load_briefs(use_all)
    rows: list[dict] = []

    for brief in briefs:
        for policy in POLICIES:
            console.print(f"[dim]{brief['id']} / {policy} ...[/dim]")
            rows.append(run_one(brief, policy))

    console.print(report.render(rows))
    notes = report.warnings(rows)
    for note in notes:
        console.print(f"[yellow]warning: {note}[/yellow]")

    path = report.write_markdown(rows, notes, len(briefs))
    console.print(f"\nWritten to {path}")

