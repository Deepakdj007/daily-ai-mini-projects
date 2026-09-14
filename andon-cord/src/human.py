"""The human, and the loop that drives the graph past every pause.

The human lives outside the graph. The graph never calls input() and never knows
whether it is talking to a terminal or a fixture — it emits a question and waits
for a dict of answers. That is what lets the same code be driven by a person, by
a test, or by a web page.

Inputs:  a compiled graph and a thread config
Outputs: the final StateSnapshot
"""

from typing import Protocol

from langgraph.types import Command


def _clean(raw: str) -> str:
    """Trim a typed answer, including the byte-order mark Windows can prepend.

    On Windows, text piped into a script often arrives with a U+FEFF in front of
    the first line. It is invisible, str.strip() does not treat it as
    whitespace, and it travels all the way into the report.
    """
    return raw.lstrip("﻿").strip()


class Human(Protocol):
    """Anything that can answer a batch of questions."""

    def answer(self, payload: dict) -> dict:
        """Map each question id in `payload` to an answer string."""
        ...


class TerminalHuman:
    """Asks the questions at the terminal, one prompt per question."""

    def answer(self, payload: dict) -> dict:
        """Show what the team assumed, then collect one answer per question."""
        print("\n== THE LINE HAS STOPPED ==========================")
        if payload.get("interpretation"):
            print(f"Reading the brief as: {payload['interpretation']}")
        replies: dict[str, str] = {}
        for question in payload["questions"]:
            print(f"\n  {question['asked_by']} asks:")
            print(f"    {question['text']}")
            replies[question["id"]] = _clean(input("  your answer > "))
        print("==================================================")
        return replies


class ScriptedHuman:
    """Answers every question with one prepared line.

    Used by src/compare.py so a run can be repeated without a person sitting
    there, and so the answer is identical in every policy being compared.
    """

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.turns = 0
        self.questions_asked = 0

    def answer(self, payload: dict) -> dict:
        """Give the same reply to every question in this batch."""
        self.turns += 1
        self.questions_asked += len(payload["questions"])
        return {question["id"]: self.reply for question in payload["questions"]}


def drive(graph, config: dict, human: Human, initial: dict | None = None,
          park: bool = False):
    """Run the graph to the end, answering every pause on the way.

    Args:
        graph: the compiled graph.
        config: the thread config.
        human: who answers the questions.
        initial: the starting state, or None to resume an existing run.
        park: stop at the first pause and leave the run frozen on disk.

    Returns:
        The final StateSnapshot. It is still frozen if `park` was set, or if the
        caller interrupted the prompt.

    Two details matter here. durability="sync" makes LangGraph finish writing the
    checkpoint before invoke() returns, which is what lets the process exit a
    moment later and still be resumable. And the loop re-reads get_state() rather
    than trusting invoke()'s return value, because an answer can produce a second
    pause and what matters is whether the run is waiting again.
    """
    if initial is not None:
        graph.invoke(initial, config, durability="sync")
    snapshot = graph.get_state(config)

    while snapshot.interrupts:
        if park:
            return snapshot
        answers = human.answer(snapshot.interrupts[0].value)
        graph.invoke(Command(resume=answers), config, durability="sync")
        snapshot = graph.get_state(config)

    return snapshot
