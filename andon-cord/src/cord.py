"""The two nodes that are allowed to stop the line.

`interrupt()` freezes the run and hands a payload to whoever is driving the
graph. When the driver calls back with `Command(resume=...)`, LangGraph re-runs
this node from its first line and `interrupt()` returns the resume value instead
of freezing again.

That replay is why both nodes here do nothing before the interrupt. Anything
above the call runs twice: once on the way down, once on the way back.

Inputs:  ResearchState
Outputs: a `clarifications` update, or nothing when nobody needs to ask
"""

from langgraph.types import interrupt

from src.state import ResearchState


def _ask(questions: list[dict], interpretation: str) -> list[dict]:
    """Freeze the run, show the questions, and pair each answer to its question.

    Args:
        questions: dicts with `id`, `asked_by` and `text`.
        interpretation: how the lead is reading the brief, so the human can see
            what the team will do if they say nothing.

    Returns:
        The same dicts with the human's `answer` filled in.

    The payload carries everything needed to answer, because whoever reads it may
    be a different process hours later with no memory of this run. The resume
    value comes back as {question id: answer}, and the id is what keeps two
    researchers' answers from being swapped between them.
    """
    answers: dict = interrupt(
        {"questions": questions, "interpretation": interpretation}
    )
    return [{**q, "answer": answers.get(q["id"], "")} for q in questions]


def ask_lead_node(state: ResearchState) -> dict:
    """Put the research lead's one question to the human.

    The router decides whether this node is entered at all, so by the time we
    are here the question is already worth asking.
    """
    question = {
        "id": "lead",
        "asked_by": "research lead",
        "text": state["question"],
    }
    return {"clarifications": _ask([question], state.get("interpretation", ""))}


def _open_questions(findings: list[dict]) -> list[dict]:
    """Collect the questions the researchers flagged, in researcher order."""
    return [
        {
            "id": f"r{finding['index']}",
            "asked_by": f"researcher {finding['index']}",
            "text": finding["ask"],
        }
        for finding in sorted(findings, key=lambda f: f["index"])
        if finding["needs_human"] and finding["ask"]
    ]


def desk_node(state: ResearchState) -> dict:
    """Batch every researcher's question into one turn for the human.

    Both researchers finish in the same superstep, so both questions are on the
    desk before anyone is disturbed. Asking from inside the researchers instead
    would interrupt the human twice, and would make each researcher repeat its
    web search on resume.
    """
    questions = _open_questions(state.get("findings", []))
    if not questions or state.get("ask_policy") != "team":
        # Nothing to ask, or the team is not allowed to. Either way, no pause.
        return {}
    return {"clarifications": _ask(questions, state.get("interpretation", ""))}


def suppressed_questions(state: ResearchState) -> list[dict]:
    """The questions that were thought of but never put to a human.

    Derived, not logged: a question is suppressed when an agent flagged it and
    no clarification carries its id. That makes `--ask never` show exactly what
    the policy cost, using the same state the other policies write.
    """
    answered = {c["id"] for c in state.get("clarifications", [])}
    pending: list[dict] = []
    if state.get("needs_human") and state.get("question") and "lead" not in answered:
        pending.append(
            {"id": "lead", "asked_by": "research lead", "text": state["question"]}
        )
    pending += [
        q for q in _open_questions(state.get("findings", [])) if q["id"] not in answered
    ]
    return pending
