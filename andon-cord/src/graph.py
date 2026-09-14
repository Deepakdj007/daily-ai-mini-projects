"""Wiring: one lead, two researchers in parallel, a question desk, a writer.

    START -> lead -> ask_lead -> lead        (only when the lead pulls the cord)
             lead -> researcher x2 -> desk -> writer -> END

The router after the lead is also the dispatcher. A conditional edge may return
a node name or a list of Send objects, and a Send is what puts two researchers
in the same superstep.

Inputs:  a checkpointer
Outputs: a compiled graph
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src import config
from src.cord import ask_lead_node, desk_node
from src.nodes import lead_node, researcher_node, writer_node
from src.state import ResearchState


def route_after_lead(state: ResearchState) -> str | list[Send]:
    """Either stop for one question, or dispatch the whole team at once.

    Args:
        state: the lead's plan, plus any clarification already answered.

    Returns:
        "ask_lead" to pull the cord, or one Send per researcher.

    The `already_asked` guard is not optional. Without it a lead that keeps
    flagging the same doubt cycles lead -> ask_lead -> lead until LangGraph
    stops it with GraphRecursionError at 25 supersteps.
    """
    already_asked = any(c["id"] == "lead" for c in state.get("clarifications", []))
    wants_to_ask = state.get("needs_human") and state.get("question")
    if wants_to_ask and not already_asked and state.get("ask_policy") != "never":
        return "ask_lead"

    questions = state.get("sub_questions") or [state["brief"]]
    return [
        Send(
            "researcher",
            {
                "index": i,
                "question": question,
                "brief": state["brief"],
                "interpretation": state.get("interpretation", ""),
                "clarifications": state.get("clarifications", []),
                "ask_policy": state.get("ask_policy", config.DEFAULT_POLICY),
            },
        )
        for i, question in enumerate(questions, 1)
    ]


def build_graph(checkpointer: BaseCheckpointSaver):
    """Assemble and compile the team.

    A checkpointer is required, not optional: interrupt() has nowhere to freeze
    the run without one, and get_state() has nothing to read back.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("lead", lead_node)
    graph.add_node("ask_lead", ask_lead_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("desk", desk_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "lead")
    graph.add_conditional_edges("lead", route_after_lead, ["ask_lead", "researcher"])
    graph.add_edge("ask_lead", "lead")
    # A plain edge is the whole fan-in: LangGraph runs desk once, in the
    # superstep after every dispatched researcher has committed its finding.
    graph.add_edge("researcher", "desk")
    graph.add_edge("desk", "writer")
    graph.add_edge("writer", END)

    return graph.compile(checkpointer=checkpointer)


def thread_config(thread_id: str) -> dict:
    """The config that ties every call to one checkpointed run."""
    return {"configurable": {"thread_id": thread_id}}


def _all_thread_ids(checkpointer: BaseCheckpointSaver) -> list[str]:
    """Every thread id in the checkpoint file, newest first.

    The list is built to completion before anyone reads a thread's state. That is
    not tidiness: SqliteSaver guards its connection with a plain, single-entry
    lock and holds it for as long as `list()` is being iterated. Calling
    get_state() inside that loop makes the process wait for a lock it already
    holds, and it hangs forever with no error to read.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id not in seen:
            seen.add(thread_id)
            ordered.append(thread_id)
    return ordered


def parked_threads(checkpointer: BaseCheckpointSaver, graph) -> list[tuple[str, dict]]:
    """Every run still waiting on a human, newest first.

    The checkpoint file is the only record. Nothing in the library lists threads
    holding a pending question, so walk every saved thread and ask each one
    whether it is waiting. One source of truth beats a second table that can
    drift out of step with the checkpoints.
    """
    parked: list[tuple[str, dict]] = []
    for thread_id in _all_thread_ids(checkpointer):
        snapshot = graph.get_state(thread_config(thread_id))
        if snapshot.interrupts:
            parked.append((thread_id, snapshot.interrupts[0].value))
    return parked
