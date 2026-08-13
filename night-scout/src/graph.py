"""The graph, its checkpointer, and the thread identity scheme.

Inputs:  a checkpointer
Outputs: a compiled graph, plus the thread_id helpers both processes must agree on

A checkpointer is REQUIRED here. interrupt() has nowhere to park without one, and
get_state() raises `ValueError: No checkpointer set` — which is what makes the
morning inbox possible at all.
"""

import hashlib
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src import config
from src.nodes import (
    commit_node,
    detail_node,
    discard_node,
    draft_node,
    gate_node,
    route_after_gate,
)
from src.state import ScoutState


def make_thread_id(night_id: str, item_id: str) -> str:
    """One checkpointed thread per item.

    Not one thread per night: interrupt() unwinds the whole run for its thread, so a
    thread holding six items would freeze items 2-6 behind the human decision on
    item 1. Per-item threads park and resume independently, in any order, hours apart.
    """
    digest = hashlib.sha256(item_id.encode()).hexdigest()[:8]
    return f"night-{night_id}-{digest}"


def thread_config(thread_id: str) -> dict:
    """The config that ties every call — park and resume — to one thread."""
    return {"configurable": {"thread_id": thread_id}}


def open_checkpointer() -> SqliteSaver:
    """Open memory.db with settings that survive two processes on one file.

    from_conn_string() is deliberately not used. It hard-codes sqlite's 5-second
    busy timeout, which is not enough when the night process is mid-write, and it is
    a context manager, so it would close itself the moment Streamlit finished its
    first script run.
    """
    conn = sqlite3.connect(
        str(config.MEM_PATH),
        timeout=config.SQLITE_TIMEOUT,
        check_same_thread=False,  # Streamlit reruns on a different thread
        isolation_level=None,     # autocommit; keeps write transactions short
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def build_graph(checkpointer):
    """Wire the nodes together and compile.

    detail -> draft -> gate -> (approve) commit -> END
                        |  \\-- (edit)    -> draft
                        \\---- (reject)   -> discard -> END
    """
    graph = StateGraph(ScoutState)

    graph.add_node("detail", detail_node)
    graph.add_node("draft", draft_node)
    graph.add_node("gate", gate_node)
    graph.add_node("commit", commit_node)
    graph.add_node("discard", discard_node)

    graph.add_edge(START, "detail")
    graph.add_edge("detail", "draft")
    graph.add_edge("draft", "gate")

    # The human's answer decides where the run goes. An edit loops back to drafting,
    # which parks again with a new draft — hence the revision cap in the router.
    graph.add_conditional_edges(
        "gate",
        route_after_gate,
        {"commit": "commit", "draft": "draft", "discard": "discard"},
    )
    graph.add_edge("commit", END)
    graph.add_edge("discard", END)

    return graph.compile(checkpointer=checkpointer)


# --- reading and resuming parked runs ---------------------------------------
# These live here rather than in inbox.py so they can be used without Streamlit —
# by a test, a CLI, or a different front end. The inbox is one caller, not the owner.


def load_parked_payload(graph, thread_id: str) -> dict | None:
    """Read the draft a parked thread is waiting on, straight from the checkpoint.

    Returns exactly the dict the gate node passed to interrupt(), or None if the
    thread has already finished. This works in a process that never ran the graph:
    get_state() rebuilds the pending interrupt from writes stored in memory.db.
    """
    snapshot = graph.get_state(thread_config(thread_id))
    if not snapshot.interrupts:
        return None
    return snapshot.interrupts[0].value


def resume(graph, thread_id: str, decision: str, feedback: str = "") -> str:
    """Hand a human decision to a parked run and let it continue.

    The return value of invoke() is deliberately ignored in favour of re-reading
    get_state(). That is version-proof across LangGraph's v1 dict and v2 GraphOutput
    shapes, and an 'edit' re-drafts and parks again — so what matters is whether it
    is waiting once more, not what invoke happened to hand back.
    """
    run_config = thread_config(thread_id)
    graph.invoke(
        Command(resume={"decision": decision, "feedback": feedback}), run_config
    )
    snapshot = graph.get_state(run_config)
    if snapshot.interrupts:
        return "parked"  # an edit produced a new draft, still awaiting a human
    return snapshot.values.get("status", "done")
