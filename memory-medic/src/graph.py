"""The repair gate: a graph whose only job is to park a decision durably.

Inputs:  a checkpointer and a store connection
Outputs: a compiled graph, plus the thread helpers both processes agree on

No model is called anywhere in this graph. Extraction and classification have
already happened by the time a plan gets here, so the graph is three nodes and
a sqlite write. That keeps it synchronous, which matters: SqliteSaver has no
async path, and a graph that awaited Groq would force the review UI to run an
event loop inside a Streamlit rerun.

A checkpointer is required. interrupt() has nowhere to park without one, and
get_state() raises `No checkpointer set` - which is exactly what makes a review
inbox in a different process possible at all.
"""

from __future__ import annotations

import hashlib
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src import config, repair, store
from src.nodes import RepairState, apply_node, gate_node, reject_node, route_after_gate
from src.policy import RepairPlan


def make_thread_id(plan: RepairPlan, evidence: str = "", generation: str = "") -> str:
    """One thread per distinct repair, derived from its content.

    Content-derived rather than random so a re-detection of the same problem
    lands on the same thread instead of parking a second identical card. New
    evidence produces a new thread, which is the behaviour we want: that really
    is a different decision.

    The store's generation is mixed in so a wiped-and-reseeded store can never
    collide with a decision already made against the old rows. Without it, row
    ids restart at 1, the ids match, and a finished checkpoint short-circuits
    the new repair - the gate stops gating and nothing says so.
    """
    material = "|".join([
        generation, plan.user_id, plan.topic, plan.scope, str(plan.old_id), plan.detector,
        hashlib.sha256((evidence or plan.evidence).encode("utf-8")).hexdigest()[:8],
    ])
    return "repair-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def thread_config(thread_id: str, conn=None, clock=None) -> dict:
    """The config that ties park and resume to one thread, plus its resources."""
    return {"configurable": {"thread_id": thread_id, "conn": conn, "clock": clock}}


def open_checkpointer(path=None) -> SqliteSaver:
    """Open the checkpoint file with settings that survive two processes.

    from_conn_string() is deliberately not used: it hard-codes a five-second
    busy timeout, too short when a sweep is mid-write, and it is a context
    manager that would close itself the moment Streamlit finished its first run.
    """
    conn = sqlite3.connect(
        str(path or config.CHECKPOINT_DB),
        timeout=config.SQLITE_TIMEOUT,
        check_same_thread=False,
        isolation_level=None,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def build_graph(checkpointer):
    """gate -> apply | reject. Nothing else, on purpose."""
    graph = StateGraph(RepairState)
    graph.add_node("gate", gate_node)
    graph.add_node("apply", apply_node)
    graph.add_node("reject", reject_node)
    graph.add_edge(START, "gate")
    graph.add_conditional_edges("gate", route_after_gate, {"apply": "apply", "reject": "reject"})
    graph.add_edge("apply", END)
    graph.add_edge("reject", END)
    return graph.compile(checkpointer=checkpointer)


def launch_repair(graph, conn, plan: RepairPlan, *, clock, policy, oracle_human=None) -> str:
    """Park a repair for review, and index it so an inbox can find it.

    The index row is written here, after invoke() returns and only if the run
    really did park. Writing it inside the gate node would duplicate it on
    every resume, because the node re-runs from the top.
    """
    thread_id = make_thread_id(plan, generation=store.generation(conn))
    run_config = thread_config(thread_id, conn=conn, clock=clock)

    snapshot = graph.get_state(run_config)
    if snapshot.interrupts:
        _ensure_indexed(conn, plan, thread_id, clock)
        return "parked"
    if snapshot.created_at is not None:
        return snapshot.values.get("outcome", "done")

    graph.invoke({"thread_id": thread_id, "plan": plan.to_payload(), "revision": 1},
                 run_config, durability="sync")
    snapshot = graph.get_state(run_config)
    if not snapshot.interrupts:
        return snapshot.values.get("outcome", "done")
    _ensure_indexed(conn, plan, thread_id, clock)
    if oracle_human is not None:
        payload = snapshot.interrupts[0].value
        return resume_thread(graph, conn, thread_id, oracle_human(payload), clock=clock)
    return "parked"


def _ensure_indexed(conn, plan: RepairPlan, thread_id: str, clock) -> None:
    """Write the parked audit row unless it is already there."""
    existing = conn.execute(
        "SELECT id FROM repairs WHERE thread_id = ? LIMIT 1", (thread_id,)
    ).fetchone()
    if existing is None:
        repair.park(conn, plan, clock=clock, thread_id=thread_id)


def load_parked_payload(graph, thread_id: str, conn=None, clock=None) -> dict | None:
    """Read what a parked thread is waiting on, straight from the checkpoint.

    Works in a process that never ran the graph: get_state rebuilds the pending
    interrupt from the writes stored in the checkpoint file.
    """
    snapshot = graph.get_state(thread_config(thread_id, conn, clock))
    if not snapshot.interrupts:
        return None
    return snapshot.interrupts[0].value


def resume_thread(graph, conn, thread_id: str, decision, *, clock) -> str:
    """Hand a human decision to a parked run and let it finish.

    invoke()'s return value is ignored in favour of re-reading get_state, which
    is version-proof across LangGraph's differing output shapes and answers the
    only question that matters: is it still waiting?
    """
    if isinstance(decision, str):
        decision = {"decision": decision}
    run_config = thread_config(thread_id, conn=conn, clock=clock)
    graph.invoke(Command(resume=decision), run_config, durability="sync")
    snapshot = graph.get_state(run_config)
    if snapshot.interrupts:
        return "parked"
    return snapshot.values.get("outcome", "done")


def parked_rows(conn) -> list[sqlite3.Row]:
    """Everything waiting for a human, oldest first."""
    return conn.execute("SELECT * FROM parked ORDER BY proposed_at ASC").fetchall()


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from src import clock as clockmod, embed
    from src.extract import Fact as F
    from src.policy import BITEMPORAL

    tmp = Path(tempfile.mkdtemp())
    clock = clockmod.FrozenClock("2026-06-01")
    conn = store.connect(tmp / "memory.db")
    june = clock.now()
    row = dict(user_id="u", topic="city", scope="", text="Lives in Pune.", value="Pune",
               volatility="slow", confidence=1.0, valid_from=june, valid_to=config.OPEN_END,
               recorded_at=june, expired_at=config.OPEN_END, superseded_by=None,
               source_kind="conversation", source_ref="t", source_hash="",
               last_verified_at=june, status="active")
    old_id = store.insert_memory(conn, row, embed.embed_one(row["text"]))

    clock.advance(days=90)
    blr = F(topic="city", text="Lives in Bengaluru.", value="Bengaluru", volatility="slow",
            confidence=0.55)
    blr.valid_from_epoch = clock.now()
    plan = RepairPlan(op="supersede", detector="arrival", user_id="u", topic="city", fact=blr,
                      old_id=old_id, before=store.row_to_dict(conn.execute(
                          "SELECT * FROM memories WHERE id=?", (old_id,)).fetchone()),
                      confidence=0.55, reason="unsure", policy="bitemporal")

    # Process A parks it and exits.
    graph_a = build_graph(open_checkpointer(tmp / "checkpoints.db"))
    outcome = launch_repair(graph_a, conn, plan, clock=clock, policy=BITEMPORAL)
    assert outcome == "parked", outcome
    thread_id = make_thread_id(plan, generation=store.generation(conn))
    assert len(parked_rows(conn)) == 1
    assert store.as_of(conn, "u", "city", "", clock.now())["value"] == "Pune", \
        "parking must not touch memory"
    print(f"process A: parked {thread_id}, memory untouched")

    # Process B has never run the graph and reads the card from the checkpoint.
    conn_b = store.connect(tmp / "memory.db")
    graph_b = build_graph(open_checkpointer(tmp / "checkpoints.db"))
    card = load_parked_payload(graph_b, thread_id, conn_b, clock)
    assert card and card["after"]["value"] == "Bengaluru", card
    print(f"process B: read the card -> {card['before']['value']} => {card['after']['value']}")

    result = resume_thread(graph_b, conn_b, thread_id, "approve", clock=clock)
    assert result == "applied", result
    assert store.as_of(conn_b, "u", "city", "", clock.now())["value"] == "Bengaluru"
    assert len(parked_rows(conn_b)) == 0
    applied = conn_b.execute("SELECT COUNT(*) FROM repairs WHERE status='applied'").fetchone()[0]
    assert applied == 1, f"exactly one applied row, got {applied}"

    # A second resume must be a no-op, not a second write.
    resume_thread(graph_b, conn_b, thread_id, "approve", clock=clock)
    assert conn_b.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
    assert not store.check_invariants(conn_b), store.check_invariants(conn_b)

    # A wiped store must not inherit that decision, even though the checkpoint
    # file survives and the repair's content is byte-identical.
    before = make_thread_id(plan, generation=store.generation(conn_b))
    store.wipe(conn_b)
    after = make_thread_id(plan, generation=store.generation(conn_b))
    assert before != after, "a reseeded store must mint fresh thread ids"
    print(f"OK - parked in one process, approved in another, applied exactly once; "
          f"a wipe moves the thread id {before[-6:]} -> {after[-6:]}")
