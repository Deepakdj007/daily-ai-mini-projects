"""The three nodes of the repair gate, and the rule about where writes go.

Inputs:  a RepairState carrying one plan
Outputs: state updates; the apply node is the only one that changes memory

Everything before interrupt() runs again when the graph resumes. That is not a
quirk to work around, it is the execution model: the node restarts from the top
with the human's answer in hand. So the gate node does nothing but park, the
parked-repair index row is written by the caller after invoke() returns, and
the memory change lives in a node that only runs after a decision exists.

A missing or malformed decision is treated as a rejection. The alternative -
defaulting to approve - would mean a crashed reviewer silently rewrites memory.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, TypedDict

from langgraph.config import get_config
from langgraph.types import interrupt

from src import config, repair, store
from src.extract import Fact
from src.policy import POLICIES, RepairPlan


class RepairState(TypedDict, total=False):
    """One repair's journey through the gate."""

    thread_id: str
    plan: dict
    decision: str
    value: str
    text: str
    scopes: list[str]
    outcome: str
    revision: int
    history: Annotated[list[str], operator.add]


def plan_from_payload(payload: dict) -> RepairPlan:
    """Rebuild a RepairPlan from the dict that was parked.

    The payload is self-contained by design: a reviewing process that never saw
    the conversation can render the card and apply the result from this alone.
    """
    after = payload.get("after") or {}
    fact = None
    if after:
        fact = Fact(
            topic=after.get("topic", payload.get("topic", "other")),
            scope=after.get("scope", ""), text=after.get("text", ""),
            value=after.get("value", ""), volatility=after.get("volatility", "slow"),
            confidence=float(after.get("confidence", 0.5)),
        )
        fact.valid_from_epoch = int(after.get("valid_from", 0))
        fact.valid_to_epoch = int(after.get("valid_to", config.OPEN_END))
    return RepairPlan(
        op=payload.get("op", "noop"), detector=payload.get("detector", "human"),
        user_id=payload.get("user_id", ""), topic=payload.get("topic", ""),
        scope=payload.get("scope", ""), fact=fact, old_id=payload.get("old_id"),
        before=payload.get("before"), confidence=float(payload.get("confidence", 0.0)),
        reason=payload.get("reason", ""), evidence=payload.get("evidence", ""),
        case=payload.get("case", ""), policy=payload.get("policy", config.WRITE_POLICY),
        extra=payload.get("extra", {}) or {},
    )


def gate_node(state: RepairState) -> dict:
    """Freeze the run and wait for a person. Deliberately side-effect free."""
    payload = interrupt({**state["plan"], "thread_id": state.get("thread_id", ""),
                         "revision": state.get("revision", 1)})
    if not isinstance(payload, dict):
        payload = {}
    decision = str(payload.get("decision", "reject")).lower()
    return {
        "decision": decision,
        "value": str(payload.get("value", "") or ""),
        "text": str(payload.get("text", "") or ""),
        "scopes": list(payload.get("scopes") or []),
        "history": [f"human said {decision}"],
    }


def route_after_gate(state: RepairState) -> str:
    """Approve, edit and keep-both all end in a write; anything else does not."""
    return "apply" if state.get("decision") in {"approve", "edit", "keep_both"} else "reject"


def _resources() -> tuple[Any, Any]:
    """The store connection and clock the runner passed through configurable.

    Read with get_config() rather than a second node parameter: LangGraph
    injects by parameter NAME, and a parameter called `config` would shadow this
    project's config module inside the very functions that need it.
    """
    configurable = (get_config() or {}).get("configurable") or {}
    conn = configurable.get("conn")
    if conn is None:
        raise RuntimeError("the graph needs a store connection in configurable['conn']")
    from src import clock as clockmod

    return conn, configurable.get("clock") or clockmod.SystemClock()


def apply_node(state: RepairState) -> dict:
    """Carry out the approved repair, exactly once.

    The already-applied check guards the window between our commit and
    LangGraph's checkpoint write. They are different files, so a crash between
    them would otherwise replay the repair on the next resume.
    """
    conn, clock = _resources()
    thread_id = state.get("thread_id", "")

    settled = conn.execute(
        "SELECT status FROM repairs WHERE thread_id = ? ORDER BY id DESC LIMIT 1", (thread_id,)
    ).fetchone()
    if settled is not None and settled["status"] == "applied":
        return {"outcome": "applied", "history": ["already applied; nothing re-run"]}

    plan = plan_from_payload(state["plan"])
    decision = state.get("decision", "approve")

    if decision == "edit" and plan.fact is not None:
        if state.get("value"):
            plan.fact.value = state["value"]
            plan.fact.text = state.get("text") or f"{plan.fact.text.rstrip('.')}: {state['value']}."
        plan.reason = f"{plan.reason} (edited by reviewer)".strip()
    if decision == "keep_both":
        # The human says the two facts hold in different situations, so this
        # stops being a replacement and becomes a scoped exception.
        scopes = state.get("scopes") or []
        plan.op, plan.case = "qualify", "b"
        if plan.fact is not None and scopes:
            plan.fact.scope = scopes[-1].strip().lower()[:40]
            plan.scope = plan.fact.scope
        if plan.old_id and len(scopes) > 1 and scopes[0].strip():
            store.update_row(conn, plan.old_id, scope=scopes[0].strip().lower()[:40])

    policy = POLICIES.get(plan.policy, POLICIES[config.WRITE_POLICY])
    # audit=False: this repair's audit row already exists, parked, and is about
    # to be settled. A second row would double-count every reviewed decision.
    applied = repair.apply(conn, plan, clock=clock, policy=policy, decided_by="human",
                           routed="parked", thread_id=thread_id, audit=False)
    store.settle_repair(conn, thread_id, "applied", decided_by="human",
                        decided_at=clock.now(), new_id=applied.new_id)
    return {"outcome": "applied", "history": [f"applied {plan.op}"]}


def reject_node(state: RepairState) -> dict:
    """Record the refusal, and leave the store in a state that stops re-asking.

    A rejected repair whose trigger is still present would be re-detected on the
    next sweep and parked again, every interval, for ever. Each detector needs
    its own way of settling down.
    """
    conn, clock = _resources()
    thread_id = state.get("thread_id", "")
    plan = plan_from_payload(state["plan"])
    now = clock.now()

    if plan.detector == "contradiction" and plan.old_id:
        # The human kept the older fact, so the newer duplicate steps aside.
        newer = (plan.extra or {}).get("newer_id")
        if newer:
            store.update_row(conn, int(newer), status="superseded", expired_at=now,
                             superseded_by=plan.old_id)
    elif plan.detector == "source_changed" and plan.old_id:
        # "Keep it despite the change": accept the new snapshot as verified so
        # the diff does not fire again.
        changes = {"last_verified_at": now, "status": "active"}
        if (plan.extra or {}).get("source_hash"):
            changes["source_hash"] = plan.extra["source_hash"]
        store.update_row(conn, plan.old_id, **changes)

    store.settle_repair(conn, thread_id, "rejected", decided_by="human", decided_at=now)
    return {"outcome": "rejected", "history": ["rejected"]}


if __name__ == "__main__":
    payload = {
        "op": "supersede", "detector": "arrival", "user_id": "u", "topic": "city",
        "scope": "", "old_id": 3, "before": {"value": "Pune", "volatility": "slow"},
        "after": {"topic": "city", "scope": "", "text": "Lives in Bengaluru.",
                  "value": "Bengaluru", "volatility": "slow", "valid_from": 1,
                  "valid_to": config.OPEN_END, "confidence": 0.6},
        "confidence": 0.6, "reason": "different city", "evidence": "user said so", "case": "",
        "policy": "bitemporal",
    }
    rebuilt = plan_from_payload(payload)
    assert rebuilt.op == "supersede" and rebuilt.fact.value == "Bengaluru"
    assert rebuilt.old_id == 3 and rebuilt.before["value"] == "Pune"
    assert json.loads(json.dumps(payload)) == payload, "a payload must survive a checkpoint"

    assert route_after_gate({"decision": "approve"}) == "apply"
    assert route_after_gate({"decision": "edit"}) == "apply"
    assert route_after_gate({"decision": "keep_both"}) == "apply"
    assert route_after_gate({"decision": "reject"}) == "reject"
    assert route_after_gate({}) == "reject", "a missing decision must fail closed"
    assert route_after_gate({"decision": "banana"}) == "reject"
    print("OK - payload round-trips, and anything that is not an approval is a refusal")
