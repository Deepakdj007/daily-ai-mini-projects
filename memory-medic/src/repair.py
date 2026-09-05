"""The only module that changes a memory row, and it always says why.

Inputs:  a RepairPlan, a clock, and the policy in force
Outputs: mutated rows plus exactly one audit row per change

Every operation runs inside one BEGIN IMMEDIATE transaction that covers the
row change, the vector-index update and the audit entry together. Taking the
write lock up front matters because the sweep and the review inbox are two
processes on one file; upgrading a read lock halfway through is how they would
deadlock.

The difference between keeping history and not is one branch in `_supersede`,
and it is the whole argument of the project. With history, the old row's valid
window closes and the row stays. Without it, the row is deleted and the only
surviving copy of what the user used to be true is the audit log's before_json.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass

from src import config, embed, store
from src.policy import RepairPlan, WritePolicy, route


@dataclass(frozen=True, slots=True)
class Applied:
    """What a repair did, for the caller's report."""

    op: str
    routed: str
    repair_id: int
    new_id: int | None = None
    old_id: int | None = None


@contextmanager
def _transaction(conn: sqlite3.Connection):
    """One repair, all or nothing."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _fact_row(plan: RepairPlan, *, at: int, valid_from: int | None = None,
              status: str = "active") -> dict:
    """Turn a planned fact into the columns a memory row needs."""
    fact = plan.fact
    assert fact is not None, "this operation needs a new fact"
    return dict(
        user_id=plan.user_id, topic=fact.topic, scope=fact.scope, text=fact.text,
        value=fact.value, volatility=fact.volatility, confidence=fact.confidence,
        valid_from=valid_from if valid_from is not None else fact.valid_from_epoch,
        valid_to=fact.valid_to_epoch, recorded_at=at, expired_at=config.OPEN_END,
        superseded_by=None, source_kind=plan.extra.get("source_kind", "conversation"),
        source_ref=plan.extra.get("source_ref", ""), source_hash=plan.extra.get("source_hash", ""),
        last_verified_at=at, status=status,
    )


def _audit(conn: sqlite3.Connection, plan: RepairPlan, *, routed: str, status: str,
           decided_by: str, at: int, new_id: int | None, thread_id: str | None = None) -> int:
    """Write the one audit row that explains this change."""
    return store.log_repair(conn, dict(
        user_id=plan.user_id, topic=plan.topic, scope=plan.scope, op=plan.op,
        detector=plan.detector, policy=plan.policy, old_id=plan.old_id, new_id=new_id,
        before_json=json.dumps(plan.before or {}, default=str),
        after_json=json.dumps({**(plan.to_payload().get("after") or {}), "case": plan.case}, default=str),
        evidence=plan.evidence[:config.EVIDENCE_MAX_CHARS], reason=plan.reason,
        confidence=plan.confidence, routed=routed, decided_by=decided_by, status=status,
        thread_id=thread_id, proposed_at=at, decided_at=at if status != "parked" else None,
    ))


def _insert(conn: sqlite3.Connection, plan: RepairPlan, at: int, **kw) -> int:
    """Add a new memory row."""
    fields = _fact_row(plan, at=at, **kw)
    return store.insert_memory(conn, fields, embed.embed_one(fields["text"]))


def _inherit_key(plan: RepairPlan) -> None:
    """A replacement must live on the same key as the fact it replaces.

    The successor of a fact IS that fact, later. If it lands on a different
    (topic, scope) the chain is severed: the old row is closed, the new one is
    filed elsewhere, and a timeline query for the original key shows a fact that
    ended and was never replaced. Observed for real when a document-sourced
    update arrived with scope="employer" against a row stored with scope="".
    """
    before = plan.before or {}
    if not before or plan.fact is None:
        return
    plan.fact.topic = plan.topic = before.get("topic", plan.topic)
    plan.fact.scope = plan.scope = before.get("scope", plan.scope)


def _supersede(conn: sqlite3.Connection, plan: RepairPlan, at: int, policy: WritePolicy) -> int:
    """Replace a fact. Keeps or destroys the old one, depending on the policy."""
    _inherit_key(plan)
    new_id = _insert(conn, plan, at)
    starts = conn.execute("SELECT valid_from FROM memories WHERE id = ?", (new_id,)).fetchone()[0]
    if policy.history:
        # Graphiti's rule: the old fact stopped being true when the new one
        # started, and we stopped believing it now. Both moments are recorded,
        # and the row itself stays queryable forever.
        # The old fact stopped being true when the new one started. If the new
        # fact starts no later than the old one did, the old row's window is
        # zero-length: we believed it, but it was never true of any interval.
        # That is a real outcome and is recorded as such, not nudged by a second.
        old_from = int((plan.before or {}).get("valid_from", starts))
        store.update_row(conn, plan.old_id, valid_to=max(int(starts), old_from),
                         expired_at=at, status="superseded", superseded_by=new_id)
    else:
        store.delete_memory(conn, plan.old_id)
    return new_id


def _qualify(conn: sqlite3.Connection, plan: RepairPlan, at: int, policy: WritePolicy) -> int | None:
    """Three sub-cases that all end with more precision and nothing lost.

    a: a sharper version of the same fact, so the old one steps aside.
    b: a different context, so both stay live under their own scope.
    c: the same key held by two unrelated things, so the newcomer gets a scope.
    """
    if plan.case == "a":
        _inherit_key(plan)  # a refinement is the same fact, sharpened
        before = plan.before or {}
        inherited = min(int(before.get("valid_from", at)), plan.fact.valid_from_epoch)
        if policy.history:
            new_id = _insert(conn, plan, at, valid_from=inherited)
            store.update_row(conn, plan.old_id, expired_at=at, status="superseded",
                             superseded_by=new_id)
            return new_id
        store.update_row(conn, plan.old_id, text=plan.fact.text, value=plan.fact.value,
                         confidence=max(plan.fact.confidence, float(before.get("confidence", 0))),
                         last_verified_at=at)
        return plan.old_id
    if plan.case == "c" and not plan.fact.scope:
        plan.fact.scope = plan.fact.value.strip().lower()[:40]
        plan.scope = plan.fact.scope
    return _insert(conn, plan, at)


def _expire(conn: sqlite3.Connection, plan: RepairPlan, at: int) -> None:
    """Take a fact out of play without claiming it was never true.

    `needs_verification` still retrieves - flagged. `expired` does not: its
    window has closed. Neither deletes anything.
    """
    target = plan.extra.get("to_status", "needs_verification")
    changes = {"status": target}
    if target == "expired" and plan.extra.get("close_window"):
        changes["valid_to"] = at
    store.update_row(conn, plan.old_id, **changes)


def _confirm(conn: sqlite3.Connection, plan: RepairPlan, at: int) -> None:
    """Fresh evidence that a fact still holds: reset its age, clear any flag."""
    before = plan.before or {}
    changes = {"last_verified_at": at, "status": "active",
               "confidence": max(float(before.get("confidence", 0)),
                                 plan.fact.confidence if plan.fact else plan.confidence)}
    if plan.extra.get("source_hash"):
        changes["source_hash"] = plan.extra["source_hash"]
    store.update_row(conn, plan.old_id, **changes)


def apply(conn: sqlite3.Connection, plan: RepairPlan, *, clock, policy: WritePolicy,
          decided_by: str = "agent", routed: str = "auto", thread_id: str | None = None,
          audit: bool = True) -> Applied:
    """Carry out one repair and record it. Returns what happened.

    `audit=False` performs the mutation without opening a new audit row. That
    is for a repair that was parked: its audit row already exists and is waiting
    to be settled, and writing a second one would double-count every reviewed
    decision in the ledger the evaluation reads.
    """
    at = clock.now()
    plan.policy = plan.policy or policy.name
    with _transaction(conn):
        new_id: int | None = None
        if plan.op == "insert":
            new_id = _insert(conn, plan, at)
        elif plan.op == "supersede":
            new_id = _supersede(conn, plan, at, policy)
        elif plan.op == "qualify":
            new_id = _qualify(conn, plan, at, policy)
        elif plan.op == "expire":
            _expire(conn, plan, at)
        elif plan.op == "confirm":
            _confirm(conn, plan, at)
        elif plan.op != "noop":
            raise ValueError(f"unknown op {plan.op!r}")
        repair_id = 0
        if audit:
            repair_id = _audit(conn, plan, routed=routed, status="applied", decided_by=decided_by,
                               at=at, new_id=new_id, thread_id=thread_id)
    return Applied(plan.op, routed, repair_id, new_id, plan.old_id)


def park(conn: sqlite3.Connection, plan: RepairPlan, *, clock, thread_id: str) -> int:
    """Record a repair that is waiting for a human. Changes no memory row.

    Called by the orchestrator after the graph has actually parked, never from
    inside the node: everything before interrupt() re-runs on resume, so a
    write there would be duplicated on every review.
    """
    at = clock.now()
    with _transaction(conn):
        return _audit(conn, plan, routed="parked", status="parked", decided_by="agent",
                      at=at, new_id=None, thread_id=thread_id)


def apply_or_park(conn: sqlite3.Connection, plan: RepairPlan, *, clock, policy: WritePolicy,
                  gate_enabled: bool, launcher=None, oracle_human=None) -> Applied | str:
    """Apply a confident repair, or hand a doubtful one to a human.

    Returns the Applied record, or the routing decision when it parked.
    """
    routed = route(plan, gate_enabled=gate_enabled)
    if routed in {"auto", "forced"}:
        return apply(conn, plan, clock=clock, policy=policy, routed=routed)
    if launcher is None:
        return apply(conn, plan, clock=clock, policy=policy, routed="forced")
    return launcher(conn, plan, clock=clock, policy=policy, oracle_human=oracle_human)


if __name__ == "__main__":
    from src import clock as clockmod
    from src.extract import Fact as F
    from src.policy import BITEMPORAL, OVERWRITE

    def build(conn, clock):
        pune = F(topic="city", text="Lives in Pune.", value="Pune", volatility="slow",
                 confidence=1.0)
        pune.valid_from_epoch = clock.now()
        first = RepairPlan(op="insert", detector="arrival", user_id="u", topic="city", fact=pune)
        return apply(conn, first, clock=clock, policy=BITEMPORAL).new_id

    # With history: the old fact survives, closed, and is still answerable by date.
    clock = clockmod.FrozenClock("2026-06-01")
    conn = store.connect(":memory:")
    old_id = build(conn, clock)
    clock.advance(days=120)
    blr = F(topic="city", text="Lives in Bengaluru.", value="Bengaluru", volatility="slow",
            confidence=1.0)
    blr.valid_from_epoch = clock.now()
    moved = RepairPlan(op="supersede", detector="arrival", user_id="u", topic="city", fact=blr,
                       old_id=old_id, before=store.row_to_dict(
                           conn.execute("SELECT * FROM memories WHERE id=?", (old_id,)).fetchone()),
                       confidence=0.99, reason="different city")
    applied = apply(conn, moved, clock=clock, policy=BITEMPORAL)
    assert not store.check_invariants(conn), store.check_invariants(conn)
    assert store.as_of(conn, "u", "city", "", clockmod.to_epoch("2026-07-01"))["value"] == "Pune"
    assert store.as_of(conn, "u", "city", "", clock.now())["value"] == "Bengaluru"
    assert len(store.history(conn, "u", "city", "")) == 2
    audit = conn.execute("SELECT * FROM repairs WHERE op='supersede'").fetchone()
    assert json.loads(audit["before_json"])["value"] == "Pune"
    print(f"history=True  -> both rows kept, as_of(July)=Pune, now=Bengaluru, audit #{applied.repair_id}")

    # Without history: the same input destroys the old row. Only the audit remembers.
    clock2 = clockmod.FrozenClock("2026-06-01")
    conn2 = store.connect(":memory:")
    old2 = build(conn2, clock2)
    clock2.advance(days=120)
    blr2 = F(topic="city", text="Lives in Bengaluru.", value="Bengaluru", volatility="slow",
             confidence=1.0)
    blr2.valid_from_epoch = clock2.now()
    apply(conn2, RepairPlan(op="supersede", detector="arrival", user_id="u", topic="city",
                            fact=blr2, old_id=old2, confidence=0.99,
                            before=store.row_to_dict(conn2.execute(
                                "SELECT * FROM memories WHERE id=?", (old2,)).fetchone())),
          clock=clock2, policy=OVERWRITE)
    assert len(store.history(conn2, "u", "city", "")) == 1, "overwrite keeps one row"
    assert store.as_of(conn2, "u", "city", "", clockmod.to_epoch("2026-07-01")) is None, \
        "and July is now unanswerable"
    recovered = json.loads(conn2.execute(
        "SELECT before_json FROM repairs WHERE op='supersede'").fetchone()[0])
    assert recovered["value"] == "Pune"
    print("history=False -> old row deleted, July unanswerable, value survives only in the audit log")
    print("OK - the same input, two stores, one of them can no longer answer a question about the past")
