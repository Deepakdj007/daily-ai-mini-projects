"""What to do about a new fact: the write policy, and the plan it produces.

Inputs:  a fact, its candidates, their relations, and a policy
Outputs: RepairPlans - decisions, with no side effects

A policy is three switches rather than three classes, so the ablation can move
one at a time and a reader can see exactly what separates the incumbent from
what we build:

  resolves     ask the classifier at all, or just pile facts up
  scope_aware  candidates restricted to a compatible scope
  history      retire by superseding (True) or by deleting (False)

`plan()` is a pure decision table. Nothing here touches the database, which is
what lets the same function be tested without one, and what keeps the physics
of each operation in exactly one place (repair.py).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from src import config
from src.classify import Relation
from src.extract import Fact


@dataclass(frozen=True, slots=True)
class WritePolicy:
    """One row of the ladder's write behaviour."""

    name: str
    resolves: bool
    scope_aware: bool
    history: bool


APPEND = WritePolicy("append", resolves=False, scope_aware=False, history=False)
OVERWRITE = WritePolicy("overwrite", resolves=True, scope_aware=False, history=False)
SCOPED = WritePolicy("scoped", resolves=True, scope_aware=True, history=False)
BITEMPORAL = WritePolicy("bitemporal", resolves=True, scope_aware=True, history=True)
POLICIES = {p.name: p for p in (APPEND, OVERWRITE, SCOPED, BITEMPORAL)}


@dataclass(slots=True)
class RepairPlan:
    """One decided change, before anything has been written.

    Carries everything an audit row, an inbox card and the eval all need, so a
    parked plan can be rendered by a process that never saw the conversation.
    """

    op: str                      # insert | supersede | qualify | expire | confirm | noop
    detector: str                # arrival | aged | source_changed | contradiction | scheduled | human
    user_id: str
    topic: str
    scope: str = ""
    fact: Fact | None = None     # the new fact, when there is one
    old_id: int | None = None
    before: dict | None = None
    confidence: float = 1.0
    reason: str = ""
    evidence: str = ""
    case: str = ""               # qualify sub-case: a, b or c
    policy: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def retires_a_row(self) -> bool:
        """True when applying this plan takes an existing memory out of play."""
        return self.op in {"supersede", "expire"} or (self.op == "qualify" and self.case == "a")

    def to_payload(self) -> dict:
        """The self-contained dict an interrupt parks and an inbox card renders."""
        after = None
        if self.fact is not None:
            after = {
                "topic": self.fact.topic, "scope": self.fact.scope, "text": self.fact.text,
                "value": self.fact.value, "volatility": self.fact.volatility,
                "valid_from": self.fact.valid_from_epoch, "valid_to": self.fact.valid_to_epoch,
                "confidence": self.fact.confidence,
            }
        return {
            "op": self.op, "detector": self.detector, "user_id": self.user_id,
            "topic": self.topic, "scope": self.scope, "old_id": self.old_id,
            "before": self.before, "after": after, "confidence": round(self.confidence, 3),
            "reason": self.reason, "evidence": self.evidence[:config.EVIDENCE_MAX_CHARS],
            "case": self.case, "policy": self.policy,
        }


def threshold_for(plan: RepairPlan) -> float:
    """How sure the agent must be to apply this without asking.

    Retiring a row needs more confidence than adding one, because the cost of
    being wrong is a lost fact rather than a duplicate.
    """
    if plan.op == "confirm":
        return config.CONFIRM_MIN_CONFIDENCE
    if plan.retires_a_row():
        return config.AUTO_APPLY_MIN_CONFIDENCE
    return config.QUALIFY_MIN_CONFIDENCE


def route(plan: RepairPlan, *, gate_enabled: bool) -> str:
    """auto, parked or forced.

    `forced` means the agent applied a decision it was not confident about
    because no gate was configured. Recording that separately is what lets the
    evaluation count the repairs a human would have caught without running one.
    """
    if plan.op == "noop":
        return "auto"
    confident = plan.confidence >= threshold_for(plan)
    stable_retire = (
        config.STABLE_ALWAYS_PARKS
        and plan.retires_a_row()
        and (plan.before or {}).get("volatility") == "stable"
    )
    if confident and not stable_retire:
        return "auto"
    return "parked" if gate_enabled else "forced"


def _best(relations: list[Relation], candidates: list[sqlite3.Row]) -> tuple[Relation | None, sqlite3.Row | None]:
    """The relation that decides the outcome, and the row it points at.

    A contradiction outranks a refinement outranks a restatement: the most
    consequential claim wins, so one 'same' among several cannot mask a real
    replacement.
    """
    if not relations:
        return None, None
    rank = {"contradicts": 3, "refines": 2, "same": 1, "unrelated": 0}
    by_id = {int(row["id"]): row for row in candidates}
    ordered = sorted(relations, key=lambda r: (rank.get(r.relation, 0), r.confidence), reverse=True)
    top = ordered[0]
    return top, by_id.get(top.candidate_id)


def plan(fact: Fact, candidates: list[sqlite3.Row], relations: list[Relation],
         *, policy: WritePolicy, user_id: str) -> list[RepairPlan]:
    """Decide what this fact does to the store. Pure - writes nothing."""
    common = dict(detector="arrival", user_id=user_id, topic=fact.topic,
                  scope=fact.scope, fact=fact, policy=policy.name)

    # Without resolution every fact is simply added. This is the tutorial store:
    # nothing is ever compared, so nothing is ever wrong, and everything piles up.
    if not policy.resolves or not candidates:
        return [RepairPlan(op="insert", confidence=fact.confidence,
                           reason="no resolution" if not policy.resolves else "nothing on file",
                           **common)]

    relation, row = _best(relations, candidates)
    if relation is None or row is None:
        return [RepairPlan(op="insert", confidence=fact.confidence,
                           reason="no usable relation", **common)]

    before = dict(row)
    different_scope = bool(row["scope"]) != bool(fact.scope) or row["scope"] != fact.scope
    shared = dict(common, old_id=int(row["id"]), before=before,
                  confidence=relation.confidence, reason=relation.reason)

    if relation.relation == "same":
        if different_scope and fact.scope and row["scope"]:
            return [RepairPlan(op="qualify", case="b", **shared)]
        return [RepairPlan(op="confirm", **shared)]

    if relation.relation == "refines":
        if different_scope and fact.scope and row["scope"]:
            return [RepairPlan(op="qualify", case="b", **shared)]
        return [RepairPlan(op="qualify", case="a", **shared)]

    if relation.relation == "contradicts":
        # Different contexts do not contradict, whatever the values look like.
        # Both stay live under their own scope.
        if different_scope and fact.scope and row["scope"]:
            return [RepairPlan(op="qualify", case="b", **shared)]
        return [RepairPlan(op="supersede", **shared)]

    # Unrelated but sharing a key: give the newcomer a scope of its own rather
    # than letting two different things collide on one key.
    if fact.topic in {"pet", "kids", "project", "appointment", "travel", "other"} or fact.scope:
        return [RepairPlan(op="qualify", case="c", **shared)]
    return [RepairPlan(op="insert", confidence=fact.confidence, reason=relation.reason, **common)]


if __name__ == "__main__":
    from src.extract import Fact as F

    def row(**kw) -> dict:
        base = dict(id=1, user_id="u", topic="city", scope="", text="Lives in Pune.",
                    value="Pune", volatility="slow", status="active")
        return {**base, **kw}

    moved = F(topic="city", text="Lives in Bengaluru.", value="Bengaluru", volatility="slow")
    rel = [Relation(1, "contradicts", 0.99, "different city")]
    decided = plan(moved, [row()], rel, policy=BITEMPORAL, user_id="u")[0]
    assert decided.op == "supersede" and decided.retires_a_row()
    assert route(decided, gate_enabled=True) == "auto", "0.99 is well over the bar"

    unsure = plan(moved, [row()], [Relation(1, "contradicts", 0.55, "maybe")],
                  policy=BITEMPORAL, user_id="u")[0]
    assert route(unsure, gate_enabled=True) == "parked", "a shaky delete must ask"
    assert route(unsure, gate_enabled=False) == "forced", "with no gate it applies, and says so"

    # A contradiction on a stable fact always asks, however confident.
    name = F(topic="name", text="Is called Rohan.", value="Rohan", volatility="stable")
    stable = plan(name, [row(topic="name", value="Rohit", volatility="stable")],
                  [Relation(1, "contradicts", 0.99, "different name")],
                  policy=BITEMPORAL, user_id="u")[0]
    assert route(stable, gate_enabled=True) == "parked", "stable facts always ask"

    # Two contexts never contradict, even when the classifier says they do.
    pref = F(topic="communication_preference", scope="incident calls",
             text="Prefers a voice call for incident calls.", value="voice call", volatility="slow")
    scoped = plan(pref, [row(topic="communication_preference", scope="design reviews",
                             value="async written feedback")],
                  [Relation(1, "contradicts", 0.9, "opposite")], policy=BITEMPORAL, user_id="u")[0]
    assert scoped.op == "qualify" and scoped.case == "b", scoped
    assert not scoped.retires_a_row(), "a scoped exception must never retire the general rule"

    appended = plan(moved, [row()], rel, policy=APPEND, user_id="u")[0]
    assert appended.op == "insert", "the append baseline never resolves anything"
    print("OK - supersede, thresholds, stable-parks, scope exception, append baseline")
