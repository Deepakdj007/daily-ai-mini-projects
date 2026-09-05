"""Turning a sweep candidate into a decided repair.

Inputs:  a Candidate from detect.py, plus the store
Outputs: RepairPlans ready to apply or park

Only two of the four detectors need a model. Ageing and expiry are arithmetic
on dates and are decided in sweep.py without spending a token. Source drift and
contradictions need judgement about meaning, so they come here.

Re-verification is the interesting one: the source changed, so we re-extract
from the new text and compare per key. A fact still stated with the same value
is confirmed - its age resets and nothing else happens. A fact stated with a
different value goes through the same classifier the conversation path uses. A
fact the source no longer mentions at all is not deleted; it stops being
verifiable, which is a different claim and is recorded as one.
"""

from __future__ import annotations

import sqlite3

from src import classify, config, detect, extract, store
from src.extract import Fact
from src.policy import RepairPlan, WritePolicy


def _row(conn: sqlite3.Connection, memory_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()


def _fact_from_row(row: sqlite3.Row) -> Fact:
    """A stored row expressed as a Fact, so plans can carry it unchanged."""
    fact = Fact(topic=row["topic"], scope=row["scope"], text=row["text"], value=row["value"],
                volatility=row["volatility"], confidence=float(row["confidence"]))
    fact.valid_from_epoch = int(row["valid_from"])
    fact.valid_to_epoch = int(row["valid_to"])
    return fact


def expire_plan(conn: sqlite3.Connection, candidate: detect.Candidate, *, to_status: str,
                close_window: bool = False) -> RepairPlan | None:
    """Flag an aged fact, or close an ended one. No model, no judgement call."""
    row = _row(conn, candidate.primary)
    if row is None:
        return None
    return RepairPlan(
        op="expire", detector=candidate.kind, user_id=row["user_id"], topic=row["topic"],
        scope=row["scope"], old_id=int(row["id"]), before=store.row_to_dict(row),
        confidence=1.0, reason=("its shelf life has passed with no fresh evidence"
                                if to_status == "needs_verification" else "its window has closed"),
        evidence=candidate.evidence,
        extra={"to_status": to_status, "close_window": close_window},
    )


async def reverify(conn: sqlite3.Connection, candidate: detect.Candidate, *,
                   policy: WritePolicy, model: str = "", clock=None) -> list[RepairPlan]:
    """Re-read a changed source and decide what each grounded fact should do."""
    rows = [row for row in (_row(conn, mid) for mid in candidate.memory_ids) if row is not None]
    if not rows:
        return []

    if candidate.payload.get("gone"):
        return [RepairPlan(
            op="expire", detector="source_changed", user_id=row["user_id"], topic=row["topic"],
            scope=row["scope"], old_id=int(row["id"]), before=store.row_to_dict(row),
            confidence=1.0, reason="the source that stated this has been deleted",
            evidence=candidate.evidence, extra={"to_status": "needs_verification"},
        ) for row in rows]

    content = candidate.payload.get("content", "")
    said_at = clock.now() if clock is not None else int(rows[0]["last_verified_at"])
    fresh = await extract.extract(content, said_at=said_at, model=model, source_kind="fixture")
    by_key = {(fact.topic, fact.scope): fact for fact in fresh.facts}
    # A document's field label is a more reliable join than its topic: the model
    # may file "Office: ..." under `other` one day and `address` the next, and a
    # topic mismatch would read as "the source dropped this field".
    by_scope = {fact.scope: fact for fact in fresh.facts if fact.scope}
    # Last resort: a stored row with no scope at all, against exactly one
    # extracted fact on the same topic. Rows seeded before this project started
    # labelling document facts by field have empty scopes, and without this they
    # match nothing and are reported as "the source no longer states this" -
    # which is how a working re-verification quietly became a no-op.
    topic_counts: dict[str, int] = {}
    for fact in fresh.facts:
        topic_counts[fact.topic] = topic_counts.get(fact.topic, 0) + 1
    by_topic = {fact.topic: fact for fact in fresh.facts if topic_counts[fact.topic] == 1}
    source_hash = candidate.payload.get("sha256", "")

    plans: list[RepairPlan] = []
    for row in rows:
        key = (row["topic"], row["scope"])
        exact = by_key.get(key) or (by_scope.get(row["scope"]) if row["scope"] else None)
        fact = exact or by_key.get((row["topic"], "")) or by_topic.get(row["topic"])
        exact = exact or (fact if fact is not None and row["scope"] == "" else None)
        common = dict(detector="source_changed", user_id=row["user_id"], topic=row["topic"],
                      scope=row["scope"], old_id=int(row["id"]), before=store.row_to_dict(row),
                      evidence=candidate.evidence, policy=policy.name)

        if fact is None:
            plans.append(RepairPlan(
                op="expire", fact=_fact_from_row(row), confidence=1.0,
                reason="the source no longer states this", extra={"to_status": "needs_verification"},
                **common))
            continue

        fact.valid_from_epoch = max(fact.valid_from_epoch, said_at)
        if fact.value.strip().lower() == row["value"].strip().lower():
            plans.append(RepairPlan(
                op="confirm", fact=fact, confidence=1.0,
                reason="the source still states this value",
                extra={"source_hash": source_hash}, **common))
            continue

        if exact is not None:
            # The same field of the same record now says something else. That is
            # a replacement by construction, and asking a model to re-derive it
            # is both a wasted call and less reliable: on this fixture the
            # classifier called "Reports to Anil Varghese" and "Manager is Sudha
            # Menon" different attributes and let a stale manager stand.
            plans.append(RepairPlan(
                op="supersede", fact=fact, confidence=0.95,
                reason=f"the source now records {fact.value!r} for this field",
                extra={"source_kind": "fixture", "source_ref": candidate.source_ref,
                       "source_hash": source_hash}, **common))
            continue

        relations, error = await classify.classify(fact, [row], model=model)
        relation = relations[0] if relations else None
        if error or relation is None:
            plans.append(RepairPlan(
                op="expire", fact=fact, confidence=0.0,
                reason=f"the source changed and the comparison failed ({error or 'no relation'})",
                extra={"to_status": "needs_verification"}, **common))
            continue
        if relation.relation in {"contradicts", "refines"}:
            plans.append(RepairPlan(
                op="supersede" if relation.relation == "contradicts" else "qualify",
                case="" if relation.relation == "contradicts" else "a",
                fact=fact, confidence=relation.confidence, reason=relation.reason,
                extra={"source_kind": "fixture", "source_ref": candidate.source_ref,
                       "source_hash": source_hash}, **common))
        else:
            plans.append(RepairPlan(
                op="confirm", fact=fact, confidence=relation.confidence,
                reason=relation.reason, extra={"source_hash": source_hash}, **common))
    return plans


async def contradiction(conn: sqlite3.Connection, candidate: detect.Candidate, *,
                        policy: WritePolicy, model: str = "") -> RepairPlan | None:
    """Two live facts on one key: ask which replaces which, and keep the newer."""
    rows = [row for row in (_row(conn, mid) for mid in candidate.memory_ids) if row is not None]
    if len(rows) < 2:
        return None
    rows.sort(key=lambda row: (int(row["valid_from"]), int(row["id"])))
    older, newer = rows[0], rows[-1]

    fact = _fact_from_row(newer)
    relations, error = await classify.classify(fact, [older], model=model)
    relation = relations[0] if relations else None
    common = dict(detector="contradiction", user_id=older["user_id"], topic=older["topic"],
                  scope=older["scope"], old_id=int(older["id"]),
                  before=store.row_to_dict(older), evidence=candidate.evidence,
                  policy=policy.name, fact=fact,
                  extra={"newer_id": int(newer["id"]), "case": candidate.payload.get("case", "")})

    if error or relation is None:
        return RepairPlan(op="supersede", confidence=0.0,
                          reason=f"two live facts on one key; comparison failed ({error})", **common)
    if relation.relation in {"same", "refines"}:
        # A duplicate rather than a change: retire the older row in favour of
        # the newer, which is already in the store.
        return RepairPlan(op="expire", confidence=relation.confidence,
                          reason=f"duplicate of #{newer['id']}: {relation.reason}",
                          **{**common, "extra": {**common["extra"], "to_status": "superseded"}})
    if relation.relation == "unrelated":
        return RepairPlan(op="noop", confidence=relation.confidence,
                          reason=f"both can stand: {relation.reason}", **common)
    return RepairPlan(op="supersede", confidence=relation.confidence, reason=relation.reason,
                      **common)


if __name__ == "__main__":
    import asyncio

    from src import clock as clockmod, embed, repair
    from src.policy import BITEMPORAL

    async def _smoke() -> None:
        config.SOURCES_DIR.mkdir(parents=True, exist_ok=True)
        path = config.SOURCES_DIR / "_probe_profile.md"
        path.write_text("# Profile\n\nEmployer: Zeta Retail\nCity: Pune\n", encoding="utf-8")

        clock = clockmod.FrozenClock("2026-06-01")
        conn = store.connect(":memory:")
        first = await extract.extract(path.read_text(encoding="utf-8"), said_at=clock.now(),
                                      source_kind="fixture")
        print("seeded keys:", [(f.topic, f.scope, f.value) for f in first.facts])
        for fact in first.facts:
            plan = RepairPlan(op="insert", detector="arrival", user_id="u", topic=fact.topic,
                              scope=fact.scope, fact=fact, policy="bitemporal",
                              extra={"source_kind": "fixture", "source_ref": "fixture:_probe_profile.md"})
            repair.apply(conn, plan, clock=clock, policy=BITEMPORAL)
        store.upsert_source(conn, "fixture:_probe_profile.md", "fixture",
                            detect.sha256(path.read_text(encoding="utf-8")),
                            path.read_text(encoding="utf-8"), clock.now())
        print(f"seeded {len(first.facts)} facts from the source")

        # Nobody says anything. The source quietly changes.
        clock.advance(days=45)
        path.write_text("# Profile\n\nEmployer: Nimbus Logistics\nCity: Pune\n", encoding="utf-8")
        found = detect.all_candidates(conn, clock=clock)
        assert len(found.source_changed) == 1, found.counts()

        plans = await reverify(conn, found.source_changed[0], policy=BITEMPORAL, clock=clock)
        by_topic = {p.topic: p for p in plans}
        print(f"reverify -> {sorted(f'{p.topic}/{p.scope}:{p.op}({p.confidence:.2f})' for p in plans)}")
        assert by_topic["employer"].op == "supersede", \
            f"a changed field must supersede, got {by_topic['employer'].op}"
        assert by_topic["city"].op == "confirm", "an unchanged line must be confirmed, not rewritten"

        for plan in plans:
            repair.apply(conn, plan, clock=clock, policy=BITEMPORAL)
        now = store.as_of(conn, "u", "employer", "employer", clock.now())
        before = store.as_of(conn, "u", "employer", "employer", clockmod.to_epoch("2026-06-15"))
        assert now and now["value"] == "Nimbus Logistics", now and now["value"]
        assert before and before["value"] == "Zeta Retail", "the old value stays dated"
        assert not store.check_invariants(conn), store.check_invariants(conn)
        path.unlink(missing_ok=True)
        print("OK - a source changed with nobody saying a word, and the old value is still dated")

    asyncio.run(_smoke())
