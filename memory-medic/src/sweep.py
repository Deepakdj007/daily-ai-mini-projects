"""The hygiene sweep: one pass over the store, looking for facts that rotted.

Inputs:  the store, a clock, and the behaviour switches
Outputs: a SweepReport, and whatever repairs it decided to make

The shape of a tick matters more than any single detector. Detection is pure
SQL, so a sweep over a store where nothing has drifted costs a few indexed
queries and returns immediately, without a single model call. That early return
is what makes an always-on sweep affordable on a free tier - and it is the same
discipline that stops a watcher from re-asking about a repair somebody already
rejected.

Ageing and expiry never ask a human. Flagging a fact as needing verification is
reversible and cheap; interrupting somebody about it every time a month passes
would empty the inbox of anything worth reading.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field

from src import config, detect, llm, propose, repair
from src.policy import POLICIES, WritePolicy


@dataclass(slots=True)
class SweepReport:
    """What one pass found and did."""

    candidates: dict[str, int] = field(default_factory=dict)
    applied: int = 0
    parked: int = 0
    forced: int = 0
    noop: int = 0
    live_calls: int = 0
    replayed_calls: int = 0
    actions: list[str] = field(default_factory=list)

    @property
    def total_candidates(self) -> int:
        return sum(self.candidates.values())

    def __bool__(self) -> bool:
        return bool(self.applied or self.parked or self.forced)


async def tick(conn: sqlite3.Connection, *, clock, policy: WritePolicy | None = None,
               gate_enabled: bool = True, graph=None, oracle_human=None,
               model: str = "", dry_run: bool = False) -> SweepReport:
    """One sweep. Returns immediately, and free, when nothing has drifted."""
    policy = policy or POLICIES[config.WRITE_POLICY]
    before = llm.stats()

    found = detect.all_candidates(conn, clock=clock)
    report = SweepReport(candidates=found.counts())
    if not found or dry_run:
        return report

    def _launch(store_conn, plan, *, clock, policy, oracle_human):
        from src.graph import launch_repair

        return launch_repair(graph, store_conn, plan, clock=clock, policy=policy,
                             oracle_human=oracle_human)

    def _settle(plan, outcome) -> None:
        if isinstance(outcome, str):
            if outcome == "parked":
                report.parked += 1
            elif outcome in {"applied", "rejected", "done"}:
                report.applied += 1
            report.actions.append(f"{plan.op} {plan.topic}/{plan.scope or '-'} -> {outcome}")
            return
        if plan.op == "noop":
            report.noop += 1
        elif outcome.routed == "forced":
            report.forced += 1
        else:
            report.applied += 1
        report.actions.append(
            f"{plan.op} {plan.topic}/{plan.scope or '-'} ({outcome.routed}) {plan.reason}"[:160]
        )

    # Rules first: no model, no human, no judgement.
    for candidate in found.aged:
        plan = propose.expire_plan(conn, candidate, to_status="needs_verification")
        if plan is not None:
            _settle(plan, repair.apply(conn, plan, clock=clock, policy=policy))
    for candidate in found.scheduled:
        plan = propose.expire_plan(conn, candidate, to_status="expired")
        if plan is not None:
            _settle(plan, repair.apply(conn, plan, clock=clock, policy=policy))

    # Then the two that need judgement.
    for candidate in found.source_changed:
        for plan in await propose.reverify(conn, candidate, policy=policy, model=model, clock=clock):
            _settle(plan, repair.apply_or_park(
                conn, plan, clock=clock, policy=policy, gate_enabled=gate_enabled,
                launcher=_launch if graph is not None else None, oracle_human=oracle_human))
    for candidate in found.contradiction:
        plan = await propose.contradiction(conn, candidate, policy=policy, model=model)
        if plan is not None:
            _settle(plan, repair.apply_or_park(
                conn, plan, clock=clock, policy=policy, gate_enabled=gate_enabled,
                launcher=_launch if graph is not None else None, oracle_human=oracle_human))

    after = llm.stats()
    report.live_calls = after["live"] - before["live"]
    report.replayed_calls = after["replayed"] - before["replayed"]
    return report


async def run_watch(conn: sqlite3.Connection, *, clock, interval: float, console=None, **kwargs):
    """Sweep on a timer until interrupted."""
    while True:
        report = await tick(conn, clock=clock, **kwargs)
        if console is not None and report:
            console.print(f"[green]{report.applied} applied, {report.parked} parked[/] "
                          f"from {report.total_candidates} candidates")
            for action in report.actions:
                console.print(f"  [dim]{action}[/]")
        await asyncio.sleep(interval)


if __name__ == "__main__":
    from src import clock as clockmod, embed, store
    from src.policy import BITEMPORAL

    async def _smoke() -> None:
        conn = store.connect(":memory:")
        clock = clockmod.FrozenClock("2026-06-01")
        june = clock.now()
        base = dict(user_id="u", scope="", confidence=1.0, source_kind="conversation",
                    source_ref="", source_hash="", expired_at=config.OPEN_END,
                    superseded_by=None, valid_to=config.OPEN_END, valid_from=june,
                    recorded_at=june, last_verified_at=june, status="active")
        rows = [
            dict(base, topic="project", text="Working on the payments migration.",
                 value="payments migration", volatility="fast"),
            dict(base, topic="city", text="Lives in Pune.", value="Pune", volatility="slow"),
            dict(base, topic="appointment", scope="dentist", text="Dentist on 20 June.",
                 value="2026-06-20", volatility="scheduled",
                 valid_to=clockmod.to_epoch("2026-06-20")),
        ]
        for row, vector in zip(rows, embed.embed_texts([r["text"] for r in rows])):
            store.insert_memory(conn, row, vector)

        quiet = await tick(conn, clock=clock, policy=BITEMPORAL)
        assert quiet.total_candidates == 0 and quiet.live_calls == 0
        print(f"day 0  -> {quiet.candidates}, {quiet.live_calls} model calls")

        clock.advance(days=60)
        worked = await tick(conn, clock=clock, policy=BITEMPORAL)
        assert worked.applied == 2, worked.actions
        assert worked.live_calls == 0, "ageing and expiry are arithmetic, not judgement"
        for action in worked.actions:
            print(f"day 60 -> {action}")

        statuses = {row["topic"]: row["status"] for row in
                    conn.execute("SELECT topic, status FROM memories").fetchall()}
        assert statuses["project"] == "needs_verification", statuses
        assert statuses["appointment"] == "expired", statuses
        assert statuses["city"] == "active", "a slow fact at 60 days is not stale yet"

        # Sweeping again must find nothing: the flags it set are not new candidates.
        again = await tick(conn, clock=clock, policy=BITEMPORAL)
        assert again.total_candidates == 0, f"a sweep must settle down: {again.candidates}"
        assert not store.check_invariants(conn), store.check_invariants(conn)
        print(f"day 60 again -> {again.candidates} (idempotent, {again.live_calls} calls)")
        print("OK - quiet when nothing rots, specific when something does, and it settles")

    asyncio.run(_smoke())
