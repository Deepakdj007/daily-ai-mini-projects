"""Running the ladder: build a store per arm, ask every probe, record everything.

Inputs:  an arm list and a fixture
Outputs: output/results-<model>.json - one row per (arm, probe), plus a manifest

Each arm gets its own store, built from the identical fixture by the identical
code path the real agent uses. The only thing that differs is the switch map.
Seed facts are written directly, which is free and deterministic; only the four
announced changes go through extraction, and those replay from cache after the
first arm, so the ladder costs answers rather than answers plus writes.

The leak check runs on the assembled memory block, not on the answer: if an arm
that is supposed to be blind to a value can see it in its own prompt, the
scorecard is measuring nothing and the run is void.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import date

from src import cache, config, detect, embed, fixture as fixmod, llm, matcher
from src import clock as clockmod
from src import repair, score, store, sweep, turn
from src.fixture import Fixture, Probe
from src.ladder import BY_NAME, LADDER, PROFILES, Arm

LADDER_ORDER = {arm.name: index for index, arm in enumerate(LADDER)}
from src.policy import RepairPlan


def _seed_store(conn, fixture: Fixture, *, clock) -> None:
    """Plant the fixture's facts directly, at the dates they were learned."""
    rows = []
    for fact in fixture.facts:
        at = int(fact["at"])
        rows.append(dict(
            user_id="u", topic=fact["topic"], scope=fact["scope"], text=fact["text"],
            value=fact["value"], volatility=fact["volatility"], confidence=1.0,
            valid_from=at, valid_to=int(fact.get("valid_to", config.OPEN_END)),
            recorded_at=at, expired_at=config.OPEN_END, superseded_by=None,
            source_kind=fact.get("source_kind", "conversation"),
            source_ref=fact.get("source_ref", f"turn:{clockmod.to_iso(at)}"),
            source_hash="", last_verified_at=at, status="active",
        ))
    vectors = embed.embed_texts([row["text"] for row in rows])
    for row, vector in zip(rows, vectors):
        store.insert_memory(conn, row, vector)


def _write_sources(fixture: Fixture, version: int) -> None:
    """Put one version of the source files on disk."""
    config.SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    bodies = fixture.sources_v1 if version == 1 else fixture.sources_v2
    for name, body in bodies.items():
        (config.SOURCES_DIR / name).write_text(body, encoding="utf-8")


def _register_sources(conn, fixture: Fixture, *, at: int) -> None:
    """Record the snapshot the seeded facts were extracted from."""
    for name, body in fixture.sources_v1.items():
        store.upsert_source(conn, f"fixture:{name}", "fixture", detect.sha256(body), body, at)


async def run_arm(arm: Arm, fixture: Fixture, *, model: str, db_path) -> list[dict]:
    """Build this arm's store from the fixture, then ask it every probe."""
    conn = store.connect(db_path)
    store.wipe(conn)
    clock = clockmod.FrozenClock(fixture.probe_at)
    probe_at = clock.now()
    write_policy = arm.config.policy()

    if arm.config.memory:
        _write_sources(fixture, 1)
        _seed_store(conn, fixture, clock=clock)
        _register_sources(conn, fixture, at=probe_at - 320 * 86_400)

        # Oldest first, so the original fact is on file before its replacement
        # arrives and the write policy has something to resolve against.
        for change in sorted(fixture.changes, key=lambda c: -c.at_days):
            said_at = probe_at - change.at_days * 86_400
            said_clock = clockmod.FrozenClock(said_at)
            await turn.ingest(conn, "u", change.message, clock=said_clock,
                              write_policy=write_policy, gate_enabled=False, model=model,
                              said_at=said_at)

        # The source files change with nobody saying a word. Only an arm that
        # goes looking will ever find out.
        _write_sources(fixture, 2)
        if arm.config.sweep:
            await sweep.tick(conn, clock=clock, policy=write_policy, gate_enabled=False,
                             model=model)

    rows: list[dict] = []
    for probe in fixture.probes:
        reply, hits, block = await turn.answer(
            conn, "u", probe.question, clock=clock, as_of=probe_at, model=model,
            show_freshness=arm.config.freshness, window=arm.config.history,
            use_memory=arm.config.memory,
        )
        checks = score.score(probe, reply)
        leaked = _leak(arm, probe, block)
        rows.append(dict(
            arm=arm.name, probe=probe.key, category=probe.category, measures=probe.measures,
            question=probe.question, gold=probe.gold, stale=probe.stale,
            answer=matcher.extract_answer(reply), raw=reply.strip()[:400],
            checks=checks, passed=score.passed(checks),
            hedged=matcher.hedged(reply), abstained=matcher.abstained(reply),
            gold_in_block=bool(probe.gold) and matcher.matches(probe.gold, block),
            stale_in_block=bool(probe.stale) and matcher.matches(probe.stale, block),
            leaked=leaked, block_chars=len(block), hits=len(hits),
        ))
    conn.close()
    return rows


def _leak(arm: Arm, probe: Probe, block: str) -> str:
    """Catch a value reaching an arm that should structurally be blind to it."""
    if not arm.config.memory and probe.category not in {"in_message", "negative"}:
        if probe.gold and matcher.matches(probe.gold, block):
            return "no-memory arm saw the answer"
    if probe.category == "source_drift" and not arm.config.sweep and probe.gold:
        if matcher.matches(probe.gold, block):
            return "an arm that never re-read the source saw its new value"
    if not arm.config.freshness and matcher.HEDGE_RE.search(block):
        return "a freshness note reached an arm with freshness off"
    return ""


async def run(arms: list[str], *, model: str = "", profile: str = "lite") -> dict:
    """Run every named arm and write the results file."""
    model = model or config.CHAT_MODEL
    built = fixmod.build()
    problems = fixmod.gates(built)
    if problems:
        raise SystemExit(f"fixture gates failed, refusing to run: {problems}")

    llm.reset_stats()
    started_tokens, _ = cache.spent_today()
    rows: list[dict] = []
    for name in arms:
        arm = BY_NAME[name]
        db_path = config.OUTPUT_DIR / f"arm-{name}.db"
        db_path.unlink(missing_ok=True)
        rows.extend(await run_arm(arm, built, model=model, db_path=db_path))
        db_path.unlink(missing_ok=True)
        print(f"  {name:12} done ({len([r for r in rows if r['arm'] == name])} probes)")

    # Arms run on an earlier day are kept, so a ladder can be built up across
    # several free-tier refills. Only ever merged when the fixture and the
    # prompts are identical: results from two different experiments in one table
    # would be worse than no table.
    carried: list[dict] = []
    existing_path = config.results_path(model)
    if existing_path.exists():
        previous = json.loads(existing_path.read_text(encoding="utf-8"))
        same_experiment = (
            previous.get("manifest", {}).get("fixture") == built.digest()
            and previous.get("manifest", {}).get("prompt_version") == config.PROMPT_VERSION
            and previous.get("manifest", {}).get("answer_model") == model
        )
        if same_experiment:
            carried = [row for row in previous.get("results", []) if row["arm"] not in arms]
    if carried:
        carried_arms = sorted({row["arm"] for row in carried})
        print(f"  carrying {len(carried)} rows from {carried_arms} (same fixture)")
    rows = carried + rows
    ordered = [arm.name for arm in
               sorted({BY_NAME[r["arm"]] for r in rows}, key=lambda a: LADDER_ORDER[a.name])]

    spent_tokens, spent_requests = cache.spent_today()
    payload = {
        "manifest": {
            "date": date.today().isoformat(), "answer_model": model,
            "extract_model": config.EXTRACT_MODEL, "profile": profile, "arms": ordered,
            "arms_run_today": arms,
            "fixture": built.digest(), "probe_at": built.probe_at,
            "temperature": config.TEMPERATURE, "reasoning_effort": config.REASONING_EFFORT,
            "max_completion_tokens": config.MAX_COMPLETION_TOKENS["answer"],
            "prompt_version": config.PROMPT_VERSION, "store_version": config.STORE_VERSION,
            "half_life_days": config.HALF_LIFE_DAYS, "fresh_floor": config.FRESH_FLOOR,
            "recall_k": config.RECALL_K,
            "llm_calls": llm.stats(),
            "tokens_this_run": spent_tokens - started_tokens,
            "requests_today": spent_requests,
        },
        "results": rows,
    }
    path = config.results_path(model)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def estimate(arms: list[str], *, model: str = "") -> dict:
    """What this run will cost, per model, before it spends anything.

    Per model matters: answers run on the large model and extraction on the
    small one, and Groq's daily ceiling applies to each separately. Summing them
    understated the real headroom by a third of a run and would have deferred
    work there was budget for.
    """
    built = fixmod.build()
    model = model or config.CHAT_MODEL
    answers = len(arms) * len(built.probes) * 420
    writes = len(built.changes) * 1200 + 2000
    answer_spent, _ = cache.spent_today(model)
    write_spent, _ = cache.spent_today(config.EXTRACT_MODEL)
    return {
        "arms": len(arms), "probes": len(built.probes),
        "answer_model": model, "answer_tokens": answers,
        "answer_spent": answer_spent, "answer_headroom": config.TOKENS_PER_DAY - answer_spent,
        "write_model": config.EXTRACT_MODEL, "write_tokens": writes,
        "write_spent": write_spent, "write_headroom": config.TOKENS_PER_DAY - write_spent,
        "fits": answers <= config.TOKENS_PER_DAY - answer_spent,
    }


if __name__ == "__main__":
    print(json.dumps(estimate(list(PROFILES["lite"])), indent=2))
