"""The overnight shift: wake on a clock, scout, park drafts, exit before you wake up.

Inputs:  CLI arguments
Outputs: parked threads in memory.db, an index and wake log in nightscout.db

    PYTHONPATH=. uv run python -m src.night run --demo
    PYTHONPATH=. uv run python -m src.night status
    PYTHONPATH=. uv run python -m src.night reset --all
"""

import argparse
import sqlite3
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta

from src import config, sources, store, triage
from src.graph import build_graph, make_thread_id, open_checkpointer, thread_config
from src.state import Item

# Windows consoles default to cp1252, which cannot encode the box characters and
# arrows in the wake log. Force UTF-8 so a print statement never ends the night.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# --- the clock --------------------------------------------------------------


def wake_schedule(night_start: datetime) -> list[datetime]:
    """Every simulated wake time across the work window.

    A 23:00-07:00 window at 45-minute cadence is 11 wakes: 23:00, 23:45 ... 06:45.
    The window is allowed to cross midnight, which is the normal case.
    """
    start = night_start.replace(
        hour=config.WINDOW_START_HOUR, minute=0, second=0, microsecond=0
    )
    if config.WINDOW_END_HOUR > config.WINDOW_START_HOUR:
        end = start.replace(hour=config.WINDOW_END_HOUR)
    else:
        end = (start + timedelta(days=1)).replace(hour=config.WINDOW_END_HOUR)

    wakes: list[datetime] = []
    moment = start
    while moment < end:
        wakes.append(moment)
        moment += timedelta(minutes=config.WAKE_EVERY_MINUTES)
    return wakes


def real_sleep_for(sim_gap: timedelta, scale: float) -> float:
    """How long to actually sleep for a gap in simulated time.

    scale 1.0 is a real night. The demo scale compresses eight simulated hours into
    about three real minutes, which is the only way a reader can watch a full night
    and check that dedup and the carry-over queue behave.
    """
    return max(sim_gap.total_seconds() / scale, config.MIN_REAL_SLEEP_SECONDS)


def night_id_for(window_start: datetime) -> str:
    """The date the window opened. One id for a night that spans two dates."""
    return window_start.strftime("%Y%m%d")


# --- one item's run ---------------------------------------------------------


def launch_item(
    graph, conn: sqlite3.Connection, item: Item, score: int, reason: str, night_id: str
) -> bool:
    """Run one item up to its park point and index it. True if it parked.

    The index row is written HERE, after invoke() returns — never inside the gate
    node. interrupt() replays its node from the top on resume, so a write placed
    before it would run twice.

    durability="sync" because this process may exit seconds from now and a different
    process has to read the checkpoint. The default is "async", which does not
    promise the write has landed.
    """
    thread_id = make_thread_id(night_id, item.item_id)
    run_config = thread_config(thread_id)
    try:
        graph.invoke(
            {
                "item": asdict(item),
                "score": score,
                "reason": reason,
                "night_id": night_id,
                "revision": 0,
            },
            run_config,
            durability="sync",
        )
    except Exception as exc:  # noqa: BLE001 — one bad item must not end the night
        print(f"    ! {item.item_id}: {type(exc).__name__}: {exc}")
        return False

    snapshot = graph.get_state(run_config)
    if snapshot.interrupts:
        store.record_parked(conn, thread_id, item, night_id, score)
        return True
    return False


# --- one wake ---------------------------------------------------------------


def do_wake(graph, conn: sqlite3.Connection, night_id: str, sim_time: datetime) -> None:
    """Poll, triage, then spend the wake's drafting budget on the best candidates."""
    known = store.known_item_ids(conn)
    polled, fresh = sources.poll_all(known)

    # Store first, then read the triage queue back out of the database. That extra
    # hop is what makes a failed triage call recoverable: the items keep score NULL
    # and the next wake picks them up, instead of being buried as "already seen".
    store.remember_items(conn, fresh)
    pending = store.untriaged(conn, config.MAX_ITEMS_PER_WAKE)
    if pending:
        try:
            store.record_scores(conn, triage.score_items(pending))
        except Exception as exc:  # noqa: BLE001 — retried on the next wake
            detail = str(exc).splitlines()[0][:90]
            print(f"    ! triage failed, retrying next wake — {detail}")

    # Drafting budget: whichever is smaller, this wake's cap or what is left tonight.
    spent = store.deep_passes_tonight(conn, night_id)
    budget = max(0, min(config.MAX_DEEP_PASSES_PER_WAKE,
                        config.MAX_DEEP_PASSES_PER_NIGHT - spent))

    parked = 0
    for item in store.pending_deep(conn, config.SCORE_THRESHOLD, budget):
        score, reason = store.item_score(conn, item.item_id)
        if launch_item(graph, conn, item, score, reason, night_id):
            parked += 1
        # Retire it either way. A failed draft that stayed queued would be retried
        # every wake for the rest of the night.
        store.mark_deep_done(conn, item.item_id)

    queued = store.queue_depth(conn, config.SCORE_THRESHOLD)
    store.log_wake(conn, night_id, sim_time.isoformat(timespec="minutes"),
                   polled, len(fresh), len(pending), parked, queued)

    real = datetime.now().strftime("%H:%M:%S")
    # flush because a real night's output is usually redirected to a log file, and
    # a buffered wake log tells you nothing while the agent is still working.
    print(
        f"  [sim {sim_time:%a %H:%M} | real {real}] "
        f"polled {polled:3d}  fresh {len(fresh):2d}  parked {parked}  queued {queued}",
        flush=True,
    )


# --- commands ---------------------------------------------------------------


def run_night(demo: bool, stub: bool) -> None:
    """Work the whole window, wake by wake, then exit."""
    if stub:
        config.STUB_LLM = True
    config.require_keys()

    scale = config.DEMO_TIME_SCALE if demo else config.TIME_SCALE
    schedule = wake_schedule(datetime.now())
    night_id = night_id_for(schedule[0])

    mode = "demo" if demo else "live"
    print(f"\nnight {night_id} · {len(schedule)} wakes · {mode} clock (x{scale:g})"
          + ("  · stub model, no Groq calls" if config.STUB_LLM else ""))
    print(f"window {config.WINDOW_START_HOUR:02d}:00 to "
          f"{config.WINDOW_END_HOUR:02d}:00 · "
          f"last wake {schedule[-1]:%H:%M}\n")

    # A live run started in the afternoon waits for the window to open. That is the
    # whole point: you start it, close the laptop lid, and go to bed.
    if not demo and datetime.now() < schedule[0]:
        delay = (schedule[0] - datetime.now()).total_seconds()
        print(f"  sleeping {delay / 3600:.1f}h until the window opens…")
        time.sleep(delay)

    conn = store.connect(config.DB_PATH, config.SQLITE_TIMEOUT)
    graph = build_graph(open_checkpointer())

    try:
        for index, sim_time in enumerate(schedule):
            do_wake(graph, conn, night_id, sim_time)
            if index + 1 < len(schedule):
                time.sleep(real_sleep_for(schedule[index + 1] - sim_time, scale))
    except KeyboardInterrupt:
        print("\n  stopped early — everything parked so far is already on disk")

    counts = store.status_counts(conn)
    print(f"\nnight {night_id} complete · "
          f"{counts.get('parked', 0)} parked and waiting for you")
    print("Review them:  PYTHONPATH=. uv run streamlit run src/inbox.py\n")


def cmd_status() -> None:
    """What happened last night, and what is still waiting."""
    conn = store.connect(config.DB_PATH, config.SQLITE_TIMEOUT)
    counts = store.status_counts(conn)
    print("\ndrafts: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                          or "none yet"))
    print(f"carry-over queue: {store.queue_depth(conn, config.SCORE_THRESHOLD)} "
          f"item(s) scored >= {config.SCORE_THRESHOLD} still awaiting a draft")

    parked = store.parked_drafts(conn)
    if parked:
        print(f"\n{len(parked)} parked:")
        for row in parked:
            print(f"  {row['thread_id']}  score {row['score']:2d}  "
                  f"{row['source']:5s}  {row['title'][:56]}")

    wakes = store.recent_wakes(conn, limit=12)
    if wakes:
        print("\nrecent wakes (newest first):")
        for row in wakes:
            print(f"  {row['sim_time']}  polled {row['polled']:3d}  "
                  f"fresh {row['fresh']:2d}  parked {row['parked']}  "
                  f"queued {row['queued']}")
    print()


def cmd_reset(everything: bool) -> None:
    """Delete generated state. WAL leaves siblings behind, so remove those too."""
    targets = [config.DB_PATH, config.READING_LIST]
    if everything:
        targets.append(config.MEM_PATH)

    for path in targets:
        for suffix in ("", "-wal", "-shm"):
            victim = path.with_name(path.name + suffix)
            if not victim.exists():
                continue
            try:
                victim.unlink()
                print(f"  removed {victim.name}")
            except PermissionError:
                # Windows holds a lock while any process has the file open — usually
                # a night run still alive in another terminal, or an orphaned child.
                print(f"  ! {victim.name} is locked. Stop the night process first:")
                print("      Get-CimInstance Win32_Process -Filter \"Name='python.exe'\""
                      " | Select-Object ProcessId, CommandLine")
                return
    print()


def main() -> None:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(prog="src.night", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    runner = sub.add_parser("run", help="work one night")
    runner.add_argument("--demo", action="store_true",
                        help="compress the night into ~3 real minutes")
    runner.add_argument("--stub", action="store_true",
                        help="canned model responses, zero Groq calls")

    sub.add_parser("status", help="what is parked and what happened")

    resetter = sub.add_parser("reset", help="delete generated state")
    resetter.add_argument("--all", action="store_true",
                          help="also drop the LangGraph checkpoints")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status()
    elif args.command == "reset":
        cmd_reset(args.all)
    else:
        run_night(demo=getattr(args, "demo", False), stub=getattr(args, "stub", False))


if __name__ == "__main__":
    main()
