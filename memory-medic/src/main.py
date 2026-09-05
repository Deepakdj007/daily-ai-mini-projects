"""The command line: talk to the agent, sweep it, and inspect what it believes.

Run:
    PYTHONPATH=. uv run python -m src.main seed
    PYTHONPATH=. uv run python -m src.main say "I moved to Bengaluru" --at 2026-06-10
    PYTHONPATH=. uv run python -m src.main sweep --once --as-of 2026-12-01
    PYTHONPATH=. uv run python -m src.main timeline demo city

Imports live inside each handler so --help is instant and the commands that
never touch a model - timeline, status, check - never ask for an API key.
"""

from __future__ import annotations

import argparse
import asyncio

from rich.console import Console
from rich.table import Table

from src import config

console = Console()


def _runtime(args, *, need_graph: bool = False):
    """Open the store, the clock and (when a gate is wanted) the graph.

    The subcommand's own date flag has its own dest and is merged here. A
    subparser flag sharing `as_of` with the global one silently overwrote it
    with its empty default, so `say --as-of 2026-06-10` recorded the real date
    - an injected clock quietly falling back to wall time, which is exactly the
    bug this project cannot afford.
    """
    from src import clock as clockmod, store

    clock = clockmod.make_clock(
        getattr(args, "at", "") or getattr(args, "as_of", "") or config.CLOCK_AT
    )
    conn = store.connect()
    graph = None
    if need_graph and not getattr(args, "no_gate", False):
        from src.graph import build_graph, open_checkpointer

        graph = build_graph(open_checkpointer())
    return conn, clock, graph


def _policy(args):
    from src.policy import POLICIES

    return POLICIES[getattr(args, "policy", "") or config.WRITE_POLICY]


def _clear_checkpoints() -> None:
    """Throw away parked threads whenever the store they point at is wiped.

    Thread ids are derived from the repair's content, which is what stops the
    same problem parking twice. But row ids restart at 1 after a wipe, so a
    reseeded store generates the SAME thread id as a decision somebody already
    made - and the finished checkpoint short-circuits the new one, which showed
    up as a gate that silently stopped gating.
    """
    for suffix in ("", "-wal", "-shm"):
        path = config.CHECKPOINT_DB.with_name(config.CHECKPOINT_DB.name + suffix)
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            console.print(f"[yellow]{path.name} is open elsewhere; close the inbox to reset it[/]")


def cmd_seed(args) -> None:
    """Fill an empty store with a plausible year of remembered facts."""
    from src import seed as seedmod, store

    conn, clock, _ = _runtime(args)
    if args.reset:
        store.wipe(conn)
        _clear_checkpoints()
    planted = seedmod.seed(conn, clock=clock)
    console.print(f"[green]seeded {planted} facts[/] for '{seedmod.DEMO_USER}' "
                  f"as of {clock.today()}; sources written to {config.SOURCES_DIR}")


def cmd_say(args) -> None:
    """Send one message: the agent answers it and learns from it."""
    from src import turn as turnmod

    conn, clock, graph = _runtime(args, need_graph=True)
    console.print(f"[dim]{clock.today()}{' (frozen)' if clock.frozen else ''}[/]")
    result = asyncio.run(turnmod.turn(
        conn, args.user, args.message, clock=clock, write_policy=_policy(args),
        gate_enabled=not args.no_gate, graph=graph, do_answer=not args.quiet,
    ))
    if result.answer:
        console.print(f"[bold cyan]{result.answer.strip()}[/]\n")
    for hit in result.hits:
        console.print(f"  [dim]recalled {hit.topic}/{hit.scope or '-'} "
                      f"({hit.flag}, sim {hit.similarity:.2f})[/]")
    for plan, outcome in zip(result.plans, result.outcomes):
        routed = outcome if isinstance(outcome, str) else outcome.routed
        style = {"parked": "yellow", "forced": "red"}.get(routed, "green")
        console.print(f"  [{style}]{plan.op.upper():10}[/] {plan.topic}/{plan.scope or '-'} "
                      f"= {plan.fact.value if plan.fact else '-'}  ({routed}, "
                      f"{plan.confidence:.2f}) {plan.reason}")
    if result.error:
        console.print(f"[red]{result.error}[/]")


def cmd_recall(args) -> None:
    """Ask what the agent remembers, optionally as of a past date."""
    from src import recall as recallmod

    conn, clock, _ = _runtime(args)
    hits = recallmod.recall(conn, args.user, args.query, clock=clock, k=args.k)
    console.print(recallmod.render(hits))


def cmd_sweep(args) -> None:
    """Run the hygiene sweep once, or on a timer."""
    from src import sweep as sweepmod

    conn, clock, graph = _runtime(args, need_graph=True)
    kwargs = dict(policy=_policy(args), gate_enabled=not args.no_gate, graph=graph)
    if args.watch:
        console.print(f"[bold]sweeping every {args.interval:g}s[/] - edit a file in "
                      f"{config.SOURCES_DIR} to see it react. Ctrl-C to stop.\n")
        asyncio.run(sweepmod.run_watch(conn, clock=clock, interval=args.interval,
                                       console=console, **kwargs))
        return
    report = asyncio.run(sweepmod.tick(conn, clock=clock, dry_run=args.dry_run, **kwargs))
    console.print(f"[bold]candidates[/] {report.candidates}")
    for action in report.actions:
        console.print(f"  {action}")
    if args.dry_run:
        console.print("[dim]dry run: nothing applied, no model called[/]")
    else:
        console.print(f"[green]{report.applied} applied[/], [yellow]{report.parked} parked[/], "
                      f"{report.forced} forced, {report.live_calls} model calls "
                      f"({report.replayed_calls} replayed)")


def cmd_timeline(args) -> None:
    """Every version of one memory, with the window each was true for."""
    from src import clock as clockmod, store

    conn, clock, _ = _runtime(args)
    rows = store.history(conn, args.user, args.topic, args.scope)
    if not rows:
        console.print(f"[dim]nothing on file for {args.topic}/{args.scope or '-'}[/]")
        return
    title = f"{args.user} - {args.topic}" + (f" [{args.scope}]" if args.scope else "")
    table = Table(title=title, pad_edge=False, padding=(0, 1))
    table.add_column("id", no_wrap=True)
    table.add_column("value", no_wrap=True)
    table.add_column("true from", no_wrap=True)
    table.add_column("true until", no_wrap=True)
    table.add_column("recorded", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("decided by", no_wrap=True)
    for row in rows:
        end = "-" if row["valid_to"] >= config.OPEN_END else clockmod.to_iso(row["valid_to"])
        decided = conn.execute(
            "SELECT decided_by, op FROM repairs WHERE new_id = ? ORDER BY id DESC LIMIT 1",
            (row["id"],)).fetchone()
        table.add_row(str(row["id"]), row["value"], clockmod.to_iso(row["valid_from"]), end,
                      clockmod.to_iso(row["recorded_at"]), row["status"],
                      f"{decided['decided_by']}/{decided['op']}" if decided else "-")
    console.print(table)
    current = store.as_of(conn, args.user, args.topic, args.scope, clock.now())
    console.print(f"as of {clock.today()}: [bold]{current['value'] if current else 'nothing'}[/]")


def cmd_review(args) -> None:
    """Approve or reject the repairs the agent was not confident about."""
    from src.graph import build_graph, load_parked_payload, open_checkpointer, parked_rows, resume_thread

    conn, clock, _ = _runtime(args)
    graph = build_graph(open_checkpointer())
    rows = parked_rows(conn)
    if not rows:
        console.print("[dim]nothing waiting for review[/]")
        return
    for row in rows:
        payload = load_parked_payload(graph, row["thread_id"], conn, clock)
        if payload is None:
            from src import store

            store.settle_repair(conn, row["thread_id"], "applied", decided_by="agent",
                                decided_at=clock.now())
            console.print(f"[dim]{row['thread_id']} was already decided elsewhere[/]")
            continue
        before = (payload.get("before") or {}).get("value", "-")
        after = (payload.get("after") or {}).get("value", "-")
        console.print(f"\n[bold]{payload['op']}[/] {payload['topic']}/{payload['scope'] or '-'} "
                      f"({payload['detector']}, confidence {payload['confidence']:.2f})")
        console.print(f"  stored:   {before}\n  proposed: {after}\n  why: {payload['reason']}")
        if payload.get("evidence"):
            console.print(f"  [dim]{payload['evidence'][:300]}[/]")
        if args.decision:
            outcome = resume_thread(graph, conn, row["thread_id"], args.decision, clock=clock)
            console.print(f"  -> [green]{outcome}[/]")
    if not args.decision:
        console.print("\n[dim]pass --decision approve|reject to settle these[/]")


def cmd_status(args) -> None:
    """What the store holds, and what it is waiting on."""
    from src import cache, store

    conn, clock, _ = _runtime(args)
    console.print(f"clock      {clock.today()}{' (frozen)' if clock.frozen else ''}")
    console.print(f"policy     {_policy(args).name}  gate={not args.no_gate}")
    console.print(f"memories   {store.counts(conn)}")
    problems = store.check_invariants(conn)
    console.print(f"invariants {'[green]clean[/]' if not problems else '[red]' + str(problems) + '[/]'}")
    tokens, requests = cache.spent_today()
    console.print(f"spend today {tokens:,} tokens / {requests} requests; "
                  f"{cache.rows()} cached responses")


def cmd_check(args) -> None:
    """Prove no module reads the wall clock behind the clock module's back."""
    import re
    from pathlib import Path

    allowed = {"clock.py", "cache.py", "main.py"}
    pattern = re.compile(r"\b(datetime\.now|time\.time|date\.today)\s*\(")
    offenders = [
        f"{path.name}:{index}"
        for path in sorted(Path(__file__).parent.glob("*.py")) if path.name not in allowed
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    if offenders:
        console.print(f"[red]wall-clock reads outside clock.py: {offenders}[/]")
        raise SystemExit(1)
    console.print("[green]clean[/] - every module takes its time from the injected clock")


def cmd_run(args) -> None:
    """Run the ablation ladder and write the results file."""
    from src import harness
    from src.ladder import PROFILES

    arms = list(args.arms) if args.arms else list(PROFILES[args.profile])
    plan = harness.estimate(arms, model=args.model)
    console.print(f"[bold]{len(arms)} arms[/] x {plan['probes']} probes "
                  f"(cached answers replay free)")
    console.print(f"  answers  {plan['answer_model']:22} ~{plan['answer_tokens']:>7,} needed; "
                  f"{plan['answer_spent']:,} spent, {plan['answer_headroom']:,} left today")
    console.print(f"  writes   {plan['write_model']:22} ~{plan['write_tokens']:>7,} needed; "
                  f"{plan['write_spent']:,} spent, {plan['write_headroom']:,} left today")
    if not plan["fits"]:
        console.print("[yellow]this will not fit today's ceiling - run fewer arms, or "
                      "re-run the same command tomorrow: finished arms are carried over[/]")
    if not args.yes:
        console.print("[yellow]pass --yes to spend it[/]")
        return
    asyncio.run(harness.run(arms, model=args.model, profile=args.profile))
    from src import report

    report.render(args.model)


def cmd_report(args) -> None:
    """Re-read the committed results and rebuild every table. Costs nothing."""
    from src import report

    report.render(args.model)


def cmd_reset(args) -> None:
    """Start over."""
    from src import store

    conn, _, _ = _runtime(args)
    store.wipe(conn)
    conn.close()
    _clear_checkpoints()
    if args.all:
        try:
            config.CACHE_PATH.unlink(missing_ok=True)
        except PermissionError:
            console.print("[red]the response cache is open in another process[/]")
            raise SystemExit(1)
    console.print("[green]reset[/]" + (" (response cache too)" if args.all else ""))


def build_parser() -> argparse.ArgumentParser:
    """Every subcommand, and the switches that change the agent's behaviour."""
    parser = argparse.ArgumentParser(prog="memory-medic")
    parser.add_argument("--policy", default="", choices=["", "append", "overwrite", "scoped", "bitemporal"],
                        help="how writes resolve against what is already stored")
    parser.add_argument("--no-gate", action="store_true", help="apply unsure repairs without asking")
    parser.add_argument("--as-of", default="", help="freeze the agent's clock at this ISO date")
    parser.add_argument("--user", default="demo")
    sub = parser.add_subparsers(dest="command", required=True)

    seed_cmd = sub.add_parser("seed", help="plant the demo memory")
    seed_cmd.add_argument("--reset", action="store_true", help="wipe first")
    seed_cmd.set_defaults(func=cmd_seed)

    say = sub.add_parser("say", help="send one message")
    say.add_argument("message")
    say.add_argument("--at", dest="at", default="", help="pretend it was said on this date")
    say.add_argument("--quiet", action="store_true", help="ingest only, do not answer")
    say.set_defaults(func=cmd_say)

    ask = sub.add_parser("recall", help="show what would be retrieved")
    ask.add_argument("query")
    ask.add_argument("--k", type=int, default=0)
    ask.set_defaults(func=cmd_recall)

    sweep_cmd = sub.add_parser("sweep", help="look for memories that have gone stale")
    sweep_cmd.add_argument("--once", action="store_true", default=True)
    sweep_cmd.add_argument("--watch", action="store_true")
    sweep_cmd.add_argument("--interval", type=float, default=config.SWEEP_INTERVAL_S)
    sweep_cmd.add_argument("--dry-run", action="store_true", help="list candidates, call nothing")
    sweep_cmd.set_defaults(func=cmd_sweep)

    timeline = sub.add_parser("timeline", help="every version of one memory")
    timeline.add_argument("user")
    timeline.add_argument("topic")
    timeline.add_argument("scope", nargs="?", default="")
    timeline.set_defaults(func=cmd_timeline)

    review = sub.add_parser("review", help="settle parked repairs from the terminal")
    review.add_argument("--decision", default="", choices=["", "approve", "reject"])
    review.set_defaults(func=cmd_review)

    run_cmd = sub.add_parser("run", help="run the ablation ladder")
    run_cmd.add_argument("--profile", default="lite", choices=["smoke", "lite", "full"])
    run_cmd.add_argument("--arms", nargs="*", default=[])
    run_cmd.add_argument("--model", default="")
    run_cmd.add_argument("--yes", action="store_true", help="confirm the token spend")
    run_cmd.set_defaults(func=cmd_run)

    report_cmd = sub.add_parser("report", help="rebuild the leaderboard from results")
    report_cmd.add_argument("--model", default="")
    report_cmd.set_defaults(func=cmd_report)

    sub.add_parser("status", help="what the store holds").set_defaults(func=cmd_status)
    sub.add_parser("check", help="audit the clock discipline").set_defaults(func=cmd_check)

    reset = sub.add_parser("reset", help="empty the store")
    reset.add_argument("--all", action="store_true", help="checkpoints and response cache too")
    reset.set_defaults(func=cmd_reset)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]stopped[/]")
