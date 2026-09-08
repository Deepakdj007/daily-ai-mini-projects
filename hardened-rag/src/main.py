"""The command line: build, check, ask, run, report.

Inputs:  argv
Outputs: terminal output, and the files under output/

`ask` is the one worth running first. It answers a single question at a chosen
rung under a chosen attack and prints the whole audit trail: what was
retrieved, what each screen made of it, what each passage claimed on its own,
and how the conflict was settled. Every number the leaderboard reports is
visible there for one case, which is the only way to believe the aggregate.
"""

from __future__ import annotations

import argparse
import asyncio

from rich.console import Console
from rich.table import Table

from src import config, gates, index, pipeline, report, screen
from src.adversary import build as build_attacks
from src.corpus import load, tier_counts
from src.ladder import ALL_ARMS, BY_NAME, PROFILES

console = Console()

CONDITIONS = ("clean", "absent", "poison-p", "poison-v", "inject-overt",
              "inject-policy", "stale", "saturate", "sametier", "embedded")


def cmd_build(args) -> None:
    """Embed the corpus and every attack into one index."""
    corpus = load()
    attacks = build_attacks(corpus)
    conn = index.connect()
    total = index.build(conn, list(corpus.passages) + attacks)
    console.print(f"indexed [bold]{total}[/] passages: {len(corpus.passages)} clean "
                  f"({tier_counts(corpus.passages)}), {len(attacks)} adversarial")
    console.print(f"corpus digest {corpus.digest}")
    conn.close()


def cmd_gates(args) -> None:
    """Run every check that costs nothing, before spending anything."""
    corpus = load()
    attacks = build_attacks(corpus)
    conn = index.connect()
    failures = gates.run_all(conn, corpus, attacks)
    conn.close()
    if failures:
        console.print(f"[red]{len(failures)} gate failures[/]")
        for failure in failures:
            console.print(f"  {failure}")
        raise SystemExit(1)
    console.print("[green]all build gates pass[/]")


def cmd_ladder(args) -> None:
    """Print the ladder and re-check the single-switch rule."""
    from src import ladder

    ladder.assert_single_switch()
    table = Table(title="the ladder")
    table.add_column("arm")
    table.add_column("what it adds")
    table.add_column("prediction, written before the run")
    for arm in ALL_ARMS:
        table.add_row(arm.name, arm.label, arm.prediction,
                      style="cyan" if arm.name in ladder.HEADLINE else "")
    console.print(table)
    console.print(f"headline: {ladder.HEADLINE[0]} vs {ladder.HEADLINE[1]}")


async def _ask(args) -> None:
    """One question, one arm, one condition, with the whole trail."""
    corpus = load()
    attacks = build_attacks(corpus)
    lookup = {passage.pid: passage for passage in attacks}
    conn = index.connect()

    question = next((q for q in corpus.all_questions
                     if q.qid == args.question or args.question.lower() in q.question.lower()),
                    None)
    if question is None:
        raise SystemExit(f"no question matching {args.question!r}. Try q01, or a phrase.")

    arm = BY_NAME[args.arm]
    relevant = [corpus.by_pid[question.gold_pid]] + [
        p for p in attacks if p.target_qid == question.qid]
    scores = await screen.guard_scores(list(corpus.passages) + relevant)
    out = await pipeline.answer(conn, corpus, lookup, question, arm.policy,
                                cond=args.cond, dose=args.dose, arm=arm.name,
                                guard_scores=scores)

    console.print(f"\n[bold]{question.question}[/]")
    console.print(f"arm [cyan]{arm.name}[/] | condition [cyan]{args.cond}[/] | "
                  f"truth [green]{question.gold}[/] | attacker wants "
                  f"[red]{question.target}[/]\n")

    table = Table(title=f"the {config.TOP_K} passages that reached the pipeline")
    table.add_column("#", justify="right")
    table.add_column("source")
    table.add_column("tier")
    table.add_column("date")
    table.add_column("guard", justify="right")
    table.add_column("echo", justify="right")
    table.add_column("kept")
    table.add_column("claimed on its own")

    claims = {claim.pid: claim for claim in out.claims}
    shown = [p for p in out.kept] + [
        corpus.by_pid.get(pid) or lookup[pid] for pid in out.dropped]
    for number, passage in enumerate(shown, start=1):
        firing = out.firings.get(passage.pid)
        dropped = out.dropped.get(passage.pid, "")
        claim = claims.get(passage.pid)
        claimed = "-" if claim is None else (claim.value or "nothing")
        if claim is not None and claim.value and not claim.verified:
            claimed += " [red](not in the passage)[/]"
        style = "red" if passage.adversarial else ""
        table.add_row(
            str(number),
            ("ATTACK " if passage.adversarial else "") + passage.doc_id[:26],
            config.TIER_NAMES[passage.tier], passage.date,
            f"{firing.guard:.3f}" if firing else "-",
            f"{firing.echo:.2f}" if firing else "-",
            f"[red]dropped: {dropped}[/]" if dropped else "yes",
            claimed, style=style)
    console.print(table)

    colour = "green" if out.verdict.answer and question.gold in out.verdict.answer else "yellow"
    console.print(f"\nstatus [bold]{out.verdict.status}[/] "
                  f"answer [bold {colour}]{out.verdict.answer or '-'}[/]")
    console.print(f"because: {out.verdict.reason}")
    if out.verdict.conflicts:
        console.print(f"values on the table: {', '.join(out.verdict.conflicts)}")
    conn.close()


def cmd_run(args) -> None:
    """Run the ladder and write the results file."""
    from src import harness

    payload = asyncio.run(harness.run(args.profile, args.model, resume=not args.restart))
    console.print(f"{len(payload['results'])} rows in "
                  f"{config.results_path(args.model or config.CHAT_MODEL)}")
    if payload["manifest"].get("stopped"):
        console.print(f"[yellow]{payload['manifest']['stopped']}[/]")


def cmd_report(args) -> None:
    """Gates, tables and the verdict, from the committed results file."""
    report.render(args.model)
    console.print(f"\nwrote {config.LEADERBOARD_PATH}")


def cmd_budget(args) -> None:
    """What today's runs have already spent, per model."""
    from src import cache

    table = Table(title="today's spend, per model - each model has its own ceiling")
    table.add_column("model")
    table.add_column("tokens", justify="right")
    table.add_column("requests", justify="right")
    table.add_column("headroom", justify="right")
    for row in cache.ledger_rows():
        model, tokens = row["model"], row["tokens"]
        limit = (config.GUARD_TOKENS_PER_DAY if model == config.GUARD_MODEL
                 else config.TOKENS_PER_DAY)
        table.add_row(model, f"{tokens:,}", f"{row['requests']:,}",
                      f"{max(0, limit - tokens):,}")
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hardened-rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="embed the corpus and the attacks").set_defaults(fn=cmd_build)
    sub.add_parser("gates", help="run the free validity checks").set_defaults(fn=cmd_gates)
    sub.add_parser("ladder", help="print the ladder").set_defaults(fn=cmd_ladder)
    sub.add_parser("budget", help="today's spend per model").set_defaults(fn=cmd_budget)

    ask = sub.add_parser("ask", help="one question, with the full audit trail")
    ask.add_argument("question", help="a qid like q01, or any phrase from the question")
    ask.add_argument("--arm", default="provenance", choices=[a.name for a in ALL_ARMS])
    ask.add_argument("--cond", default="poison-p", choices=CONDITIONS)
    ask.add_argument("--dose", type=int, default=3, help="poison copies, where the condition uses them")
    ask.set_defaults(fn=lambda args: asyncio.run(_ask(args)))

    run = sub.add_parser("run", help="run the ladder")
    run.add_argument("--profile", default="full", choices=sorted(PROFILES))
    run.add_argument("--model", default="", help="defaults to CHAT_MODEL")
    run.add_argument("--restart", action="store_true", help="discard previous rows")
    run.set_defaults(fn=cmd_run)

    rep = sub.add_parser("report", help="gates, tables, verdict, leaderboard")
    rep.add_argument("--model", default="", help="defaults to CHAT_MODEL")
    rep.set_defaults(fn=cmd_report)
    return parser


if __name__ == "__main__":
    _args = build_parser().parse_args()
    _args.fn(_args)
