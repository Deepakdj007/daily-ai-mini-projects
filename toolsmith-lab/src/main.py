"""The command line: look at the corpus, solve one task, run the ladder, report.

    PYTHONPATH=. uv run python -m src.main --help

Inputs:  command-line arguments
Outputs: tables, one task's trace, results files, the leaderboard

Imports live inside the handlers, so --help is instant and the commands that
cost nothing never ask for an API key.
"""

from __future__ import annotations

import argparse
import asyncio

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

console = Console()


def cmd_tasks(args: argparse.Namespace) -> int:
    """Show the corpus: six families, four tasks each, and the hidden inputs."""
    from src import corpus

    for entry in corpus.families():
        console.print(f"\n[bold]{entry['family']}[/] - {entry['label']}  "
                      f"[dim]({entry['compare']}, {len(entry['holdout'])} held-out "
                      f"inputs nobody sees)[/]")
        console.print(f"  [dim]{entry['why_tool']}[/]")
        table = Table(show_header=True)
        for column in ("task", "inputs", "answer"):
            table.add_column(column, overflow="fold")
        for task in entry["tasks"]:
            table.add_row(task["id"], str(task["inputs"])[:60], str(task["answer"])[:60])
        console.print(table)
    console.print(f"\ncorpus {corpus.digest()}\noracle {corpus.stamp()['oracle']}")
    return 0


def cmd_ladder(args: argparse.Namespace) -> int:
    """Show the arms, what each switches on, and what it was predicted to do."""
    from src import ladder

    table = Table(title="the ladder - one switch per rung")
    for column in ("arm", "changed", "what it adds"):
        table.add_column(column, overflow="fold")
    for arm in ladder.ALL_ARMS:
        table.add_row(arm.name, arm.changed or "-", arm.label)
    console.print(table)
    console.print(f"[bold]headline[/] {ladder.HEADLINE[0]} against {ladder.HEADLINE[1]}"
                  f"    [bold]controls[/] {', '.join(ladder.CONTROLS)}\n")
    for arm in ladder.ALL_ARMS:
        console.print(f"[bold]{arm.name}[/]  {arm.prediction}\n")
    return 0


def cmd_gates(args: argparse.Namespace) -> int:
    """Every check that costs no tokens, over the corpus."""
    from src import gates

    failed = 0
    table = Table(title="validity gates - before a token is spent")
    table.add_column("gate")
    table.add_column("")
    table.add_column("detail", overflow="fold")
    for name, problems in gates.run_all():
        failed += len(problems)
        table.add_row(name, "[green]pass[/]" if not problems else "[red]FAIL[/]",
                      "\n".join(problems[:3]) or "-")
    console.print(table)
    console.print(f"[bold]{'all gates pass' if not failed else f'{failed} problem(s)'}[/]")
    return 1 if failed else 0


def cmd_redteam(args: argparse.Namespace) -> int:
    """Run the attack corpus at both layers."""
    from src import redteam

    return redteam.main()


def cmd_tools(args: argparse.Namespace) -> int:
    """What the agent can call right now, fixed and written."""
    from src import registry

    reg = registry.Registry(persist=True, arm="inspect")
    reg.load()
    table = Table(title="the toolbox")
    for column in ("tool", "origin", "born on", "tests", "held-out", "description"):
        table.add_column(column, overflow="fold")
    for schema in reg.schemas():
        name = schema["function"]["name"]
        tool = reg.get(name)
        table.add_row(name, "written" if tool else "fixed",
                      tool.task_id if tool else "-",
                      str(tool.tests_passed) if tool else "-",
                      f"{tool.holdout_passed}/{tool.holdout_total}" if tool else "-",
                      schema["function"]["description"][:70])
    console.print(table)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print a written tool's source. It is a real file on disk."""
    from src import config, registry

    reg = registry.Registry(persist=True, arm="inspect")
    reg.load()
    tool = reg.get(args.name)
    if tool is None:
        console.print(f"[red]no written tool called {args.name!r}[/]. "
                      f"Written so far: {', '.join(t.name for t in reg.generated()) or 'none'}")
        return 1
    console.print(f"[bold]generated/{tool.name}.py[/]  written on task {tool.task_id} "
                  f"by arm {tool.born_arm}, round {tool.born_round}")
    console.print(f"passed {tool.tests_passed} of its own tests and "
                  f"{tool.holdout_passed}/{tool.holdout_total} held-out inputs")
    console.print(f"digest {tool.digest}\n")
    console.print(Syntax(tool.source, "python", theme="ansi_dark", line_numbers=True))
    console.print(f"\n[dim]{config.GENERATED_DIR / (tool.name + '.py')}[/]")
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    """Run one task and show the toolbox growing, round by round."""
    from src import agent, corpus, ladder, registry

    arm = ladder.BY_NAME.get(args.arm)
    if arm is None:
        console.print(f"[red]unknown arm {args.arm!r}[/]; try {', '.join(ladder.BY_NAME)}")
        return 1
    task = corpus.task(args.task)
    family = corpus.family(task["family"])
    if args.fresh:
        registry.clear_generated()
    reg = registry.Registry(persist=arm.policy.persist, arm=arm.name)
    loaded = reg.load()

    console.print(f"[bold]{task['id']}[/] under [bold]{arm.name}[/] "
                  f"({arm.label}), starting with {loaded} written tool(s)")
    console.print(f"[dim]{task['prompt']}[/]\n")
    row = asyncio.run(agent.run_task(task, family, arm=arm.name,
                                     policy=arm.policy, reg=reg))

    seen: set[str] = set()
    for index, names in enumerate(row["offered"]):
        fresh = [n for n in names if n not in seen]
        seen.update(names)
        new = f"   [green]new: {', '.join(fresh)}[/]" if index and fresh else ""
        console.print(f"  round {index}: {', '.join(names)}{new}")
    for note in row["notes"]:
        console.print(f"  [yellow]rejected[/] {note}")
    if row["called"]:
        console.print(f"  called: {', '.join(row['called'])}")

    verdict = "[green]correct[/]" if row["correct"] else f"[red]{row['outcome']}[/]"
    console.print(f"\nanswer [bold]{row['answer'][:90]}[/]   gold [bold]{row['gold']}[/]"
                  f"   {verdict}")
    console.print(f"[dim]{row['completion_tokens']} completion tokens, "
                  f"{row['writes']} tool(s) written, {row['registered']} registered"
                  + (f", reused from {row['reused_from']}" if row["reused"] else "") + "[/]")
    if args.show_source:
        for tool in reg.generated():
            console.print(f"\n[bold]generated/{tool.name}.py[/]")
            console.print(Syntax(tool.source, "python", theme="ansi_dark"))
    return 0 if row["correct"] else 1


def cmd_run(args: argparse.Namespace) -> int:
    """Run the ladder, resumably."""
    from src import harness

    asyncio.run(harness.run(args.profile, args.model, resume=not args.restart,
                            only_arm=args.arm))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Gates, tables, and the pre-registered verdict."""
    from src import report

    report.render(args.model)
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    """What the free tier has left today, by the local ledger."""
    from src import cache, config

    spent, requests = cache.spent_today()
    ceiling = int(config.TOKENS_PER_DAY * config.DAILY_STOP_AT)
    console.print(f"spent today   {spent:,} tokens over {requests} requests")
    console.print(f"stop at       {ceiling:,} ({config.DAILY_STOP_AT:.0%} of "
                  f"{config.TOKENS_PER_DAY:,})")
    console.print(f"left          {max(ceiling - spent, 0):,}")
    console.print(f"cached        {cache.rows():,} responses replay for free")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """Empty the written-tools folder so a demo starts from three tools."""
    from src import registry

    removed = registry.clear_generated()
    console.print(f"removed {removed} written tool(s) from generated/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Every command, and the arguments it takes."""
    parser = argparse.ArgumentParser(
        prog="toolsmith",
        description="An agent that writes its own tools, and the measurement of "
                    "whether keeping them is worth anything.")
    parser.add_argument("--model", default="", help="override the chat model")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tasks", help="the corpus").set_defaults(fn=cmd_tasks)
    sub.add_parser("ladder", help="the arms and their predictions").set_defaults(fn=cmd_ladder)
    sub.add_parser("gates", help="every check that costs nothing").set_defaults(fn=cmd_gates)
    sub.add_parser("redteam", help="the attack corpus, at both layers").set_defaults(fn=cmd_redteam)
    sub.add_parser("tools", help="what the agent can call right now").set_defaults(fn=cmd_tools)
    sub.add_parser("budget", help="what the free tier has left today").set_defaults(fn=cmd_budget)
    sub.add_parser("clean", help="empty generated/").set_defaults(fn=cmd_clean)

    inspect = sub.add_parser("inspect", help="print a written tool's source")
    inspect.add_argument("name")
    inspect.set_defaults(fn=cmd_inspect)

    solve = sub.add_parser("solve", help="run one task and watch the toolbox grow")
    solve.add_argument("task")
    solve.add_argument("--arm", default="persist")
    solve.add_argument("--show-source", action="store_true")
    solve.add_argument("--fresh", action="store_true", help="empty generated/ first")
    solve.set_defaults(fn=cmd_solve)

    run = sub.add_parser("run", help="run the ladder")
    run.add_argument("--profile", default="lite", choices=("smoke", "lite", "full"))
    run.add_argument("--arm", default="", help="run only this arm")
    run.add_argument("--restart", action="store_true", help="ignore existing results")
    run.set_defaults(fn=cmd_run)

    sub.add_parser("report", help="the leaderboard and the verdict").set_defaults(fn=cmd_report)
    return parser


def main() -> int:
    """Parse, dispatch, and turn the two expected failures into one sentence."""
    args = build_parser().parse_args()
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]stopped[/]. Results written so far are kept; "
                      "the same command resumes.")
        return 130
    except SystemExit as exc:
        if exc.code not in (0, None):
            console.print(f"[red]{exc}[/]" if isinstance(exc.code, str) else "")
        return int(exc.code) if isinstance(exc.code, int) else 1


if __name__ == "__main__":
    raise SystemExit(main())
