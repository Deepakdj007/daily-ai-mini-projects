"""The command line: see the tasks, watch it write a tool, read what it wrote.

    PYTHONPATH=. uv run python -m src.main --help

Inputs:  command-line arguments
Outputs: tables, one task's trace, and the tools it has written

Imports live inside the handlers, so --help is instant and the commands that
cost nothing never ask for an API key.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from src import config

console = Console()


def load_tasks() -> dict:
    """The demo corpus: two families, two tasks each, plus hidden inputs."""
    if not config.TASKS_PATH.exists():
        raise SystemExit(f"{config.TASKS_PATH} is missing.")
    return json.loads(config.TASKS_PATH.read_text(encoding="utf-8"))


def find_task(task_id: str) -> tuple[dict, dict]:
    """(task, its family) by id."""
    for family in load_tasks()["families"]:
        for task in family["tasks"]:
            if task["id"] == task_id:
                return task, family
    ids = [t["id"] for f in load_tasks()["families"] for t in f["tasks"]]
    raise SystemExit(f"no task called {task_id!r}. Try one of: {', '.join(ids)}")


def cmd_tasks(args: argparse.Namespace) -> int:
    """Show the demo tasks, and check that the hidden inputs really are hidden."""
    families = load_tasks()["families"]
    for family in families:
        console.print(f"\n[bold]{family['family']}[/] - {family['label']}  "
                      f"[dim]({len(family['holdout'])} held-out inputs nobody sees)[/]")
        table = Table(show_header=True)
        table.add_column("task")
        table.add_column("answer", overflow="fold")
        for task in family["tasks"]:
            table.add_row(task["id"], str(task["answer"]))
        console.print(table)

    # A held-out input that appears in a prompt is not held out, and a written
    # tool would be checked against something the model was shown.
    leaked = [f["family"] for f in families for row in f["holdout"]
              for item in row["inputs"]
              if any(str(item) in t["prompt"] for t in f["tasks"])]
    console.print(f"\nheld-out inputs appear in no prompt: "
                  f"{'[green]yes[/]' if not leaked else f'[red]NO - {leaked}[/]'}")
    return 1 if leaked else 0


def cmd_redteam(args: argparse.Namespace) -> int:
    """Run the attack corpus at both layers. Costs no tokens."""
    from src import redteam

    return redteam.main()


def cmd_tools(args: argparse.Namespace) -> int:
    """What the agent can call right now, fixed and written."""
    from src import registry

    reg = registry.Registry()
    reg.load()
    table = Table(title="the toolbox")
    for column in ("tool", "origin", "written on", "tests", "held-out", "description"):
        table.add_column(column, overflow="fold")
    for schema in reg.schemas():
        name = schema["function"]["name"]
        tool = reg.get(name)
        table.add_row(name, "written" if tool else "fixed",
                      tool.task_id if tool else "-",
                      str(tool.tests_passed) if tool else "-",
                      str(tool.holdout_passed) if tool else "-",
                      schema["function"]["description"][:60])
    console.print(table)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print a written tool's source. It is a real file on disk.

    The name is optional, and that is not just convenience: the model chooses
    what to call the tool it writes, so a demo script cannot hard-code one.
    With no name this prints the most recently written tool.
    """
    from src import registry

    reg = registry.Registry()
    reg.load()
    written = reg.written()
    if not args.name:
        if not written:
            console.print("[red]no written tools yet[/]. Run: solve gstin_1")
            return 1
        tool = written[-1]
    else:
        tool = reg.get(args.name)
    if tool is None:
        console.print(f"[red]no written tool called {args.name!r}[/]. Written so far: "
                      f"{', '.join(t.name for t in reg.written()) or 'none'}")
        return 1
    console.print(f"[bold]generated/{tool.name}.py[/]  written on task {tool.task_id}, "
                  f"passed {tool.tests_passed} of its own tests and "
                  f"{tool.holdout_passed} held-out inputs\n")
    console.print(Syntax(tool.source, "python", theme="ansi_dark", line_numbers=True))
    console.print(f"\n[dim]{config.GENERATED_DIR / (tool.name + '.py')}[/]")
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    """Run one task and show the toolbox growing, round by round."""
    from src import agent, registry

    task, family = find_task(args.task)
    if args.fresh:
        registry.clear()
    reg = registry.Registry()
    loaded = reg.load()

    console.print(f"[bold]{task['id']}[/], starting with {loaded} written tool(s)")
    console.print(f"[dim]{task['prompt'][-260:]}[/]\n")
    run = asyncio.run(agent.solve(task, family, reg))

    seen: set[str] = set()
    for index, names in enumerate(run.offered):
        fresh = [n for n in names if n not in seen]
        seen.update(names)
        new = f"   [green]new: {', '.join(fresh)}[/]" if index and fresh else ""
        console.print(f"  round {index}: {', '.join(names)}{new}")
    for note in run.rejections:
        console.print(f"  [yellow]rejected[/] {note}")
    if run.called:
        console.print(f"  called: {', '.join(run.called)}")

    verdict = "[green]correct[/]" if run.correct else f"[red]{run.outcome}[/]"
    console.print(f"\nanswer [bold]{run.answer[:80]}[/]\n"
                  f"gold   [bold]{task['answer'][:80]}[/]   {verdict}")
    console.print(f"[dim]{run.completion_tokens} completion tokens, {run.writes} tool(s) "
                  f"written, {run.registered} registered"
                  + (f", reused the tool from {run.reused}" if run.reused else "") + "[/]")
    if args.show_source:
        for tool in reg.written():
            console.print(f"\n[bold]generated/{tool.name}.py[/]")
            console.print(Syntax(tool.source, "python", theme="ansi_dark"))
    return 0 if run.correct else 1


def cmd_clean(args: argparse.Namespace) -> int:
    """Empty the toolbox so a demo starts from the two fixed tools."""
    from src import registry

    console.print(f"removed {registry.clear()} written tool(s) from generated/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Every command, and the arguments it takes."""
    parser = argparse.ArgumentParser(
        prog="toolsmith", description="An agent that writes its own tools.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tasks", help="the demo tasks").set_defaults(fn=cmd_tasks)
    sub.add_parser("redteam", help="the attack corpus, both layers").set_defaults(fn=cmd_redteam)
    sub.add_parser("tools", help="what the agent can call right now").set_defaults(fn=cmd_tools)
    sub.add_parser("clean", help="empty generated/").set_defaults(fn=cmd_clean)

    inspect = sub.add_parser("inspect", help="print a written tool's source")
    inspect.add_argument("name", nargs="?", default="",
                         help="defaults to the most recently written tool")
    inspect.set_defaults(fn=cmd_inspect)

    solve = sub.add_parser("solve", help="run one task and watch the toolbox grow")
    solve.add_argument("task")
    solve.add_argument("--show-source", action="store_true")
    solve.add_argument("--fresh", action="store_true", help="empty generated/ first")
    solve.set_defaults(fn=cmd_solve)
    return parser


def main() -> int:
    """Parse, dispatch, and turn the expected failures into one sentence."""
    args = build_parser().parse_args()
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]stopped[/].")
        return 130
    except SystemExit as exc:
        if isinstance(exc.code, str):
            console.print(f"[red]{exc.code}[/]")
            return 1
        return int(exc.code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
