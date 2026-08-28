"""Command line for the whole project.

Inputs:  a subcommand
Outputs: whatever that subcommand produces

Imports live inside each handler so `--help` is instant and a command that
needs no API key never asks for one. Two of these cost nothing at all:
`report` and `plot` rebuild every table and figure from the committed results
file, which is what a reader with no daily tokens left should run first.
"""

from __future__ import annotations

import argparse
import asyncio


def cmd_cache_probe(args: argparse.Namespace) -> None:
    """Measure whether cached tokens are rate-limit exempt on this account."""
    from src import preflight

    asyncio.run(preflight.run(args.prefix))


def cmd_scenario(_: argparse.Namespace) -> None:
    """Build the transcript, derive the zones, run the build gates."""
    from src import scenario

    scenario.main()


def cmd_summary(args: argparse.Namespace) -> None:
    """Generate and gate the frozen running summary."""
    import runpy

    if args.force:
        from src import config

        config.SUMMARY_FIXTURE_PATH.unlink(missing_ok=True)
    runpy.run_module("src.summarizer", run_name="__main__")


def cmd_run(args: argparse.Namespace) -> None:
    """Run the ablation."""
    from src import harness

    asyncio.run(harness.run(
        args.profile, args.model, only=args.arms, limit=args.limit, yes=args.yes
    ))


def cmd_report(args: argparse.Namespace) -> None:
    """Rebuild every table from the committed results. Zero API cost."""
    from src import report

    report.main(args.model)


def cmd_plot(args: argparse.Namespace) -> None:
    """Rebuild both figures. Zero API cost."""
    from src import plot

    plot.main(args.model)


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask one question at turn N, layer off then on."""
    from src import chat

    asyncio.run(chat.ask(args.question, args.turn))


def cmd_chat(args: argparse.Namespace) -> None:
    """The live demo: a long session with the layer off, then on."""
    from src import chat

    asyncio.run(chat.demo(args.turns))


def build_parser() -> argparse.ArgumentParser:
    """Wire up the subcommands."""
    parser = argparse.ArgumentParser(prog="context-engineering-layer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("cache-probe", help="does the cache exempt you from the rate limit?")
    p.add_argument("--prefix", type=int, default=0, help="prefix size in tokens")
    p.set_defaults(func=cmd_cache_probe)

    p = sub.add_parser("scenario", help="transcript, zones and build gates (free)")
    p.set_defaults(func=cmd_scenario)

    p = sub.add_parser("summary", help="generate and gate the frozen summary")
    p.add_argument("--force", action="store_true", help="regenerate from scratch")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("run", help="run the ablation")
    p.add_argument("--profile", default="lite", choices=("smoke", "lite", "full"))
    p.add_argument("--model", default="")
    p.add_argument("--arms", nargs="*", default=(), help="override the profile")
    p.add_argument("--limit", type=int, default=0, help="first N probes only")
    p.add_argument("--yes", action="store_true", help="spend tokens")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="tables from the committed results (free)")
    p.add_argument("--model", default="")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("plot", help="figures from the committed results (free)")
    p.add_argument("--model", default="")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("ask", help="one question, layer off then on")
    p.add_argument("question")
    p.add_argument("--turn", type=int, default=0, help="how far into the session")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="the live break demo")
    p.add_argument("--turns", type=int, default=12)
    p.set_defaults(func=cmd_chat)

    return parser


def main() -> None:
    """Parse and dispatch."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
