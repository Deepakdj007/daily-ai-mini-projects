"""The command line: parse arguments, run a command, explain what went wrong.

    PYTHONPATH=. uv run python -m src.main run "<brief>" [--ask MODE] [--park]
    PYTHONPATH=. uv run python -m src.main inbox
    PYTHONPATH=. uv run python -m src.main answer <thread_id>
    PYTHONPATH=. uv run python -m src.main compare [--all]
    PYTHONPATH=. uv run python -m src.main ask-rate [--samples N] [--all]

What each command does lives in src/commands.py. This file is the shell around
it, so the two failures a reader will actually hit — no API key, and a spent
free tier — are handled in one obvious place.

Inputs:  command-line arguments
Outputs: a process exit code
"""

import argparse
import sys

from groq import RateLimitError

from src import config
from src.commands import answer, inbox, run

# Windows consoles default to cp1252, which cannot encode the arrows and dashes
# a model happily returns. Force UTF-8 before anything prints.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    """Parse the command line and run it. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="A research team that asks you only when it has to."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Research a brief.")
    p_run.add_argument("brief", help="What you want researched.")
    p_run.add_argument("--ask", choices=config.ASK_POLICIES,
                       default=config.DEFAULT_POLICY, help="Who may pull the cord.")
    p_run.add_argument("--park", action="store_true",
                       help="Freeze at the first question and exit.")

    sub.add_parser("inbox", help="List runs waiting on an answer.")

    p_answer = sub.add_parser("answer", help="Answer a parked run and finish it.")
    p_answer.add_argument("thread_id", help="The id printed when the run parked.")

    p_compare = sub.add_parser("compare", help="Same prompts, cord on and off.")
    p_compare.add_argument("--all", action="store_true",
                           help="Use all six briefs instead of four.")

    p_rate = sub.add_parser("ask-rate", help="How often the lead pulls the cord.")
    p_rate.add_argument("--samples", type=int, default=5,
                        help="Times to ask the lead each brief.")
    p_rate.add_argument("--all", action="store_true",
                        help="Use all six briefs instead of four.")

    args = parser.parse_args()
    try:
        _dispatch(args)
    except RuntimeError as exc:
        # A missing key is a setup mistake, not a crash. Print the sentence that
        # fixes it, not sixty lines of stack the reader cannot act on.
        print(f"\n{exc}")
        return 1
    except RateLimitError as exc:
        # The free tier allows 200,000 tokens a day. max_retries cannot rescue
        # this one: the server asks for minutes, and the SDK only waits out a
        # retry-after of 60 seconds or less. So say what happened in a sentence.
        print(f"\nGroq rate limit reached.\n  {_limit_detail(exc)}")
        print("\nThe free tier allows 8,000 tokens per minute and 200,000 per "
              "day, shared across every key in your organisation.")
        print("A per-minute limit clears on its own; a daily one needs tomorrow.")
        return 1
    return 0


def _limit_detail(exc: RateLimitError) -> str:
    """Pull the readable sentence out of a Groq rate-limit error."""
    try:
        return exc.body["error"]["message"]  # type: ignore[index]
    except Exception:  # noqa: BLE001 - the message shape is not a contract
        return str(exc)


def _dispatch(args) -> None:
    """Route a parsed command to its handler."""
    if args.command == "run":
        run(args.brief, args.ask, args.park)
    elif args.command == "inbox":
        inbox()
    elif args.command == "answer":
        answer(args.thread_id)
    elif args.command == "compare":
        from src.compare import run_comparison

        run_comparison(use_all=args.all)
    elif args.command == "ask-rate":
        from src.askrate import ask_rate

        ask_rate(samples=args.samples, use_all=args.all)


if __name__ == "__main__":
    sys.exit(main())
