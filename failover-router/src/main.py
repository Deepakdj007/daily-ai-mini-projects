"""Command line for the failover router demo.

Usage (from the project root):
  PYTHONPATH=. uv run python -m src.main selftest
  PYTHONPATH=. uv run python -m src.main run [--kill gemini@3-4] [--flaky groq] [--bad-key gemini]
  PYTHONPATH=. uv run python -m src.main litellm [--kill gemini@3-4]
Outputs: the agent run, then an operator summary of who answered each ticket.
"""

import argparse
import asyncio
import sys

from src import config
from src.chaos import ChaosPlan
from src.providers import build_providers
from src.router import LLMRouter
from src.tickets import run_agent


def print_summary(results: list[dict]) -> None:
    """Operator view: the routing story the customer never saw."""
    print("\n--- operator summary ---")
    print(f"{'ticket':<7}{'provider':<10}{'attempts':<10}{'failovers':<11}{'seconds':<9}needs_human")
    for r in results:
        print(f"{r['step']:<7}{r['provider']:<10}{r['attempts']:<10}{r['failovers']:<11}"
              f"{r['latency_s']:<9.2f}{r['needs_human']}")
    answered = sum(not r["needs_human"] for r in results)
    print(f"\n{answered}/{len(results)} tickets answered by a model, "
          f"{len(results) - answered} handed to a human, 0 crashes")


async def run_custom(args: argparse.Namespace) -> None:
    """The hand-built router: our adapters, our retries, our breakers."""
    keys = {"gemini": config.GEMINI_API_KEY, "groq": config.GROQ_API_KEY}
    if args.bad_key:
        keys[args.bad_key] = "not-a-real-key"  # a real auth failure, not a simulated one
    providers = build_providers(keys["gemini"], keys["groq"])
    chaos = ChaosPlan(kill=args.kill, flaky=args.flaky)
    chaos.attach(providers)
    router = LLMRouter(providers, cooldown_s=args.cooldown)
    print_summary(await run_agent(router, before_ticket=chaos.begin_ticket))


def parse_args() -> argparse.Namespace:
    """Three commands: selftest, run (custom router), litellm (library router)."""
    parser = argparse.ArgumentParser(description="Multi-provider LLM router with automatic failover")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="free checks, no API calls")
    for name in ("run", "litellm"):
        p = sub.add_parser(name)
        p.add_argument("--kill", default="", help="e.g. gemini@3-4 or gemini,groq")
        p.add_argument("--flaky", default="", help="providers that fail once with a 503")
        p.add_argument("--bad-key", choices=["gemini", "groq"], help="use a wrong API key")
        p.add_argument("--cooldown", type=float, default=config.BREAKER_COOLDOWN_S)
    return parser.parse_args()


def main() -> None:
    """Dispatch to the chosen command."""
    sys.stdout.reconfigure(encoding="utf-8")  # models emit characters a Windows console can't print
    args = parse_args()
    if args.command == "selftest":
        from src.selftest import run_selftest
        sys.exit(0 if run_selftest() else 1)
    if args.command == "litellm":
        from src.litellm_router import run_litellm
        asyncio.run(run_litellm(args))
    else:
        asyncio.run(run_custom(args))


if __name__ == "__main__":
    main()
