"""Command line for the semantic cache demo.

Commands:
  ask "question"    answer one question through the persistent cache
  replay            run all traffic without, then with, the cache; print totals + chart
  sweep             hits vs false hits at each threshold (free, no LLM calls)
  invalidate-demo   show a stale answer, then three ways to get rid of it
  clear             empty the cache
"""

import argparse
import asyncio

from src.agent import Record, ask
from src.cache import SemanticCache
from src.config import ROOT, THRESHOLD
from src.report import print_totals, save_chart
from src.sweep import sweep
from src.traffic import POLICY, TRAFFIC

POLICY_V1 = POLICY.format(version=1, return_days=7)


def show(n: int, record: Record) -> None:
    """Print one line per question: what happened, how long it took."""
    if record.hit:
        how = f'HIT  {record.similarity:.2f} ~ "{record.matched}"'
    else:
        how = "MISS -> LLM"
    print(f"{n:>3}  {record.latency_ms:>7.0f} ms  {how:<58}  {record.question}")


async def run_traffic(cache: SemanticCache | None) -> list[Record]:
    """Send every question in TRAFFIC through the agent, in order."""
    records = []
    for n, (question, _) in enumerate(TRAFFIC, start=1):
        record = await ask(question, POLICY_V1, 1, cache)
        show(n, record)
        records.append(record)
    return records


async def cmd_replay() -> None:
    """The headline: same traffic, cache off vs cache on (starting empty)."""
    print("=== Pass 1: no cache - every question calls the LLM ===")
    plain = await run_traffic(None)
    print(f"\n=== Pass 2: semantic cache, threshold {THRESHOLD} ===")
    cache = SemanticCache()
    cache.clear()
    cached = await run_traffic(cache)
    saved = print_totals(plain, cached)
    path = ROOT / "cache_report.png"
    save_chart(plain, cached, saved, path)
    print(f"Chart saved to {path.name}")


def cmd_sweep() -> None:
    """Show what each threshold buys and what it risks."""
    thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]
    points, (top_sim, q1, q2) = sweep(thresholds)
    repeats = len(TRAFFIC) - len({intent for _, intent in TRAFFIC})
    print(f"{len(TRAFFIC)} questions, {repeats} repeat an earlier intent "
          f"(best possible: {repeats / len(TRAFFIC):.0%} served from cache)\n")
    print(f"{'threshold':>9}  {'hits':>9}  {'false hits':>10}")
    for p in points:
        print(f"{p.threshold:>9.2f}  {p.hits:>3} ({p.hits / len(TRAFFIC):>3.0%})  {p.false_hits:>10}")
    lowest = points[0]
    print(f"\nAt {lowest.threshold}: {lowest.example}")
    print(f'Closest pair that asks different things: {top_sim:.3f}\n  "{q1}"\n  "{q2}"')


async def ask_and_show(label: str, cache: SemanticCache, question: str,
                       policy: str, version: int) -> None:
    """Ask one question and print where the answer came from."""
    record = await ask(question, policy, version, cache)
    source = f"HIT {record.similarity:.2f}" if record.hit else "MISS -> LLM"
    print(f"[{label}] {source:<12} {question}\n    {record.answer}")


async def cmd_invalidate_demo() -> None:
    """A policy changes under the cache: stale hit, targeted delete, version bump, TTL."""
    cache = SemanticCache()
    cache.clear()
    returns_q, delivery_q = "What is the return window?", "How long does delivery take?"

    print("1. Fill the cache (return window is 7 days)")
    await ask_and_show("v1", cache, returns_q, POLICY_V1, 1)
    await ask_and_show("v1", cache, delivery_q, POLICY_V1, 1)

    print("\n2. The policy changes to 10 days. Nobody tells the cache:")
    policy_10 = POLICY.format(version=2, return_days=10)
    await ask_and_show("STALE", cache, returns_q, policy_10, 1)

    print("\n3. Targeted: delete only entries about returns")
    for question in cache.invalidate_similar("return window, returning an item"):
        print(f"    deleted: {question}")
    await ask_and_show("fixed", cache, returns_q, policy_10, 1)
    await ask_and_show("kept", cache, delivery_q, policy_10, 1)

    print("\n4. Coarse: bump the policy version - every old entry stops matching")
    await ask_and_show("v2", cache, delivery_q, policy_10, 2)

    print("\n5. TTL: an entry older than ttl_s is evicted on lookup")
    short = SemanticCache(ttl_s=1)
    await asyncio.sleep(1.5)
    await ask_and_show("ttl", short, delivery_q, policy_10, 2)


def main() -> None:
    """Parse the command and run it."""
    parser = argparse.ArgumentParser(description="Semantic cache demo")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ask").add_argument("question")
    for name in ("replay", "sweep", "invalidate-demo", "clear"):
        sub.add_parser(name)
    args = parser.parse_args()

    if args.command == "ask":
        asyncio.run(ask_and_show("v1", SemanticCache(), args.question, POLICY_V1, 1))
    elif args.command == "replay":
        asyncio.run(cmd_replay())
    elif args.command == "sweep":
        cmd_sweep()
    elif args.command == "invalidate-demo":
        asyncio.run(cmd_invalidate_demo())
    else:
        SemanticCache().clear()
        print("Cache cleared.")


if __name__ == "__main__":
    main()
