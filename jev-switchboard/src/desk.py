"""The support desk CLI: classify with Jev, route, answer, check the reply.

Inputs:  a command (ask / demo / race) and, for ask, one customer message.
Outputs: the routing trace and the reply, printed to the terminal.
"""

import argparse
import asyncio
import sys

from typesafe_sdk import AsyncTypeSafeClient

from src import config, guard, handlers, jev_router, store
from src.race import run_race

DEMO_MESSAGES = [
    "Where is my order KH-1042?",
    "Is the block-print bedsheet king size? How do I wash it?",
    "I want to return the brass diyas from order KH-1043, one is scratched.",
    "Order KH-1044 saree came late, wrong colour, and I was charged twice.",
    "Please connect me to a real person.",
    "hello??",
]

# A draft a careless LLM might write, used to show the reply check catching it.
BAD_DRAFT = ("So sorry! We'll refund KH-1044 in full and add a 20% coupon for your "
             "next order as an apology.")


async def ask(client: AsyncTypeSafeClient, message: str) -> None:
    """Run one message through the full pipeline and print every step."""
    s = await jev_router.classify(client, message)
    print(f"jev    intent={s.intent} ({s.intent_confidence:.2f})  "
          f"complexity={s.complexity:.2f}  wants_human={s.wants_human:.2f}  [{s.ms:.0f} ms]")
    route = jev_router.decide(s)
    print(f"route  {route.name}: {route.reason}")
    reply, context, ms = await handlers.handle(route.name, message)
    print(f"reply  {reply}  [{ms:.0f} ms]")
    if route.name in ("small_llm", "big_llm"):
        verdict = await guard.check_reply(client, message, reply, context)
        status = "send" if verdict.send else f"HOLD for review {verdict.flags}"
        print(f"guard  {status}  [{verdict.ms:.0f} ms]")


async def demo(client: AsyncTypeSafeClient) -> None:
    """Run every demo message, then show the reply check on a bad draft."""
    for message in DEMO_MESSAGES:
        print(f"\n> {message}")
        await ask(client, message)
    print(f"\n> reply check on a hand-written bad draft:\n  {BAD_DRAFT}")
    facts = f"{store.RETURN_POLICY}\n{handlers.order_facts('KH-1044')}"
    verdict = await guard.check_reply(client, DEMO_MESSAGES[3], BAD_DRAFT, facts)
    print(f"guard  {'send' if verdict.send else f'HOLD for review {verdict.flags}'}")


async def main() -> None:
    """Parse the command and run it with one shared Jev client."""
    parser = argparse.ArgumentParser(description="Jev-routed support desk")
    parser.add_argument("command", choices=["ask", "demo", "race"])
    parser.add_argument("message", nargs="?", default="Where is my order KH-1042?")
    args = parser.parse_args()
    async with AsyncTypeSafeClient(model=config.JEV_MODEL) as client:
        if args.command == "ask":
            await ask(client, args.message)
        elif args.command == "demo":
            await demo(client)
        else:
            await run_race(client, DEMO_MESSAGES)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # LLM replies contain characters like U+2011
    asyncio.run(main())
