"""The agent: triage a morning's support tickets for Kumkum Home in one run.

Inputs:  six hand-written tickets (below) and an LLMRouter.
Outputs: one result dict per ticket - category, priority, the reply the customer
         sees, and the operator-only routing facts. History is shared across the
         run, so ticket 6 only makes sense if ticket 2 survived any failover.
"""

import json
from collections.abc import Callable

from src.router import LLMRouter

SYSTEM_PROMPT = (
    "You triage support tickets for Kumkum Home, an Indian online store for sarees, "
    "brass decor and home textiles. Policy: we ship within India only; returns within 7 days of delivery; damaged "
    "items get a free replacement; refunds reach the bank in 5-7 working days. "
    "For each ticket answer ONLY with JSON: "
    '{"category": "damaged|refund|delivery|product_question|other", '
    '"priority": "low|medium|high", "reply": "one or two sentences to the customer"}. '
    "Earlier tickets in this conversation are context: if a customer writes again, use what they said before."
)

TICKETS = [
    "Meera (order KH-1041): Is the Banarasi silk saree dry-clean only?",
    "Arjun (order KH-1042): My brass diya set arrived with two diyas cracked. Diwali is next week!",
    "Fatima (order KH-1043): Returned the bedsheet 4 days ago, when will I get my money?",
    "Ravi (order KH-1044): Tracking has said 'out for delivery' for three days now.",
    "Sneha (order KH-1045): Do you ship to Singapore?",
    "Arjun again: Any update on what I wrote about earlier? I need them before the festival.",
]


def parse_triage(text: str) -> dict:
    """Read the model's JSON; keep the run alive if a model returns junk."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}
    return {
        "category": str(data.get("category", "other")),
        "priority": str(data.get("priority", "medium")),
        "reply": str(data.get("reply", text.strip()[:200])),
    }


async def run_agent(router: LLMRouter, before_ticket: Callable[[int], None] | None = None) -> list[dict]:
    """Triage every ticket in order through the router, sharing one history."""
    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    results = []
    for step, ticket in enumerate(TICKETS, start=1):
        if before_ticket:
            before_ticket(step)
        print(f"\nTicket {step}: {ticket}")
        history.append({"role": "user", "content": ticket})
        reply = await router.complete(history)
        triage = parse_triage(reply.text)
        if reply.provider == "degraded":
            history.pop()  # no real answer: don't teach later turns a canned one
        else:
            history.append({"role": "assistant", "content": reply.text})
        print(f"  Customer sees: {triage['reply']}")
        results.append({"step": step, **triage, "provider": reply.provider, "attempts": reply.attempts,
                        "failovers": reply.failovers, "latency_s": reply.latency_s,
                        "needs_human": reply.provider == "degraded"})
    return results
