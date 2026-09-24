"""The reply check: one Jev call reads the LLM's draft before the customer sees it.

Inputs:  the customer message, the draft reply, and the facts the reply may use.
Outputs: a Verdict (send or hold, which checks fired, ms spent).
"""

import time
from dataclasses import dataclass

from typesafe_sdk import AsyncTypeSafeClient, Noul, NoulCriteria

from src import config

CHECKS = {
    "over_promises": Noul(
        instructions="Does the reply offer something the policy in `allowed_facts` does not allow?",
        criteria=NoulCriteria(
            true="Offers a discount, coupon, compensation, or a return outside the policy",
            false="Only offers what the policy allows, or offers nothing",
        ),
    ),
    "invents_facts": Noul(
        instructions="Does the reply state an order detail that `allowed_facts` does not contain?",
        criteria=NoulCriteria(
            true="Names a price, date, delivery status or refund amount not in `allowed_facts`",
            false="Every order detail matches `allowed_facts`; general next steps are fine",
        ),
    ),
    "answers_customer": Noul(
        instructions="Does the reply respond to what the customer asked in `message`?",
    ),
}


@dataclass
class Verdict:
    """Whether a reply can go out, and which checks fired."""

    send: bool
    flags: dict[str, float]
    ms: float


async def check_reply(
    client: AsyncTypeSafeClient, message: str, reply: str, allowed_facts: str
) -> Verdict:
    """Ask Jev three yes/no questions about the draft reply in one call."""
    state = {"message": message, "reply": reply, "allowed_facts": allowed_facts or "(none)"}
    start = time.perf_counter()
    result = await client.system_one(state, CHECKS)
    ms = (time.perf_counter() - start) * 1000
    p = {name: answer.noul for name, answer in result.nouls.items()}
    fired = {name: p[name] for name in ("over_promises", "invents_facts")
             if p[name] > config.GUARD_HOLD_AT}
    if p["answers_customer"] < config.GUARD_MIN_ANSWERED:
        fired["answers_customer"] = p["answers_customer"]
    return Verdict(send=not fired, flags=fired, ms=ms)
