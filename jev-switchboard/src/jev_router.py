"""The decision layer: one Jev call classifies a message, plain code picks the route.

Inputs:  a customer message and an AsyncTypeSafeClient.
Outputs: Signals (Jev's typed answers) and a Route (where the message goes).
"""

import time
from dataclasses import dataclass

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

from src import config

QUESTIONS = {
    "intent": Choice(
        instructions="What is the customer's main reason for writing?",
        criteria={
            "order_status": "Where an order is, or when it will arrive",
            "product_question": "Details about a product: size, material, care, price",
            "return_exchange": "Wants to return, exchange or get a refund for an item",
            "complaint": "Unhappy with the service and wants it put right",
            "other": "Anything that fits none of the above",
        },
    ),
    "complexity": Score(
        instructions="How hard is this request to resolve?",
        criteria=[
            "Simple: one fact or one standard step",
            "Needs judgment: policy has to be applied to this case",
            "Messy: several problems at once, or an unusual situation",
        ],
    ),
    "wants_human": Noul(instructions="Is the customer asking to speak to a human person?"),
}


@dataclass
class Signals:
    """Jev's answers about one message, plus how long the call took."""

    intent: str
    intent_confidence: float
    complexity: float
    wants_human: float
    ms: float


@dataclass
class Route:
    """Where a message goes, and the reason in plain words."""

    name: str  # "clarify" | "human" | "tool" | "small_llm" | "big_llm"
    reason: str


async def classify(client: AsyncTypeSafeClient, message: str) -> Signals:
    """Ask Jev all three questions about the message in a single call."""
    start = time.perf_counter()
    result = await client.system_one(message, QUESTIONS)
    ms = (time.perf_counter() - start) * 1000
    intent = result.choices["intent"]
    return Signals(
        intent=intent.choice,
        intent_confidence=intent.confidence,
        complexity=result.scores["complexity"].score,
        wants_human=result.nouls["wants_human"].noul,
        ms=ms,
    )


def decide(s: Signals) -> Route:
    """Turn Jev's answers into a route. Ordinary if-statements, readable by anyone."""
    if s.wants_human > config.WANTS_HUMAN_CUTOFF:
        return Route("human", f"asked for a person ({s.wants_human:.2f})")
    if s.intent_confidence < config.MIN_INTENT_CONFIDENCE:
        return Route("clarify", f"not sure of the intent ({s.intent} @ {s.intent_confidence:.2f})")
    if s.intent == "other":
        return Route("clarify", "not a request the desk handles yet")
    if s.intent == "order_status":
        return Route("tool", "status is a database lookup, no LLM needed")
    if s.intent == "product_question":
        return Route("small_llm", "catalog question, a small model is enough")
    if s.intent == "complaint" and s.complexity > config.MAX_AUTO_COMPLEXITY:
        return Route("human", f"messy complaint (complexity {s.complexity:.2f})")
    return Route("big_llm", f"{s.intent} needs policy judgment")
