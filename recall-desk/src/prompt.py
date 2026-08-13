"""Prompt assembly — where the three tiers meet the static product docs.

The split matters. Product documentation does not vary per customer and does not
accumulate, so it belongs in the system prompt, not in a vector store. Memory
holds only what is specific to a customer or learned over time.

So answering "why am I getting 429s?" takes one fact from memory (this customer
is on Growth) and one from the docs (Growth allows 500/min). Neither alone is
enough, which is what makes the semantic probe measure exactly one thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langgraph.store.sqlite import SqliteStore

from src import episodic, procedural, semantic
from src.config import CUSTOMERS, MemoryConfig

# Static product knowledge. Always available, in every ablation config.
#
# Deliberately says NOTHING about egress IP ranges, firewall allowlists, or
# webhook delivery internals. That knowledge exists only in episodic memory, and
# if it leaked in here the episodic probe would pass with the tier switched off.
PRODUCT_DOCS = """PaySetu product reference:

Plans and limits:
- Starter: 100 requests/minute, settlement T+3
- Growth:  500 requests/minute, settlement T+1
- Scale:   2000 requests/minute, settlement T+0

A 429 response means the plan's per-minute rate limit was exceeded. The limit is
per plan, not per key.

API keys:
- Rotate from Dashboard > Developers > API keys > Rotate.
- The previous key keeps working for 24 hours after rotation.
- Sandbox and live keys are separate and are not interchangeable.

Regions: Mumbai (in-1) and Singapore (sg-1). A customer's region is fixed at
signup and cannot be changed in place.

Webhooks: either PaySetu-hosted or self-hosted. Self-hosted endpoints must return
2xx within 5 seconds or the delivery is retried with backoff.

Invoicing: GSTIN on the business profile drives the tax lines on every invoice."""

ROLE = """You are a support engineer on the PaySetu payments API desk.

Answer the customer's ticket directly and technically. You may only rely on the
product reference and the context given below — never invent limits, dates, or
account details."""


@dataclass
class Trace:
    """What was injected into one turn, and what it cost.

    Character counts are exact. Per-tier token counts are deliberately absent:
    gpt-oss's tokenizer is not available locally, and running the text through a
    different tokenizer would produce a confident wrong number. The true total
    comes back from the API's usage field after the call.
    """

    facts: list = field(default_factory=list)
    episodes: list = field(default_factory=list)
    playbook: procedural.Playbook | None = None
    sizes: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line description of what each tier contributed."""
        parts = [
            f"semantic {len(self.facts)} facts/{self.sizes.get('semantic', 0)}ch",
            f"episodic {len(self.episodes)} eps/{self.sizes.get('episodic', 0)}ch",
            f"procedural {len(self.playbook.rules) if self.playbook else 0} rules"
            f"/{self.sizes.get('procedural', 0)}ch",
        ]
        return " | ".join(parts)


def build(
    store: SqliteStore,
    customer_id: str,
    ticket: str,
    memory: MemoryConfig,
) -> tuple[str, Trace]:
    """Assemble the system prompt for one turn.

    Args:
        store: the memory store.
        customer_id: which customer is writing.
        ticket: their message, used as the retrieval query for both searched tiers.
        memory: which tiers are allowed to contribute.

    Returns:
        The system prompt, and a Trace of what went into it.
    """
    trace = Trace()
    blocks = [ROLE, PRODUCT_DOCS, f"You are speaking to {CUSTOMERS.get(customer_id, customer_id)}."]

    if memory.semantic:
        trace.facts = semantic.recall(store, customer_id, ticket)
        block = semantic.render(trace.facts)
        trace.sizes["semantic"] = len(block)
        if block:
            blocks.append(block)

    if memory.episodic:
        trace.episodes = episodic.recall(store, ticket)
        block = episodic.render(trace.episodes)
        trace.sizes["episodic"] = len(block)
        if block:
            blocks.append(block)

    if memory.procedural:
        trace.playbook = procedural.load(store)
        block = trace.playbook.render()
        trace.sizes["procedural"] = len(block)
        blocks.append(block)

    return "\n\n".join(blocks), trace
