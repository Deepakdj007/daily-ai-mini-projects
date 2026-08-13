"""Semantic memory — durable facts about one customer's account.

Scope is per customer: namespace ("facts", customer_id). Read by similarity to
the incoming ticket. Written on any turn that reveals something durable.

The write policy is upsert-with-supersede, and it is enforced by the key rather
than by the prompt. Each fact is filed under a fixed topic slug, so "we upgraded
to Growth" lands on the same key as "we're on Starter" and replaces it. Nothing
depends on the model correctly recalling what it wrote last week.
"""

from __future__ import annotations

import hashlib
from enum import Enum

from langgraph.store.base import SearchItem
from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel, Field

from src.config import K_SEM
from src.llm import ask
from src.store import facts_ns


class Topic(str, Enum):
    """The closed set of account attributes this desk tracks.

    A closed set is what makes the supersede work: two statements about the
    plan tier collide on the same key by construction. An open-ended topic
    string would let the model file "plan" and "plan_tier" separately and keep
    both contradicting facts.
    """

    plan = "plan"
    region = "region"
    webhooks = "webhooks"
    sdk = "sdk"
    volume = "volume"
    contact = "contact"
    other = "other"


class Fact(BaseModel):
    """One durable account fact."""

    topic: Topic = Field(description="which account attribute this fact is about")
    text: str = Field(description="the fact as one plain statement, no preamble")


class FactExtraction(BaseModel):
    """Facts worth keeping from a single turn. Empty list is a valid answer."""

    facts: list[Fact] = Field(description="durable account facts, or an empty list")


_EXTRACT_SYSTEM = """You maintain the account record for a payments API support desk.

Read the customer's message and pull out only DURABLE facts about their account
— their plan tier, region, webhook setup, SDK version, transaction volume, or
who to contact.

Do NOT record:
- the symptom they are reporting right now (that is a ticket, not an account fact)
- anything transient ("it broke this morning")
- pleasantries

If the message contains no durable account fact, return an empty list. That is
the normal case, not a failure."""


def _key_for(fact: Fact) -> str:
    """Storage key for a fact.

    Fixed topics use the topic name, so a newer fact replaces the older one.
    `other` facts get a content hash suffix instead, because they have no
    natural identity to collide on — the one case that would need explicit
    conflict resolution to deduplicate.
    """
    if fact.topic is Topic.other:
        digest = hashlib.sha1(fact.text.encode()).hexdigest()[:8]
        return f"other-{digest}"
    return fact.topic.value


def recall(
    store: SqliteStore, customer_id: str, query: str, k: int = K_SEM
) -> list[SearchItem]:
    """Fetch the facts most similar to the incoming ticket.

    Args:
        store: the memory store.
        customer_id: whose facts to search — this tier is per customer.
        query: the ticket text.
        k: how many facts to return.

    Returns:
        Matching facts, highest similarity first.
    """
    return store.search(facts_ns(customer_id), query=query, limit=k)


def all_facts(store: SqliteStore, customer_id: str) -> list[SearchItem]:
    """Every fact on file for a customer, for the inspector panels."""
    return store.search(facts_ns(customer_id), limit=100)


def write(store: SqliteStore, customer_id: str, facts: list[Fact]) -> list[str]:
    """Upsert facts, replacing any earlier fact on the same topic.

    Args:
        store: the memory store.
        customer_id: whose record to update.
        facts: facts to write.

    Returns:
        The keys written.
    """
    written: list[str] = []
    for fact in facts:
        key = _key_for(fact)
        store.put(
            facts_ns(customer_id),
            key,
            {"text": fact.text, "topic": fact.topic.value},
        )
        written.append(key)
    return written


def extract_facts(message: str) -> list[Fact]:
    """Ask the model which durable facts a turn revealed.

    Args:
        message: the customer's latest message.

    Returns:
        Facts to write, possibly empty.
    """
    result = ask(FactExtraction, _EXTRACT_SYSTEM, message)
    return result.facts


def render(hits: list[SearchItem]) -> str:
    """Format recalled facts for the prompt."""
    if not hits:
        return ""
    lines = [f"- {h.value['text']}" for h in hits]
    return "What you know about this customer:\n" + "\n".join(lines)


if __name__ == "__main__":
    # The contradiction test. No API key needed — this exercises the write
    # policy, not the extractor.
    from src.store import open_store

    store = open_store(":memory:")

    write(store, "acme", [Fact(topic=Topic.plan, text="Acme Retail is on the Starter plan.")])
    write(store, "acme", [Fact(topic=Topic.region, text="Acme Retail is in the Mumbai region.")])
    print("after first write:")
    for item in all_facts(store, "acme"):
        print(f"  [{item.key}] {item.value['text']}")

    write(store, "acme", [Fact(topic=Topic.plan, text="Acme Retail upgraded to the Growth plan.")])
    print("\nafter 'we upgraded to Growth':")
    facts = all_facts(store, "acme")
    for item in facts:
        print(f"  [{item.key}] {item.value['text']}")

    plan_facts = [f for f in facts if f.key == "plan"]
    assert len(plan_facts) == 1, f"expected 1 plan fact, got {len(plan_facts)}"
    assert "Growth" in plan_facts[0].value["text"], "the newer plan fact did not win"
    assert len(facts) == 2, f"expected 2 facts total, got {len(facts)}"

    hits = recall(store, "acme", "which plan am I on?")
    print(f"\nrecall 'which plan am I on?' -> {hits[0].value['text']} ({hits[0].score:.3f})")
    print("\nOK — exactly one plan fact, and it is the newer one.")
