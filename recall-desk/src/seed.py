"""Seed the store with prior history: account facts, past tickets, playbook v1.

This is also the ablation fixture. Every config in the harness starts from
exactly this state and differs only in which tiers it is allowed to read, so a
scorecard difference can only come from the switch that was flipped.

Two constraints here are load-bearing for probe independence:

- No seeded fact repeats a rate limit or settlement window. Those live in the
  product docs, so probe A tests only "does it know Acme is on Growth".
- No seeded episode mentions any customer's plan or region, or episodic memory
  would answer probe A with semantic memory switched off.
"""

from __future__ import annotations

from src.episodic import EPISODE_FIXTURE, append, all_episodes
from src.procedural import DEFAULT_RULES, load
from src.semantic import Fact, Topic, all_facts, write
from src.store import open_store, wipe

SEED_FACTS: dict[str, list[Fact]] = {
    "acme": [
        Fact(topic=Topic.plan, text="Acme Retail is on the Growth plan."),
        Fact(topic=Topic.region, text="Acme Retail runs in the Mumbai (in-1) region."),
        Fact(
            topic=Topic.webhooks,
            text="Acme Retail self-hosts its webhook endpoint rather than using the hosted one.",
        ),
        Fact(topic=Topic.sdk, text="Acme Retail integrates through the Python SDK v2.3."),
    ],
    "beta": [
        Fact(topic=Topic.plan, text="Beta Corp is on the Starter plan."),
        Fact(topic=Topic.region, text="Beta Corp runs in the Singapore (sg-1) region."),
        # Self-hosted, like Acme. This is deliberate, and it was a bug the first
        # time round: Beta was seeded with PaySetu-hosted webhooks, and the agent
        # correctly refused to apply the egress-firewall lesson to them, because
        # a customer who does not host the endpoint has no firewall in the path.
        # Good reasoning, broken probe. A lesson only transfers when the
        # precondition holds, so the probe customer has to satisfy it.
        # Worded to state the precondition (self-hosted) without using any of
        # probe B's answer tokens. "behind a corporate firewall" would have said
        # the quiet part out loud, and the check below catches exactly that.
        Fact(
            topic=Topic.webhooks,
            text="Beta Corp self-hosts its webhook endpoint on its own infrastructure.",
        ),
    ],
}


def seed(fresh: bool = True) -> None:
    """Write the fixture into the store.

    Args:
        fresh: wipe every tier first, so repeated runs are idempotent.
    """
    store = open_store()
    if fresh:
        wipe(store)

    for customer_id, facts in SEED_FACTS.items():
        write(store, customer_id, facts)

    for episode in EPISODE_FIXTURE:
        append(store, episode)

    playbook = load(store)  # creates v1 from DEFAULT_RULES on first call

    print("semantic memory (per customer):")
    for customer_id in SEED_FACTS:
        for item in all_facts(store, customer_id):
            print(f"  {customer_id:5} [{item.key:8}] {item.value['text']}")

    print(f"\nepisodic memory (global, {len(all_episodes(store))} episodes):")
    for item in all_episodes(store):
        print(f"  {item.key}  {item.value['text']}")

    print(f"\nprocedural memory:\n{playbook.render()}")

    assert len(playbook.rules) == len(DEFAULT_RULES)
    _check_probe_independence(store)
    print("\nOK — seeded.")


def _check_probe_independence(store) -> None:
    """Assert the seed data cannot answer a probe from the wrong tier."""
    blob = " ".join(str(i.value).lower() for i in all_episodes(store))
    for leak in ("growth", "starter", "mumbai", "singapore"):
        assert leak not in blob, f"'{leak}' leaked into episodic memory"

    facts_blob = " ".join(
        str(i.value).lower()
        for customer_id in SEED_FACTS
        for i in all_facts(store, customer_id)
    )
    # Probe A: the limits and settlement windows must live in the product docs
    # and NOT in the account record, so the reply needs the plan from memory and
    # the number from the docs. Neither tier alone is enough.
    for leak in ("req/min", "t+1", "t+3", "500", "100 requests"):
        assert leak not in facts_blob, f"'{leak}' leaked into semantic memory"

    # Probe B: the auto-pause mechanism must exist ONLY in episodic memory —
    # not in the account record, and not in the static docs either.
    from src.prompt import PRODUCT_DOCS

    for leak in ("pause", "resume"):
        assert leak not in facts_blob, f"'{leak}' leaked into semantic memory"
        assert leak not in PRODUCT_DOCS.lower(), f"'{leak}' leaked into the product docs"

    print("\nprobe independence:")
    print("  no plan/region in episodes      (episodic cannot answer probe A)")
    print("  no limits in the account record (semantic alone cannot answer probe A)")
    print("  no auto-pause in facts or docs  (only episodic can answer probe B)")


if __name__ == "__main__":
    seed()
