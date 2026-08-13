"""Episodic memory — what happened on past tickets, and what actually fixed them.

Scope is GLOBAL: namespace ("episodes",), shared across every customer. A ticket
Acme raised in March is what lets the desk solve Beta Corp's identical symptom in
August. That cross-customer transfer is the whole point of the tier.

Because it is global, episodes are scrubbed of customer identity on write. An
episode is a lesson, not a transcript.

Write policy is append-only. Episodes are never edited, and they deliberately
record the attempts that FAILED — an episode that lists only the successful fix
teaches nothing about what to skip.
"""

from __future__ import annotations

import hashlib
import re

from langgraph.store.base import SearchItem
from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel, Field

from src.config import CUSTOMERS, EPISODE_MIN_SCORE, K_EPI
from src.llm import ask
from src.store import EPISODES_NS


class Episode(BaseModel):
    """One resolved ticket, reduced to a reusable lesson."""

    text: str = Field(
        description="the SYMPTOM in the words a customer would use, one or two sentences"
    )
    tried: list[str] = Field(description="things attempted that did NOT fix it")
    root_cause: str = Field(description="what was actually wrong")
    fix: str = Field(description="what resolved it")


class EpisodeOrNothing(BaseModel):
    """A summarised ticket, or an explicit decision that it is not worth keeping."""

    worth_keeping: bool = Field(
        description="false if the ticket was trivial or was never actually resolved"
    )
    episode: Episode


_SUMMARY_SYSTEM = """You write up resolved support tickets for a payments API desk.

Reduce the conversation to a reusable lesson another agent could apply to a
DIFFERENT customer with the same symptom.

Rules:
- `text` must be the SYMPTOM as a customer would describe it. Never put the
  diagnosis in `text`.
- `tried` must list the dead ends. These are the most valuable part.
- Never include a customer name, account ID, endpoint URL, or API key.
- Set worth_keeping=false if nothing was actually resolved."""

# The seeded ticket history. Lives here rather than in seed.py because the
# retrieval smoke test below has to run against the real fixture, not a copy.
EPISODE_FIXTURE: list[Episode] = [
    # The probe-B lesson. Its root cause is deliberately a PaySetu-specific
    # mechanism rather than generic network troubleshooting.
    #
    # An earlier version blamed a firewall egress allowlist, and that made a
    # useless probe: "check your firewall rules" is advice the model gives from
    # general knowledge with episodic memory switched off, so the assertion
    # passed either way. An auto-pause threshold that has to be cleared by hand
    # is not something a model can guess — it either read it here or it did not.
    Episode(
        text="Webhooks stopped firing overnight. Nothing changed on our side.",
        tried=[
            "checked endpoint health",
            "verified the signing secret",
            "rotated the API key",
            "checked firewall and network rules",
        ],
        root_cause=(
            "the webhook subscription auto-paused after 50 consecutive delivery "
            "failures during a short endpoint outage, and stayed paused after the "
            "endpoint came back healthy"
        ),
        fix=(
            "resume the subscription from Dashboard > Webhooks > Resume. Deliveries "
            "do not restart on their own once a subscription is paused"
        ),
    ),
    Episode(
        text="A shopper was charged twice for the same order.",
        tried=["searched for a duplicate order record", "checked for a retry storm"],
        root_cause="the create-payment call was retried without an idempotency key",
        fix="send a stable idempotency key on every create-payment retry",
    ),
    Episode(
        text="Our invoices are being rejected by our accountant.",
        tried=["re-downloaded the PDFs", "checked the invoice template"],
        root_cause="the GSTIN field was left blank on the business profile, so tax lines rendered empty",
        fix="fill in GSTIN on the business profile and regenerate the invoices",
    ),
    Episode(
        text="Payments succeed in testing but every live call returns 401.",
        tried=["regenerated the key", "checked the Authorization header format"],
        root_cause="the sandbox key was being sent to the live endpoint",
        fix="use the live key for live endpoints — the two key sets are not interchangeable",
    ),
]


def scrub(text: str) -> str:
    """Remove known customer identities from episode text.

    The summariser is also told not to include names, but a prompt is a request.
    This is the part that actually holds — the same reason `safe-sql-agent` puts
    its guard in code rather than in the system prompt.
    """
    cleaned = text
    for name in CUSTOMERS.values():
        cleaned = re.sub(re.escape(name), "the customer", cleaned, flags=re.IGNORECASE)
        first = name.split()[0]
        cleaned = re.sub(rf"\b{re.escape(first)}\b", "the customer", cleaned, flags=re.IGNORECASE)
    return cleaned


def _key_for(episode: Episode) -> str:
    """Content-addressed key, so writing the same lesson twice is a no-op."""
    material = f"{episode.text}|{episode.root_cause}"
    return f"ep-{hashlib.sha1(material.encode()).hexdigest()[:10]}"


def append(store: SqliteStore, episode: Episode) -> str:
    """Store one episode. Never edits an existing one.

    Only `text` is embedded — the diagnosis and fix ride along unindexed, so a
    symptom query matches a symptom.

    Args:
        store: the memory store.
        episode: the lesson to file.

    Returns:
        The key written.
    """
    key = _key_for(episode)
    store.put(
        EPISODES_NS,
        key,
        {
            "text": scrub(episode.text),
            "tried": [scrub(t) for t in episode.tried],
            "root_cause": scrub(episode.root_cause),
            "fix": scrub(episode.fix),
        },
    )
    return key


def recall(
    store: SqliteStore,
    symptom: str,
    k: int = K_EPI,
    min_score: float = EPISODE_MIN_SCORE,
) -> list[SearchItem]:
    """Find past tickets whose symptom resembles this one.

    Applies a relevance floor. Without it, a rate-limit ticket still gets the two
    nearest episodes injected at scores around 0.18 — paying tokens for lessons
    about webhooks and 401s that have nothing to do with the question.

    Args:
        store: the memory store.
        symptom: the incoming ticket text.
        k: how many episodes to return at most.
        min_score: drop anything below this similarity.

    Returns:
        Matching episodes above the floor, closest symptom first.
    """
    hits = store.search(EPISODES_NS, query=symptom, limit=k)
    return [h for h in hits if (h.score or 0.0) >= min_score]


def all_episodes(store: SqliteStore) -> list[SearchItem]:
    """Every episode on file, for the inspector panel."""
    return store.search(EPISODES_NS, limit=100)


def summarize_episode(transcript: str) -> Episode | None:
    """Reduce a resolved ticket to a lesson.

    Args:
        transcript: the whole thread, not a single turn — an episode needs the
            outcome, which only appears at the end.

    Returns:
        The episode to file, or None if the ticket was not worth keeping.
    """
    result = ask(EpisodeOrNothing, _SUMMARY_SYSTEM, transcript)
    return result.episode if result.worth_keeping else None


def render(hits: list[SearchItem]) -> str:
    """Format recalled episodes as a few-shot block, dead ends included."""
    if not hits:
        return ""
    blocks = []
    for h in hits:
        v = h.value
        tried = ", ".join(v.get("tried", [])) or "nothing recorded"
        blocks.append(
            f"- Symptom: {v['text']}\n"
            f"  Already ruled out on a past ticket: {tried}\n"
            f"  Actual root cause: {v['root_cause']}\n"
            f"  What fixed it: {v['fix']}"
        )
    return "Similar tickets this desk has already solved:\n" + "\n".join(blocks)


if __name__ == "__main__":
    # THE GATE. If a symptom query does not retrieve the right lesson out of
    # four, nothing downstream in this project is worth building.
    from src.store import open_store

    store = open_store(":memory:")
    for ep in EPISODE_FIXTURE:
        append(store, ep)

    query = "Our webhooks suddenly stopped arriving on the 4th. We changed nothing."
    print(f"query: {query}\n")
    ranked = recall(store, query, k=4, min_score=0.0)  # floor off, to see the spread
    for rank, hit in enumerate(ranked, 1):
        keep = "kept" if (hit.score or 0) >= EPISODE_MIN_SCORE else "below floor"
        print(f"  {rank}. {hit.score:.4f}  {hit.value['text'][:52]:52} {keep}")

    assert "auto-paused" in ranked[0].value["root_cause"], (
        f"expected the auto-pause episode top-1, got: {ranked[0].value['root_cause']}"
    )
    print(f"\nmargin over runner-up: {ranked[0].score - ranked[1].score:.4f}")

    kept = recall(store, query)
    assert len(kept) == 1, f"floor should keep exactly the right episode, kept {len(kept)}"

    # An unrelated ticket must recall nothing rather than the nearest noise.
    noise = recall(store, "We're getting 429s on the payouts endpoint. What's our limit?")
    print(f"unrelated ticket recalls {len(noise)} episodes (floor {EPISODE_MIN_SCORE})")
    assert noise == [], "an unrelated ticket pulled in an episode"

    for item in all_episodes(store):
        blob = str(item.value)
        for name in CUSTOMERS.values():
            assert name.lower() not in blob.lower(), f"{name} leaked into {item.key}"
    print("no customer identity in any episode value")
    print("\nOK — retrieval discriminates between four episodes.")
