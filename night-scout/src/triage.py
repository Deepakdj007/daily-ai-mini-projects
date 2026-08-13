"""The cheap pass: score everything the wake found, in one call.

Inputs:  a list of fresh Items
Outputs: one Verdict per item, scores clamped to 0-10

Triage stays outside the graph on purpose. It is stateless and disposable — if the
process dies here, the items were never marked as seen, so the next wake simply
finds them again. Only the items that survive triage are worth a durable thread.
"""

from typing import Sequence

from src import config
from src.llm import call_structured
from src.state import Item, TriageBatch, Verdict

SYSTEM = """You are the triage desk for one developer's overnight news scout.

Their interest profile:
{profile}

You will get a numbered list of items. Score each one 0-10 on how well it matches
that profile, and nothing else. Ignore how famous the author is, how many points a
story has, and how excited the title sounds.

Anchors:
  0-3   off-profile, or news about people and money rather than engineering
  4-6   adjacent and mildly interesting, but they could skip it
  7-8   clearly on-profile, worth 10 minutes in the morning
  9-10  directly changes how they would build something this week

Return one verdict per item. Copy each item_id exactly as given. Be decisive: most
items are not a 7. Keep every reason under 15 words."""


def _render(items: Sequence[Item]) -> str:
    """One compact line per item. Long fields are already truncated upstream."""
    lines = []
    for n, item in enumerate(items, start=1):
        snippet = item.snippet[: config.TRIAGE_SNIPPET_CHARS]
        lines.append(
            f"{n}. item_id: {item.item_id}\n"
            f"   source: {item.source}\n"
            f"   title: {item.title}\n"
            f"   snippet: {snippet or '(none provided)'}"
        )
    return "\n\n".join(lines)


def score_items(items: Sequence[Item]) -> list[Verdict]:
    """Score every fresh item in a single Groq call.

    One call rather than one per item, for three reasons: the interest profile is
    ~250 tokens and re-sending it per item would burn a third of the 8,000
    tokens-per-minute ceiling on repetition; the free tier allows only 30 requests a
    minute; and scoring items side by side lets the model rank them against each
    other instead of drifting toward "everything is a 7".
    """
    if not items:
        return []

    batch = call_structured(
        TriageBatch,
        [
            ("system", SYSTEM.format(profile=config.INTEREST_PROFILE)),
            ("human", f"Score these {len(items)} items:\n\n{_render(items)}"),
        ],
    )

    # The schema cannot carry minimum/maximum (strict mode rejects those keywords),
    # so the range is enforced here. Ids the model invented are dropped.
    valid_ids = {item.item_id for item in items}
    verdicts: list[Verdict] = []
    for verdict in batch.verdicts:
        if verdict.item_id not in valid_ids:
            continue
        verdicts.append(
            Verdict(
                item_id=verdict.item_id,
                score=max(0, min(10, verdict.score)),
                reason=verdict.reason.strip()[:160],
            )
        )

    # An item the model skipped entirely scores 0 rather than staying unscored —
    # otherwise it sits in the carry-over queue forever waiting for a verdict.
    scored = {v.item_id for v in verdicts}
    for item in items:
        if item.item_id not in scored:
            verdicts.append(
                Verdict(item_id=item.item_id, score=0, reason="not scored by triage")
            )
    return verdicts
