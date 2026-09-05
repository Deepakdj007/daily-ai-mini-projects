"""Deciding how a new fact relates to what is already stored.

Inputs:  one new fact plus the live rows it could collide with
Outputs: a relation per candidate - same, refines, contradicts or unrelated

Which candidates the model is even shown is the load-bearing decision here, and
it happens before the call. With scope awareness on, a fact only meets rows in
a compatible scope, so "for design reviews I prefer async" and "for incident
calls I prefer a voice call" are never compared. The published failure this
avoids is a resolver seeing async-versus-sync, calling it a contradiction, and
deleting a preference the user still holds.

The cache key deliberately excludes row ids and timestamps. Two policies that
have arrived at the same live content ask the same question, so they share the
answer and the ablation pays once.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from src import cache, config, llm, store
from src.extract import Fact

RELATIONS = ["same", "refines", "contradicts", "unrelated"]

RELATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["relations"],
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate_id", "relation", "confidence", "reason"],
                "properties": {
                    "candidate_id": {"type": "integer"},
                    "relation": {"type": "string", "enum": RELATIONS},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}

SYSTEM_PROMPT = """You compare a NEW fact about a user against facts already on file.

For each stored fact, choose exactly one relation:
  same        the new fact restates the stored one. Same value, no new detail.
  refines     the new fact is the stored one with more precision, and both can
              be true at once ("drives a Honda" -> "drives a Honda City").
  contradicts the new fact replaces the stored one. Both cannot be true of the
              same moment ("lives in Pune" -> "lives in Bengaluru").
  unrelated   they are about different things and can both stand.

Judge the VALUES, not the wording.

Two facts that hold in different situations do NOT contradict. If the stored
fact is qualified by a context and the new one is qualified by a different
context, they are unrelated, however opposite the values sound.

A fact about a different person, or one the user only considered, is unrelated.

confidence is 0.0-1.0: how sure you are of the relation you picked.
reason is one short clause, for a human reading an audit log later."""


@dataclass(frozen=True, slots=True)
class Relation:
    """How one new fact relates to one stored row."""

    candidate_id: int
    relation: str
    confidence: float
    reason: str


def candidates_for(conn: sqlite3.Connection, user_id: str, fact: Fact, *, at: int,
                   scope_aware: bool, window: bool = True) -> list[sqlite3.Row]:
    """The stored rows this fact could plausibly collide with.

    With scope_aware on, a scoped fact only meets rows in its own scope or in
    the unqualified scope. That single filter is what keeps two context-specific
    preferences from ever being offered to the model as a contradiction.
    """
    rows = store.live_rows(conn, user_id, fact.topic, at=at, window=window)
    if not scope_aware:
        return rows
    return [row for row in rows if row["scope"] == fact.scope or not row["scope"] or not fact.scope]


def _render_candidate(row: sqlite3.Row) -> str:
    """One stored row as a line the model can reason about."""
    from src import clock as clockmod

    scope = f" [context: {row['scope']}]" if row["scope"] else ""
    return (f"id={row['id']}{scope} {row['text']} "
            f"(value: {row['value']}, on file since {clockmod.to_iso(row['valid_from'])})")


async def classify(fact: Fact, candidates: list[sqlite3.Row], *, model: str = "",
                   use_cache: bool = True) -> tuple[list[Relation], str]:
    """Relate one new fact to each candidate. Returns (relations, error)."""
    if not candidates:
        return [], ""

    scope = f" [context: {fact.scope}]" if fact.scope else ""
    user_message = (
        f"NEW fact{scope}: {fact.text} (value: {fact.value})\n\n"
        "Stored facts:\n" + "\n".join(_render_candidate(row) for row in candidates)
    )
    model = model or config.EXTRACT_MODEL
    key = cache.key(model, {
        "kind": "relate", "prompt": config.PROMPT_VERSION,
        "prompt_hash": cache.fingerprint(SYSTEM_PROMPT),
        "schema": config.RELATION_SCHEMA_VERSION,
        # Content, not identity: two policies holding the same live facts ask
        # the same question and should share the cached answer.
        "new": [fact.topic, fact.scope, fact.value, fact.text],
        "candidates": sorted(
            [row["topic"], row["scope"], row["value"], row["text"]] for row in candidates
        ),
    })
    data, completion = await llm.complete_json(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
        schema=RELATION_SCHEMA, schema_name="relations", model=model,
        cache_key=key if use_cache else "",
        max_completion_tokens=config.MAX_COMPLETION_TOKENS["classify"],
    )
    if data is None:
        return [], completion.error or "classification failed"

    valid_ids = {int(row["id"]) for row in candidates}
    # The model sees ids in the prompt, but a cached answer was produced for a
    # different set of row ids with the same content. Re-key by position when
    # the ids do not line up, so a replay is never silently dropped.
    relations: list[Relation] = []
    raw_items = data.get("relations", [])
    for index, item in enumerate(raw_items):
        candidate_id = int(item.get("candidate_id", -1))
        if candidate_id not in valid_ids:
            if index >= len(candidates):
                continue
            candidate_id = int(candidates[index]["id"])
        relation = str(item.get("relation", "unrelated"))
        if relation not in RELATIONS:
            relation = "unrelated"
        relations.append(Relation(
            candidate_id=candidate_id, relation=relation,
            confidence=min(max(float(item.get("confidence", 0.0)), 0.0), 1.0),
            reason=str(item.get("reason", ""))[:300],
        ))
    return relations, ""


if __name__ == "__main__":
    import asyncio

    from src import clock as clockmod, embed
    from src.extract import Fact as F

    async def _smoke() -> None:
        conn = store.connect(":memory:")
        june = clockmod.to_epoch("2026-06-01")
        base = dict(user_id="u", confidence=1.0, source_kind="conversation", source_ref="t",
                    source_hash="", expired_at=config.OPEN_END, superseded_by=None,
                    valid_to=config.OPEN_END, valid_from=june, recorded_at=june,
                    last_verified_at=june, status="active")
        rows = [
            dict(base, topic="city", scope="", text="Lives in Pune.", value="Pune", volatility="slow"),
            dict(base, topic="communication_preference", scope="design reviews",
                 text="Prefers async written feedback for design reviews.",
                 value="async written feedback", volatility="slow"),
        ]
        for row, vector in zip(rows, embed.embed_texts([r["text"] for r in rows])):
            store.insert_memory(conn, row, vector)

        moved = F(topic="city", scope="", text="Lives in Bengaluru.", value="Bengaluru",
                  volatility="slow", confidence=0.95)
        cands = candidates_for(conn, "u", moved, at=june, scope_aware=True)
        relations, err = await classify(moved, cands)
        assert not err, err
        assert relations[0].relation == "contradicts", relations
        print(f"moved city    -> {relations[0].relation} ({relations[0].confidence:.2f}) "
              f"{relations[0].reason}")

        # The published Mem0 failure: a differently-scoped preference must never
        # even reach the model, so it cannot be resolved as a contradiction.
        incident = F(topic="communication_preference", scope="incident calls",
                     text="Prefers a voice call for incident calls.", value="voice call",
                     volatility="slow", confidence=0.95)
        scoped = candidates_for(conn, "u", incident, at=june, scope_aware=True)
        assert scoped == [], f"scope-aware candidates must be empty, got {[dict(r) for r in scoped]}"

        # With scope awareness off - the incumbent behaviour - it does reach it.
        unscoped = candidates_for(conn, "u", incident, at=june, scope_aware=False)
        assert len(unscoped) == 1, unscoped
        relations2, _ = await classify(incident, unscoped)
        print(f"incident call -> seen by unscoped policy as '{relations2[0].relation}' "
              f"({relations2[0].confidence:.2f})")
        print("OK - scope filter hides the false contradiction before the model sees it")

    asyncio.run(_smoke())
