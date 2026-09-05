"""Point-in-time retrieval, and the freshness flag that travels with a hit.

Inputs:  a query, a user, a moment, and the two read switches
Outputs: ranked Hits carrying age and a staleness flag, and a rendered block

The retrieval is one vector query with the valid-time window applied as two
range filters. The same SQL answers "what is true now" and "what did I believe
in March"; only the date and the state floor change.

Nothing is re-ranked by freshness. An aged fact that is the best semantic match
stays the best semantic match - it just arrives carrying its age. Dropping it
would trade a confident wrong answer for a confident missing one, and hiding
the age is exactly the failure this project is about.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Sequence

from sqlite_vec import serialize_float32

from src import clock as clockmod
from src import config, embed

_KNN = """
SELECT memory_id, distance FROM memories_vec
WHERE embedding MATCH ? AND k = ?
  AND user_id = ? AND state >= ?
  AND valid_from_epoch <= ? AND valid_to_epoch > ?
ORDER BY distance
"""

_KNN_NO_WINDOW = """
SELECT memory_id, distance FROM memories_vec
WHERE embedding MATCH ? AND k = ? AND user_id = ? AND state >= ?
ORDER BY distance
"""


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved memory, with everything the prompt and the UI need."""

    memory_id: int
    topic: str
    scope: str
    text: str
    value: str
    volatility: str
    status: str
    valid_from: int
    valid_to: int
    last_verified_at: int
    source_ref: str
    distance: float
    age_days: float
    freshness: float
    flag: str

    @property
    def similarity(self) -> float:
        """Cosine similarity, since the vectors are unit length."""
        return 1.0 - self.distance / 2.0


def freshness_of(volatility: str, age_days: float) -> float:
    """Exponential decay on the fact's own half-life. Scheduled facts do not decay."""
    if volatility == "scheduled":
        return 1.0
    half_life = config.HALF_LIFE_DAYS.get(volatility, config.HALF_LIFE_DAYS["slow"])
    return 0.5 ** (max(age_days, 0.0) / half_life)


def _flag_for(row: sqlite3.Row, at: int, age_days: float, fresh: float) -> str:
    """Name the one thing a reader most needs to know about this row's age."""
    if row["status"] == "needs_verification":
        return "needs_verification"
    if row["valid_to"] <= at:
        return "expired"
    if row["status"] == "superseded":
        return "historical"
    if fresh < config.FRESH_FLOOR:
        return "aging"
    return "fresh"


def recall(conn: sqlite3.Connection, user_id: str, query: str, *, clock,
           as_of: int | None = None, k: int = 0, window: bool = True,
           include_retired: bool = False) -> list[Hit]:
    """Retrieve the memories that were live for this user at a moment.

    `window=False` is what a store without bitemporality looks like: no valid-
    time filter, so a finished appointment keeps being retrieved as current.
    """
    at = as_of if as_of is not None else clock.now()
    k = k or config.RECALL_K
    state_floor = 0 if include_retired else 1
    vector = serialize_float32(embed.embed_one(query))

    if window:
        rows = conn.execute(_KNN, (vector, k, user_id, state_floor, at, at)).fetchall()
    else:
        rows = conn.execute(_KNN_NO_WINDOW, (vector, k, user_id, state_floor)).fetchall()
    if not rows:
        return []

    distances = {int(row["memory_id"]): float(row["distance"]) for row in rows}
    placeholders = ", ".join("?" for _ in distances)
    records = conn.execute(
        f"SELECT * FROM memories WHERE id IN ({placeholders})", list(distances)
    ).fetchall()
    by_id = {int(record["id"]): record for record in records}

    hits: list[Hit] = []
    for memory_id, distance in sorted(distances.items(), key=lambda item: item[1]):
        record = by_id.get(memory_id)
        if record is None:
            continue
        age_days = clockmod.days_between(at, int(record["last_verified_at"]))
        fresh = freshness_of(record["volatility"], age_days)
        hits.append(Hit(
            memory_id=memory_id, topic=record["topic"], scope=record["scope"],
            text=record["text"], value=record["value"], volatility=record["volatility"],
            status=record["status"], valid_from=int(record["valid_from"]),
            valid_to=int(record["valid_to"]), last_verified_at=int(record["last_verified_at"]),
            source_ref=record["source_ref"], distance=distance, age_days=age_days,
            freshness=fresh, flag=_flag_for(record, at, age_days, fresh),
        ))
    return hits


def _age_phrase(age_days: float) -> str:
    """'3 weeks ago' reads better in a prompt than an epoch or an ISO date."""
    if age_days < 1:
        return "today"
    if age_days < 14:
        return f"{int(age_days)} days ago"
    if age_days < 60:
        return f"{int(age_days / 7)} weeks ago"
    if age_days < 730:
        return f"{int(age_days / 30)} months ago"
    return f"{age_days / 365:.1f} years ago"


def render(hits: Sequence[Hit], *, show_freshness: bool = True) -> str:
    """Format hits for the answer prompt.

    With show_freshness off the age note disappears entirely. That is a real
    ablation rung, not a cosmetic one: it separates "the store knew the fact was
    old" from "the answer said so".
    """
    if not hits:
        return "(no memories on file)"
    lines = []
    for hit in hits:
        label = f"{hit.topic}/{hit.scope}" if hit.scope else hit.topic
        line = f"- [{label}] {hit.text}"
        if show_freshness:
            confirmed = _age_phrase(hit.age_days)
            if hit.flag == "needs_verification":
                line += f" (last confirmed {confirmed}; NEEDS VERIFICATION - confirm before relying on it)"
            elif hit.flag == "expired":
                line += f" (this ended on {clockmod.to_iso(hit.valid_to)}; no longer current)"
            elif hit.flag == "historical":
                line += f" (superseded; was true until {clockmod.to_iso(hit.valid_to)})"
            elif hit.flag == "aging":
                line += f" (last confirmed {confirmed}; may be out of date)"
            else:
                line += f" (confirmed {confirmed})"
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    from src import store

    conn = store.connect(":memory:")
    june = clockmod.to_epoch("2026-06-01")
    clock = clockmod.FrozenClock("2026-06-10")
    base = dict(user_id="u", scope="", confidence=1.0, source_kind="conversation",
                source_ref="turn:1", source_hash="", expired_at=config.OPEN_END,
                superseded_by=None, valid_to=config.OPEN_END, valid_from=june,
                recorded_at=june, last_verified_at=june, status="active")
    rows = [
        dict(base, topic="city", text="Lives in Pune.", value="Pune", volatility="slow"),
        dict(base, topic="project", text="Working on the payments migration.",
             value="payments migration", volatility="fast"),
    ]
    vectors = embed.embed_texts([row["text"] for row in rows])
    for row, vector in zip(rows, vectors):
        store.insert_memory(conn, row, vector)

    hits = recall(conn, "u", "Which city does the user live in?", clock=clock)
    assert hits and hits[0].topic == "city", [h.topic for h in hits]
    assert hits[0].flag == "fresh", hits[0].flag
    print("fresh   :", render(hits[:1]))

    # Two months on, the fast-moving fact has decayed and the slow one has not.
    # This is the whole point of volatility classes: same age, different verdict.
    clock.advance(days=60)
    hits = recall(conn, "u", "What is the user working on right now?", clock=clock)
    project = next(h for h in hits if h.topic == "project")
    city = next(h for h in hits if h.topic == "city")
    assert project.flag == "aging", f"a 69-day-old fast fact must be flagged: {project.flag}"
    assert city.flag == "fresh", f"a 69-day-old slow fact must not be: {city.flag}"
    print("aged    :", render([project]))
    print("unflagged:", render([project], show_freshness=False))
    print(f"same 69 days: project(fast) {project.freshness:.3f} flagged, "
          f"city(slow) {city.freshness:.3f} not")

    # Far enough out, even the slow fact drops below the floor.
    clock.advance(days=120)
    city_later = next(h for h in recall(conn, "u", "Where does the user live?", clock=clock)
                      if h.topic == "city")
    assert city_later.flag == "aging", city_later.flag
    print(f"OK - at 189 days the slow fact reaches {city_later.freshness:.3f} and is flagged too")
