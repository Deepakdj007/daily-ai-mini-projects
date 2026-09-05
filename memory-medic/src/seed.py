"""The demo user, and the two source files the sweep re-reads.

Inputs:  a store and a clock
Outputs: a seeded memory, plus data/sources/*.md on disk

Seeding writes rows directly rather than replaying conversations through the
model. It is free, it is deterministic, and it means the demo starts from an
identical store every time - which is also what the evaluation needs when it
runs the same fixture through seven different configurations.

The dates matter. Everything is planted far enough in the past that a sweep run
"today" has something to find, and spread widely enough that the three
volatility classes do not all rot at once.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src import clock as clockmod
from src import config, detect, embed, store

DEMO_USER = "demo"

PROFILE_MD = """# Rohan Nair - profile

Employer: Zeta Retail
Job title: Backend engineer
Office: Whitefield, Bengaluru
Emergency contact: Meera Nair
"""

CALENDAR_MD = """# Standing commitments

Parking permit: valid until 2026-08-31
On-call rotation: until 2026-07-14
Gym: Cult Fit Indiranagar, weekday mornings
"""

# (topic, scope, text, value, volatility, days before the seed date, source)
#
# Ages are chosen so that nothing is stale on day 0, the fast fact alone rots by
# day 60, and the slow ones follow around day 120. A seed where half the store
# is already flagged would make the first sweep look like noise instead of a
# finding.
SEED_FACTS = [
    ("name", "", "Is called Rohan Nair.", "Rohan Nair", "stable", 400, "conversation"),
    ("city", "", "Lives in Pune.", "Pune", "slow", 100, "conversation"),
    ("car", "", "Drives a Honda City.", "Honda City", "slow", 90, "conversation"),
    ("employer", "", "Works at Zeta Retail.", "Zeta Retail", "slow", 80, "fixture:profile.md"),
    ("job_title", "", "Is a backend engineer.", "Backend engineer", "slow", 80, "fixture:profile.md"),
    ("diet", "", "Is vegetarian.", "vegetarian", "slow", 70, "conversation"),
    ("project", "current", "Is working on the payments migration.", "payments migration",
     "fast", 20, "conversation"),
    ("communication_preference", "design reviews",
     "Prefers async written feedback for design reviews.", "async written feedback",
     "slow", 60, "conversation"),
    ("communication_preference", "incident calls",
     "Prefers a voice call for incident calls.", "voice call", "slow", 58, "conversation"),
]

# Scheduled facts, given as (topic, scope, text, value, ISO end date, source)
SEED_SCHEDULED = [
    ("appointment", "dentist", "Has a dentist appointment on 12 July 2026.", "2026-07-12",
     "2026-07-12", "conversation"),
    ("other", "parking permit", "Parking permit is valid until 31 August 2026.", "2026-08-31",
     "2026-08-31", "fixture:calendar.md"),
]


def write_sources() -> list[Path]:
    """Put the fixture sources on disk so the sweep has something to re-read."""
    config.SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, body in (("profile.md", PROFILE_MD), ("calendar.md", CALENDAR_MD)):
        path = config.SOURCES_DIR / name
        path.write_text(body, encoding="utf-8")
        written.append(path)
    return written


def seed(conn: sqlite3.Connection, *, clock, user_id: str = DEMO_USER,
         reset_sources: bool = True) -> int:
    """Fill an empty store with a plausible year of remembered facts."""
    if reset_sources:
        write_sources()
    now = clock.now()
    day = 86_400
    rows: list[dict] = []

    for topic, scope, text, value, volatility, age_days, source in SEED_FACTS:
        moment = now - age_days * day
        rows.append(dict(
            user_id=user_id, topic=topic, scope=scope, text=text, value=value,
            volatility=volatility, confidence=1.0, valid_from=moment, valid_to=config.OPEN_END,
            recorded_at=moment, expired_at=config.OPEN_END, superseded_by=None,
            source_kind="fixture" if source.startswith("fixture:") else "conversation",
            source_ref=source if source.startswith("fixture:") else f"turn:{clockmod.to_iso(moment)}",
            source_hash="", last_verified_at=moment, status="active",
        ))

    for topic, scope, text, value, ends, source in SEED_SCHEDULED:
        moment = now - 120 * day
        rows.append(dict(
            user_id=user_id, topic=topic, scope=scope, text=text, value=value,
            volatility="scheduled", confidence=1.0, valid_from=moment,
            valid_to=clockmod.to_epoch(ends), recorded_at=moment, expired_at=config.OPEN_END,
            superseded_by=None,
            source_kind="fixture" if source.startswith("fixture:") else "conversation",
            source_ref=source if source.startswith("fixture:") else f"turn:{clockmod.to_iso(moment)}",
            source_hash="", last_verified_at=moment, status="active",
        ))

    vectors = embed.embed_texts([row["text"] for row in rows])
    for row, vector in zip(rows, vectors):
        store.insert_memory(conn, row, vector)
        store.log_repair(conn, dict(
            user_id=user_id, topic=row["topic"], scope=row["scope"], op="insert",
            detector="human", policy="seed", old_id=None, new_id=None, before_json="{}",
            after_json="{}", evidence="seeded fixture", reason="initial state",
            confidence=1.0, routed="auto", decided_by="oracle", status="applied",
            thread_id=None, proposed_at=row["recorded_at"], decided_at=row["recorded_at"],
        ))

    for name in ("profile.md", "calendar.md"):
        path = config.SOURCES_DIR / name
        if path.exists():
            body = path.read_text(encoding="utf-8")
            store.upsert_source(conn, f"fixture:{name}", "fixture", detect.sha256(body),
                                body, now - 200 * day)
    return len(rows)


if __name__ == "__main__":
    conn = store.connect(":memory:")
    clock = clockmod.FrozenClock("2026-06-01")
    planted = seed(conn, clock=clock)
    assert planted == len(SEED_FACTS) + len(SEED_SCHEDULED)
    assert not store.check_invariants(conn), store.check_invariants(conn)

    def topics(candidates) -> list[str]:
        return sorted(
            conn.execute("SELECT topic FROM memories WHERE id = ?", (c.primary,)).fetchone()["topic"]
            for c in candidates
        )

    quiet = detect.all_candidates(conn, clock=clock)
    assert quiet.total() == 0, f"a freshly seeded store must be quiet, found {quiet.counts()}"
    print(f"seeded {planted} facts; on day 0 the sweep finds {quiet.counts()}")

    # The demo's whole point: move the clock and watch specific things rot.
    clock.advance(days=60)
    day60 = detect.all_candidates(conn, clock=clock)
    print(f"day 60  : aged={topics(day60.aged)} ended={topics(day60.scheduled)}")
    assert topics(day60.aged) == ["project"], "only the fast-moving fact should have rotted yet"
    assert topics(day60.scheduled) == ["appointment"], topics(day60.scheduled)

    clock.advance(days=60)
    day120 = detect.all_candidates(conn, clock=clock)
    aged = topics(day120.aged)
    print(f"day 120 : aged={aged} ended={topics(day120.scheduled)}")
    assert "city" in aged and "car" in aged, "the slow facts follow two months later"
    assert "name" not in aged, "a stable fact must not rot in four months"
    print("OK - day 0 quiet, day 60 the fast fact, day 120 the slow ones, never the stable one")
