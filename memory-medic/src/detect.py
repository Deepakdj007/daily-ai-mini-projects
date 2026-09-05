"""Finding stale memories when nothing has arrived to contradict them.

Inputs:  the store and a moment
Outputs: Candidates - rows worth a second look, with the evidence why

Four detectors, all pure SQL plus one file read. No model is called here, which
is what makes a quiet sweep free: with nothing drifting, a tick costs a handful
of indexed queries and returns an empty list.

Rows that already have a parked repair are excluded from every detector. Without
that, a sweep on a timer would re-park the same decision every interval until
somebody got round to reviewing the pile.
"""

from __future__ import annotations

import difflib
import hashlib
import sqlite3
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from sqlite_vec import serialize_float32

from src import config, embed
from src import clock as clockmod

_NOT_PARKED = ("NOT EXISTS (SELECT 1 FROM repairs r WHERE r.old_id = m.id AND r.status = 'parked')")


@dataclass(frozen=True, slots=True)
class Candidate:
    """One row (or pair) the sweep thinks may be stale, and why."""

    kind: str                 # aged | source_changed | contradiction | scheduled
    user_id: str
    memory_ids: list[int]
    evidence: str
    source_ref: str = ""
    payload: dict = field(default_factory=dict)

    @property
    def primary(self) -> int:
        return self.memory_ids[0]


@dataclass(frozen=True, slots=True)
class Candidates:
    """Everything one sweep found, grouped by detector."""

    aged: list[Candidate] = field(default_factory=list)
    source_changed: list[Candidate] = field(default_factory=list)
    contradiction: list[Candidate] = field(default_factory=list)
    scheduled: list[Candidate] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.aged or self.source_changed or self.contradiction or self.scheduled)

    def total(self) -> int:
        return len(self.aged) + len(self.source_changed) + len(self.contradiction) + len(self.scheduled)

    def counts(self) -> dict[str, int]:
        return {"aged": len(self.aged), "source_changed": len(self.source_changed),
                "contradiction": len(self.contradiction), "scheduled": len(self.scheduled)}


def sha256(text: str) -> str:
    """Content hash. Change detection is by content, never by modified time."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_source(source_ref: str) -> str | None:
    """Read a source by reference. Returns None when it has gone away.

    'fixture:name.md' reads data/sources/name.md, which is what the demo edits.
    'url:https://...' fetches the page, so a reader can point this at something
    real without changing any other code.
    """
    if source_ref.startswith("fixture:"):
        path = config.SOURCES_DIR / source_ref.split(":", 1)[1]
        return path.read_text(encoding="utf-8") if path.exists() else None
    if source_ref.startswith("url:"):
        url = source_ref.split(":", 1)[1]
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "memory-medic/0.1"})
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception:
            return None
    return None


def _diff(old: str, new: str) -> str:
    """A unified diff, trimmed to what fits on an inbox card."""
    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(), fromfile="on file", tofile="now", lineterm="", n=1
    )
    return "\n".join(list(lines)[:40])[:config.EVIDENCE_MAX_CHARS]


def find_aged(conn: sqlite3.Connection, at: int) -> list[Candidate]:
    """Active facts that have outlived their half-life without fresh evidence."""
    rows = conn.execute(
        f"""SELECT m.* FROM memories m
            WHERE m.status = 'active' AND m.volatility != 'scheduled'
              AND m.last_verified_at < ? - (CASE m.volatility
                    WHEN 'stable' THEN ? WHEN 'slow' THEN ? ELSE ? END)
              AND {_NOT_PARKED}
            ORDER BY m.last_verified_at""",
        (at, config.HALF_LIFE_DAYS["stable"] * 86400,
         config.HALF_LIFE_DAYS["slow"] * 86400, config.HALF_LIFE_DAYS["fast"] * 86400),
    ).fetchall()
    out = []
    for row in rows:
        age = clockmod.days_between(at, int(row["last_verified_at"]))
        half_life = config.HALF_LIFE_DAYS.get(row["volatility"], config.HALF_LIFE_DAYS["slow"])
        out.append(Candidate(
            kind="aged", user_id=row["user_id"], memory_ids=[int(row["id"])],
            evidence=f"last confirmed {age:.0f} days ago; a {row['volatility']} fact's "
                     f"half-life is {half_life} days",
        ))
    return out


def find_scheduled(conn: sqlite3.Connection, at: int) -> list[Candidate]:
    """Scheduled facts whose window has closed. They ended; they were never wrong."""
    rows = conn.execute(
        f"""SELECT m.* FROM memories m
            WHERE m.volatility = 'scheduled' AND m.status IN ('active','needs_verification')
              AND m.valid_to <= ? AND {_NOT_PARKED}""",
        (at,),
    ).fetchall()
    return [Candidate(
        kind="scheduled", user_id=row["user_id"], memory_ids=[int(row["id"])],
        evidence=f"ended on {clockmod.to_iso(int(row['valid_to']))}",
    ) for row in rows]


def find_source_changed(conn: sqlite3.Connection, at: int) -> list[Candidate]:
    """Sources whose content no longer matches the snapshot we extracted from."""
    out: list[Candidate] = []
    for source in conn.execute("SELECT * FROM sources").fetchall():
        current = fetch_source(source["source_ref"])
        gone = current is None
        if not gone and sha256(current) == source["sha256"]:
            conn.execute("UPDATE sources SET last_checked_at = ? WHERE source_ref = ?",
                         (at, source["source_ref"]))
            continue
        rows = conn.execute(
            f"""SELECT m.* FROM memories m WHERE m.source_ref = ?
                AND m.status IN ('active','needs_verification') AND {_NOT_PARKED}""",
            (source["source_ref"],),
        ).fetchall()
        if not rows:
            continue
        evidence = (f"source {source['source_ref']} has been deleted" if gone
                    else _diff(source["content"], current))
        out.append(Candidate(
            kind="source_changed", user_id=rows[0]["user_id"],
            memory_ids=[int(row["id"]) for row in rows], evidence=evidence,
            source_ref=source["source_ref"],
            payload={"content": current or "", "gone": gone,
                     "sha256": sha256(current) if current else ""},
        ))
    return out


def find_contradictions(conn: sqlite3.Connection, at: int) -> list[Candidate]:
    """Two live facts that cannot both be current.

    c1 is structural: more than one live row on the same key. That is what a
    store with no resolution leaves behind, and what a bug in the write path
    would leave behind too.

    c2 is semantic: near-identical facts filed under differently spelled
    scopes ("design review" and "design reviews"), which the key check cannot
    see because the keys genuinely differ.
    """
    out: list[Candidate] = []
    groups = conn.execute(
        f"""SELECT m.user_id, m.topic, m.scope, COUNT(*) n, GROUP_CONCAT(m.id) ids
            FROM memories m
            WHERE m.status IN ('active','needs_verification')
              AND m.valid_from <= ? AND m.valid_to > ? AND {_NOT_PARKED}
            GROUP BY m.user_id, m.topic, m.scope HAVING n > 1""",
        (at, at),
    ).fetchall()
    for group in groups:
        ids = [int(part) for part in group["ids"].split(",")]
        rows = conn.execute(
            f"SELECT id, value, text FROM memories WHERE id IN ({','.join('?' * len(ids))})", ids
        ).fetchall()
        if len({row["value"].strip().lower() for row in rows}) < 2:
            continue  # same value twice is a duplicate, not a contradiction
        out.append(Candidate(
            kind="contradiction", user_id=group["user_id"], memory_ids=ids,
            evidence="two live facts share one key: " + " | ".join(
                f"#{row['id']} {row['value']}" for row in rows),
            payload={"case": "c1", "topic": group["topic"], "scope": group["scope"]},
        ))

    seen: set[tuple[int, int]] = set()
    live = conn.execute(
        f"""SELECT m.* FROM memories m WHERE m.status IN ('active','needs_verification')
            AND m.valid_from <= ? AND m.valid_to > ? AND m.scope != '' AND {_NOT_PARKED}""",
        (at, at),
    ).fetchall()
    for row in live:
        vector = serialize_float32(embed.embed_one(row["text"]))
        neighbours = conn.execute(
            """SELECT memory_id, distance FROM memories_vec
               WHERE embedding MATCH ? AND k = 3 AND user_id = ? AND state >= 1
               ORDER BY distance""",
            (vector, row["user_id"]),
        ).fetchall()
        for neighbour in neighbours:
            other_id = int(neighbour["memory_id"])
            if other_id == int(row["id"]):
                continue
            pair = (min(other_id, int(row["id"])), max(other_id, int(row["id"])))
            if pair in seen:
                continue
            other = conn.execute("SELECT * FROM memories WHERE id = ?", (other_id,)).fetchone()
            if other is None or other["topic"] != row["topic"] or not other["scope"]:
                continue
            if other["scope"] == row["scope"] or other["value"].strip().lower() == row["value"].strip().lower():
                continue
            if 1.0 - float(neighbour["distance"]) / 2.0 < config.SCOPE_SIM_THRESHOLD:
                continue
            seen.add(pair)
            out.append(Candidate(
                kind="contradiction", user_id=row["user_id"], memory_ids=list(pair),
                evidence=f"near-identical facts under different scopes: "
                         f"#{row['id']} [{row['scope']}] vs #{other_id} [{other['scope']}]",
                payload={"case": "c2", "topic": row["topic"], "scope": row["scope"]},
            ))
    return out


def all_candidates(conn: sqlite3.Connection, *, clock, at: int | None = None) -> Candidates:
    """Run every detector once."""
    moment = at if at is not None else clock.now()
    return Candidates(
        aged=find_aged(conn, moment),
        source_changed=find_source_changed(conn, moment),
        contradiction=find_contradictions(conn, moment),
        scheduled=find_scheduled(conn, moment),
    )


if __name__ == "__main__":
    from src import store

    conn = store.connect(":memory:")
    clock = clockmod.FrozenClock("2026-06-01")
    june = clock.now()
    base = dict(user_id="u", scope="", confidence=1.0, source_kind="conversation",
                source_ref="", source_hash="", expired_at=config.OPEN_END, superseded_by=None,
                valid_to=config.OPEN_END, valid_from=june, recorded_at=june,
                last_verified_at=june, status="active")
    rows = [
        dict(base, topic="project", text="Working on the payments migration.",
             value="payments migration", volatility="fast"),
        dict(base, topic="city", text="Lives in Pune.", value="Pune", volatility="slow"),
        dict(base, topic="appointment", scope="dentist", text="Dentist appointment on 20 June.",
             value="2026-06-20", volatility="scheduled", valid_to=clockmod.to_epoch("2026-06-20")),
        dict(base, topic="communication_preference", scope="design reviews",
             text="Prefers async written feedback for design reviews.",
             value="async written feedback", volatility="slow"),
        dict(base, topic="communication_preference", scope="incident calls",
             text="Prefers a voice call for incident calls.", value="voice call", volatility="slow"),
    ]
    for row, vector in zip(rows, embed.embed_texts([r["text"] for r in rows])):
        store.insert_memory(conn, row, vector)

    quiet = all_candidates(conn, clock=clock)
    assert not quiet, f"a fresh store must be quiet, found {quiet.counts()}"
    print(f"day 0    -> {quiet.counts()} (no work, so no model call)")

    # Two months on: the fast fact has aged out, the appointment has ended, and
    # the slow fact has not moved.
    clock.advance(days=60)
    found = all_candidates(conn, clock=clock)
    assert len(found.aged) == 1 and found.aged[0].primary == 1, found.aged
    assert len(found.scheduled) == 1, found.scheduled
    assert not found.contradiction, found.contradiction
    print(f"day 60   -> {found.counts()}")
    print(f"   aged      : {found.aged[0].evidence}")
    print(f"   scheduled : {found.scheduled[0].evidence}")

    # The two scoped preferences must NOT pair: that is the false positive the
    # similarity floor exists to prevent.
    pairs = [c for c in found.contradiction if c.payload.get("case") == "c2"]
    assert not pairs, f"scoped preferences must not be flagged as contradicting: {pairs}"
    print("   scoped preferences correctly left alone")
    print(f"OK - detectors are quiet when nothing drifts, and specific when it does")
