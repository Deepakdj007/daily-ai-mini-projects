"""The bitemporal memory store: schema, mutations, and the invariant audit.

Inputs:  a sqlite path and a clock
Outputs: a connection with the schema applied, and row-level helpers

Two time axes, borrowed from Graphiti's model. Valid time (valid_from,
valid_to) is when a fact was true in the world. Transaction time (recorded_at,
expired_at) is when this system believed it. A contradiction closes the old
row's valid window and stamps expired_at; it never deletes. That is what lets
"what did I believe about her car in March" and "what is true now" be the same
query with a different date.

Open ends use a sentinel epoch rather than NULL. sqlite-vec metadata columns
support only = != < <= > >=, so a NULL end would be unfilterable in the vector
index - and using the same representation in both tables keeps the
point-in-time predicate byte-identical instead of nearly identical.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Iterable, Sequence

import sqlite_vec
from sqlite_vec import serialize_float32

from src import config

OPEN_END = config.OPEN_END
STATE = {"active": 2, "needs_verification": 1, "superseded": 0, "expired": 0}

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS memories (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT    NOT NULL,
    topic            TEXT    NOT NULL,
    scope            TEXT    NOT NULL DEFAULT '',
    text             TEXT    NOT NULL,
    value            TEXT    NOT NULL,
    volatility       TEXT    NOT NULL CHECK (volatility IN ('stable','slow','fast','scheduled')),
    confidence       REAL    NOT NULL,
    valid_from       INTEGER NOT NULL,
    valid_to         INTEGER NOT NULL DEFAULT {OPEN_END},
    recorded_at      INTEGER NOT NULL,
    expired_at       INTEGER NOT NULL DEFAULT {OPEN_END},
    superseded_by    INTEGER REFERENCES memories(id),
    source_kind      TEXT    NOT NULL CHECK (source_kind IN ('conversation','fixture','url','human')),
    source_ref       TEXT    NOT NULL DEFAULT '',
    source_hash      TEXT    NOT NULL DEFAULT '',
    last_verified_at INTEGER NOT NULL,
    status           TEXT    NOT NULL CHECK (status IN ('active','needs_verification','superseded','expired'))
);
CREATE INDEX IF NOT EXISTS mem_key    ON memories(user_id, topic, scope, status);
CREATE INDEX IF NOT EXISTS mem_verify ON memories(status, volatility, last_verified_at);
CREATE INDEX IF NOT EXISTS mem_source ON memories(source_ref);

CREATE TABLE IF NOT EXISTS sources (
    source_ref      TEXT PRIMARY KEY,
    kind            TEXT NOT NULL CHECK (kind IN ('fixture','url')),
    sha256          TEXT NOT NULL,
    content         TEXT NOT NULL,
    fetched_at      INTEGER NOT NULL,
    last_checked_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS repairs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    topic       TEXT NOT NULL,
    scope       TEXT NOT NULL DEFAULT '',
    op          TEXT NOT NULL CHECK (op IN ('insert','supersede','qualify','expire','reverify','confirm','noop')),
    detector    TEXT NOT NULL CHECK (detector IN ('arrival','aged','source_changed','contradiction','scheduled','human')),
    policy      TEXT NOT NULL,
    old_id      INTEGER,
    new_id      INTEGER,
    before_json TEXT NOT NULL DEFAULT '',
    after_json  TEXT NOT NULL DEFAULT '',
    evidence    TEXT NOT NULL DEFAULT '',
    reason      TEXT NOT NULL DEFAULT '',
    confidence  REAL NOT NULL DEFAULT 0.0,
    routed      TEXT NOT NULL CHECK (routed IN ('auto','parked','forced')),
    decided_by  TEXT NOT NULL CHECK (decided_by IN ('agent','human','oracle')),
    status      TEXT NOT NULL CHECK (status IN ('parked','applied','rejected')),
    thread_id   TEXT,
    proposed_at INTEGER NOT NULL,
    decided_at  INTEGER
);
CREATE INDEX IF NOT EXISTS repairs_status ON repairs(status, proposed_at);
CREATE INDEX IF NOT EXISTS repairs_old    ON repairs(old_id);
CREATE INDEX IF NOT EXISTS repairs_thread ON repairs(thread_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE VIEW IF NOT EXISTS parked AS
SELECT id AS repair_id, thread_id, user_id, topic, scope, op, detector, confidence, proposed_at
FROM repairs WHERE status = 'parked';
"""

VEC_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    memory_id        INTEGER PRIMARY KEY,
    embedding        FLOAT[{config.DIMS}] distance_metric=cosine,
    user_id          TEXT PARTITION KEY,
    state            INTEGER,
    valid_from_epoch INTEGER,
    valid_to_epoch   INTEGER
)
"""


def connect(path=None) -> sqlite3.Connection:
    """Open the store with settings that survive two processes on one file.

    Autocommit (isolation_level=None) keeps write transactions short; every
    multi-statement repair opens its own BEGIN IMMEDIATE so it takes the write
    lock up front instead of upgrading halfway through and deadlocking.
    """
    conn = sqlite3.connect(
        str(path or config.MEMORY_DB),
        timeout=config.SQLITE_TIMEOUT,
        check_same_thread=False,  # Streamlit reruns on a different thread
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.execute(VEC_SCHEMA)
    return conn


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    """sqlite3.Row -> plain dict, so it can be JSON-dumped into an audit row."""
    return dict(row) if row is not None else None


def insert_memory(conn: sqlite3.Connection, fields: dict, vector: Sequence[float]) -> int:
    """Insert one memory row and its vector. Returns the new id."""
    columns = (
        "user_id", "topic", "scope", "text", "value", "volatility", "confidence",
        "valid_from", "valid_to", "recorded_at", "expired_at", "superseded_by",
        "source_kind", "source_ref", "source_hash", "last_verified_at", "status",
    )
    values = [fields.get(name) for name in columns]
    placeholders = ", ".join("?" for _ in columns)
    cur = conn.execute(
        f"INSERT INTO memories ({', '.join(columns)}) VALUES ({placeholders})", values
    )
    memory_id = int(cur.lastrowid)
    conn.execute(
        "INSERT INTO memories_vec (memory_id, embedding, user_id, state, valid_from_epoch, valid_to_epoch)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            memory_id, serialize_float32(list(vector)), fields["user_id"],
            STATE[fields["status"]], int(fields["valid_from"]), int(fields["valid_to"]),
        ),
    )
    return memory_id


def update_row(conn: sqlite3.Connection, memory_id: int, **changes: Any) -> None:
    """Update a memory row and keep its vector-index metadata in step.

    The index update is not optional and not deferred: a row whose status says
    superseded while its index copy still says active is a row that keeps being
    retrieved, which is the exact bug this project exists to prevent.
    """
    if not changes:
        return
    assignments = ", ".join(f"{name} = ?" for name in changes)
    conn.execute(f"UPDATE memories SET {assignments} WHERE id = ?", [*changes.values(), memory_id])
    if not {"status", "valid_from", "valid_to"} & set(changes):
        return
    row = conn.execute(
        "SELECT status, valid_from, valid_to FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    conn.execute(
        "UPDATE memories_vec SET state = ?, valid_from_epoch = ?, valid_to_epoch = ? WHERE memory_id = ?",
        (STATE[row["status"]], int(row["valid_from"]), int(row["valid_to"]), memory_id),
    )


def delete_memory(conn: sqlite3.Connection, memory_id: int) -> None:
    """Physically remove a row. Only the `overwrite` baseline policy calls this."""
    conn.execute("DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,))
    conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))


def live_rows(conn: sqlite3.Connection, user_id: str, topic: str, *, at: int,
              scope: str | None = None, window: bool = True) -> list[sqlite3.Row]:
    """Rows we still believe for a topic, optionally narrowed to one scope.

    `window=False` drops the valid-time filter, which is what a store without
    bitemporality looks like: an ended appointment stays retrievable forever.
    """
    sql = ["SELECT * FROM memories WHERE user_id = ? AND topic = ? AND status IN ('active','needs_verification')"]
    params: list[Any] = [user_id, topic]
    if window:
        sql.append("AND valid_from <= ? AND valid_to > ?")
        params += [at, at]
    if scope is not None:
        sql.append("AND scope = ?")
        params.append(scope)
    sql.append("ORDER BY valid_from DESC, id DESC")
    return conn.execute(" ".join(sql), params).fetchall()


def history(conn: sqlite3.Connection, user_id: str, topic: str, scope: str = "") -> list[sqlite3.Row]:
    """Every version of one key, oldest first - the timeline panel's data."""
    return conn.execute(
        "SELECT * FROM memories WHERE user_id = ? AND topic = ? AND scope = ? "
        "ORDER BY valid_from ASC, id ASC",
        (user_id, topic, scope),
    ).fetchall()


def as_of(conn: sqlite3.Connection, user_id: str, topic: str, scope: str, at: int) -> sqlite3.Row | None:
    """What this system would have said about one key at a given moment."""
    return conn.execute(
        "SELECT * FROM memories WHERE user_id = ? AND topic = ? AND scope = ? "
        "AND valid_from <= ? AND valid_to > ? ORDER BY recorded_at DESC, id DESC LIMIT 1",
        (user_id, topic, scope, at, at),
    ).fetchone()


def log_repair(conn: sqlite3.Connection, fields: dict) -> int:
    """Append one audit row. Returns its id."""
    columns = (
        "user_id", "topic", "scope", "op", "detector", "policy", "old_id", "new_id",
        "before_json", "after_json", "evidence", "reason", "confidence", "routed",
        "decided_by", "status", "thread_id", "proposed_at", "decided_at",
    )
    values = [fields.get(name) for name in columns]
    placeholders = ", ".join("?" for _ in columns)
    cur = conn.execute(
        f"INSERT INTO repairs ({', '.join(columns)}) VALUES ({placeholders})", values
    )
    return int(cur.lastrowid)


def settle_repair(conn: sqlite3.Connection, thread_id: str, status: str, *,
                  decided_by: str, decided_at: int, new_id: int | None = None) -> None:
    """Move a parked audit row to applied or rejected. Only these fields move."""
    conn.execute(
        "UPDATE repairs SET status = ?, decided_by = ?, decided_at = ?, "
        "new_id = COALESCE(?, new_id) WHERE thread_id = ? AND status = 'parked'",
        (status, decided_by, decided_at, new_id, thread_id),
    )


def upsert_source(conn: sqlite3.Connection, source_ref: str, kind: str, sha256: str,
                  content: str, at: int) -> None:
    """Record the snapshot we just extracted from, not a re-read of the file."""
    conn.execute(
        "INSERT INTO sources (source_ref, kind, sha256, content, fetched_at, last_checked_at) "
        "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(source_ref) DO UPDATE SET "
        "sha256 = excluded.sha256, content = excluded.content, "
        "fetched_at = excluded.fetched_at, last_checked_at = excluded.last_checked_at",
        (source_ref, kind, sha256, content, at, at),
    )


def rebuild_index(conn: sqlite3.Connection, embed_fn) -> int:
    """Drop and re-embed the whole vector index from the truth table.

    The recovery path for any skew between the two. Also the proof that the
    index holds no information of its own.
    """
    conn.execute("DROP TABLE IF EXISTS memories_vec")
    conn.execute(VEC_SCHEMA)
    rows = conn.execute("SELECT id, user_id, text, status, valid_from, valid_to FROM memories").fetchall()
    if not rows:
        return 0
    vectors = embed_fn([row["text"] for row in rows])
    for row, vector in zip(rows, vectors):
        conn.execute(
            "INSERT INTO memories_vec (memory_id, embedding, user_id, state, valid_from_epoch, valid_to_epoch)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (row["id"], serialize_float32(vector), row["user_id"], STATE[row["status"]],
             int(row["valid_from"]), int(row["valid_to"])),
        )
    return len(rows)


def check_invariants(conn: sqlite3.Connection) -> list[str]:
    """Audit the store. An empty list means every rule below holds.

    These are checked rather than enforced by constraints because one of them -
    at most one live row per key - is deliberately violated by the `append`
    baseline policy, and the sweep's contradiction detector is what repairs it.
    """
    problems: list[str] = []

    rows = conn.execute(
        "SELECT user_id, topic, scope, COUNT(*) n FROM memories "
        "WHERE status IN ('active','needs_verification') "
        "GROUP BY user_id, topic, scope, valid_from HAVING n > 1"
    ).fetchall()
    for row in rows:
        problems.append(
            f"overlapping live rows: {row['user_id']}/{row['topic']}/{row['scope']!r} x{row['n']}"
        )

    bad = conn.execute(
        f"SELECT id FROM memories WHERE status = 'superseded' AND "
        f"(superseded_by IS NULL OR expired_at >= {OPEN_END})"
    ).fetchall()
    problems += [f"superseded row {r['id']} missing successor or expired_at" for r in bad]

    chain = conn.execute(
        "SELECT a.id, a.valid_to, b.valid_from FROM memories a JOIN memories b ON a.superseded_by = b.id "
        "WHERE a.valid_to > b.valid_from"
    ).fetchall()
    problems += [f"row {r['id']} valid_to runs past its successor's valid_from" for r in chain]

    # valid_from == valid_to is allowed: a fact corrected before it was ever
    # true of an interval has a zero-length window, and the window filter
    # (valid_to > T) correctly never returns it.
    ordering = conn.execute(
        "SELECT id FROM memories WHERE valid_from > valid_to OR last_verified_at < recorded_at "
        "OR recorded_at > expired_at"
    ).fetchall()
    problems += [f"row {r['id']} has times out of order" for r in ordering]

    orphan = conn.execute(
        "SELECT m.id FROM memories m LEFT JOIN memories_vec v ON v.memory_id = m.id WHERE v.memory_id IS NULL"
    ).fetchall()
    problems += [f"row {r['id']} is missing from the vector index" for r in orphan]

    skew = conn.execute(
        "SELECT m.id FROM memories m JOIN memories_vec v ON v.memory_id = m.id "
        "WHERE v.valid_to_epoch != m.valid_to OR v.valid_from_epoch != m.valid_from"
    ).fetchall()
    problems += [f"row {r['id']} index window disagrees with the truth table" for r in skew]

    return problems


def generation(conn: sqlite3.Connection) -> str:
    """An id for this incarnation of the store, minted on first use and on wipe.

    Parked repairs are keyed by a thread id derived from their content, which is
    what stops one problem parking twice. But a wipe resets the row-id sequence,
    so a reseeded store regenerates thread ids identical to ones already decided
    - and a finished checkpoint then short-circuits the new decision, leaving the
    human gate silently not gating. Deleting the checkpoint file covers the
    normal case; this covers the case where the file is locked by a running
    inbox and the delete quietly fails.
    """
    row = conn.execute("SELECT value FROM meta WHERE key = 'generation'").fetchone()
    if row is not None:
        return row["value"]
    minted = uuid.uuid4().hex[:8]
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('generation', ?)", (minted,))
    return minted


def wipe(conn: sqlite3.Connection) -> None:
    """Empty every table, and mint a new generation. Used between eval arms."""
    for table in ("memories", "sources", "repairs"):
        conn.execute(f"DELETE FROM {table}")
    conn.execute("DELETE FROM memories_vec")
    conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('memories','repairs')")
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('generation', ?)",
                 (uuid.uuid4().hex[:8],))


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row counts by status, for `status` and the inbox sidebar."""
    rows = conn.execute("SELECT status, COUNT(*) n FROM memories GROUP BY status").fetchall()
    out = {row["status"]: int(row["n"]) for row in rows}
    out["parked"] = int(conn.execute("SELECT COUNT(*) FROM parked").fetchone()[0])
    return out


if __name__ == "__main__":
    from src import clock as clockmod

    june = clockmod.to_epoch("2026-06-01")
    october = clockmod.to_epoch("2026-10-01")
    december = clockmod.to_epoch("2026-12-01")
    conn = connect(":memory:")

    def fake_vec(seed: float) -> list[float]:
        return [seed] + [0.0] * (config.DIMS - 1)

    base = dict(user_id="u", topic="city", scope="", volatility="slow", confidence=1.0,
                source_kind="conversation", source_ref="turn:1", source_hash="",
                expired_at=OPEN_END, superseded_by=None, valid_to=OPEN_END)
    pune = insert_memory(conn, {**base, "text": "Lives in Pune.", "value": "Pune",
                                "valid_from": june, "recorded_at": june,
                                "last_verified_at": june, "status": "active"}, fake_vec(1.0))
    # Supersede it, Graphiti-style: close the old window, never delete.
    blr = insert_memory(conn, {**base, "text": "Lives in Bengaluru.", "value": "Bengaluru",
                               "valid_from": october, "recorded_at": october,
                               "last_verified_at": october, "status": "active"}, fake_vec(0.9))
    update_row(conn, pune, valid_to=october, expired_at=october,
               status="superseded", superseded_by=blr)

    assert as_of(conn, "u", "city", "", clockmod.to_epoch("2026-08-01"))["value"] == "Pune"
    assert as_of(conn, "u", "city", "", december)["value"] == "Bengaluru"
    assert len(history(conn, "u", "city", "")) == 2, "history keeps both"
    assert len(live_rows(conn, "u", "city", at=december)) == 1

    # A future-dated fact flips on its own date with no code running.
    future = clockmod.to_epoch("2027-03-01")
    insert_memory(conn, {**base, "topic": "car", "text": "Will drive a Nexon.", "value": "Nexon",
                         "valid_from": future, "recorded_at": december,
                         "last_verified_at": december, "status": "active"}, fake_vec(0.5))
    assert not live_rows(conn, "u", "car", at=december), "a future fact is not current yet"
    assert live_rows(conn, "u", "car", at=clockmod.to_epoch("2027-06-01")), "and becomes current later"

    problems = check_invariants(conn)
    assert not problems, problems
    assert rebuild_index(conn, lambda texts: [fake_vec(0.3) for _ in texts]) == 3
    assert not check_invariants(conn), "rebuild must leave the index consistent"
    print(f"OK - supersede keeps history, as_of picks by date, invariants clean; {counts(conn)}")
