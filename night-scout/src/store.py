"""Durable memory: what the agent has seen, what is parked, what happened each wake.

Inputs:  a path to nightscout.db
Outputs: a sqlite3.Connection and typed helpers over three tables

seen_items is an item store, not just a set of ids. Keeping the score and a
deep_done flag is what lets a high scorer that missed tonight's per-wake budget be
picked up by a later wake instead of being silently dropped.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from src.state import Item, Verdict

SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_items (
    item_id       TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    title         TEXT NOT NULL,
    url           TEXT NOT NULL,
    snippet       TEXT NOT NULL,
    ts            REAL NOT NULL,
    created_at    TEXT NOT NULL,
    score         INTEGER,
    reason        TEXT,
    deep_done     INTEGER NOT NULL DEFAULT 0,
    first_seen_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS seen_pending ON seen_items(deep_done, score);

CREATE TABLE IF NOT EXISTS drafts (
    thread_id  TEXT PRIMARY KEY,
    item_id    TEXT NOT NULL,
    night_id   TEXT NOT NULL,
    source     TEXT NOT NULL,
    title      TEXT NOT NULL,
    url        TEXT NOT NULL,
    score      INTEGER NOT NULL,
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS drafts_status ON drafts(status);

CREATE TABLE IF NOT EXISTS wake_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    night_id  TEXT NOT NULL,
    sim_time  TEXT NOT NULL,
    real_time TEXT NOT NULL,
    polled    INTEGER NOT NULL,
    fresh     INTEGER NOT NULL,
    triaged   INTEGER NOT NULL,
    parked    INTEGER NOT NULL,
    queued    INTEGER NOT NULL
);
"""


def now_iso() -> str:
    """UTC timestamp, second resolution. Used for every stored real time."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path, timeout: float = 30.0) -> sqlite3.Connection:
    """Open the database with settings two processes can share.

    WAL lets the Streamlit inbox read while the night process writes. busy_timeout
    makes the loser of a write race wait instead of raising `database is locked`
    immediately — the default is 5 seconds, which is not enough. isolation_level=None
    is autocommit, which keeps write transactions as short as possible.
    """
    conn = sqlite3.connect(
        str(db_path),
        timeout=timeout,
        check_same_thread=False,  # Streamlit reruns the script on another thread
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


# --- seen_items -------------------------------------------------------------


def known_item_ids(conn: sqlite3.Connection) -> set[str]:
    """Every item id the agent has ever polled. This is the anti-repeat memory."""
    return {row["item_id"] for row in conn.execute("SELECT item_id FROM seen_items")}


def remember_items(conn: sqlite3.Connection, items: Sequence[Item]) -> None:
    """Store fresh items untriaged. INSERT OR IGNORE makes a re-poll a no-op."""
    conn.executemany(
        "INSERT OR IGNORE INTO seen_items "
        "(item_id, source, title, url, snippet, ts, created_at, first_seen_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (i.item_id, i.source, i.title, i.url, i.snippet, i.ts, i.created_at,
             now_iso())
            for i in items
        ],
    )


def untriaged(conn: sqlite3.Connection, limit: int) -> list[Item]:
    """Stored items that still have no score, newest first.

    Normally this is exactly what the current wake just polled. It is also the
    recovery path: if a triage call fails, those items keep score NULL, and because
    they are already stored they would never come back through poll_all — which
    treats them as seen. Reading the queue from the database instead of from the
    poll result means a failed call costs one wake, not the items.
    """
    rows = conn.execute(
        "SELECT item_id, source, title, url, snippet, ts, created_at FROM seen_items "
        "WHERE score IS NULL ORDER BY ts DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [Item(**dict(row)) for row in rows]


def record_scores(conn: sqlite3.Connection, verdicts: Iterable[Verdict]) -> None:
    """Attach triage results to stored items so a later wake can act on them."""
    conn.executemany(
        "UPDATE seen_items SET score = ?, reason = ? WHERE item_id = ?",
        [(v.score, v.reason, v.item_id) for v in verdicts],
    )


def pending_deep(conn: sqlite3.Connection, threshold: int, limit: int) -> list[Item]:
    """Items that earned a draft but have not had one yet, best first.

    This is the carry-over queue. Without it, anything that scored above the bar
    while the per-wake draft budget was already spent would be marked seen and never
    looked at again.
    """
    rows = conn.execute(
        "SELECT item_id, source, title, url, snippet, ts, created_at FROM seen_items "
        "WHERE deep_done = 0 AND score IS NOT NULL AND score >= ? "
        "ORDER BY score DESC, first_seen_at ASC LIMIT ?",
        (threshold, limit),
    ).fetchall()
    return [Item(**dict(row)) for row in rows]


def item_score(conn: sqlite3.Connection, item_id: str) -> tuple[int, str]:
    """The stored score and reason for one item."""
    row = conn.execute(
        "SELECT score, reason FROM seen_items WHERE item_id = ?", (item_id,)
    ).fetchone()
    if row is None:
        return 0, ""
    return int(row["score"] or 0), row["reason"] or ""


def mark_deep_done(conn: sqlite3.Connection, item_id: str) -> None:
    """Retire an item from the carry-over queue, drafted or failed."""
    conn.execute("UPDATE seen_items SET deep_done = 1 WHERE item_id = ?", (item_id,))


def queue_depth(conn: sqlite3.Connection, threshold: int) -> int:
    """How many winners are still waiting for a draft."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM seen_items "
        "WHERE deep_done = 0 AND score IS NOT NULL AND score >= ?",
        (threshold,),
    ).fetchone()
    return int(row["n"])


# --- drafts -----------------------------------------------------------------


def record_parked(
    conn: sqlite3.Connection, thread_id: str, item: Item, night_id: str, score: int
) -> None:
    """Index a parked thread so the inbox can find it.

    Pointers only — the draft body itself lives in exactly one place, the LangGraph
    checkpoint. ON CONFLICT makes a re-park after an edit idempotent.
    """
    conn.execute(
        "INSERT INTO drafts "
        "(thread_id, item_id, night_id, source, title, url, score, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'parked', ?) "
        "ON CONFLICT(thread_id) DO UPDATE SET status = 'parked', decided_at = NULL",
        (thread_id, item.item_id, night_id, item.source, item.title, item.url,
         score, now_iso()),
    )


def parked_drafts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Everything waiting on a human, best score first. One indexed query."""
    return conn.execute(
        "SELECT * FROM drafts WHERE status = 'parked' "
        "ORDER BY score DESC, created_at ASC"
    ).fetchall()


def settle_draft(conn: sqlite3.Connection, thread_id: str, status: str) -> None:
    """Record the human's decision against the index row."""
    conn.execute(
        "UPDATE drafts SET status = ?, decided_at = ? WHERE thread_id = ?",
        (status, now_iso(), thread_id),
    )


def deep_passes_tonight(conn: sqlite3.Connection, night_id: str) -> int:
    """Drafts already spent tonight — approved, rejected or still parked."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM drafts WHERE night_id = ?", (night_id,)
    ).fetchone()
    return int(row["n"])


def status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Draft counts by status, for the status command and the inbox sidebar."""
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM drafts GROUP BY status")
    return {row["status"]: int(row["n"]) for row in rows}


# --- wake_log ---------------------------------------------------------------


def log_wake(
    conn: sqlite3.Connection, night_id: str, sim_time: str,
    polled: int, fresh: int, triaged: int, parked: int, queued: int,
) -> None:
    """One row per wake. This is the log you read in the morning."""
    conn.execute(
        "INSERT INTO wake_log "
        "(night_id, sim_time, real_time, polled, fresh, triaged, parked, queued) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (night_id, sim_time, now_iso(), polled, fresh, triaged, parked, queued),
    )


def recent_wakes(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    """The most recent wakes, newest first."""
    return conn.execute(
        "SELECT * FROM wake_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
