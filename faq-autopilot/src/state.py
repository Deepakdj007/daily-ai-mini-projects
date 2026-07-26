"""Durable state for the agent: doc snapshots, FAQ entries, and an audit changelog.

Inputs:  a sqlite3 connection (see connect()).
Outputs: CRUD helpers. The DB survives restarts, which is what lets the agent behave
         like a long-running employee instead of a one-shot script.

Tables:
  doc_snapshots(path, sha256, content, updated_at)   -- last-seen state of each source doc
  faq(id, question, answer, source_file, source_anchor, status, created_at, updated_at)
  changelog(id, ts, event, detail)                   -- every autonomous action, for the audit trail
"""

import sqlite3
from datetime import datetime
from pathlib import Path


def _now() -> str:
    """Return an ISO timestamp to the second, for stamping rows."""
    return datetime.now().isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the database with dict-like row access and ensure the schema exists."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables on first use. Safe to call every run."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS doc_snapshots (
            path TEXT PRIMARY KEY, sha256 TEXT NOT NULL,
            content TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS faq (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL, answer TEXT NOT NULL,
            source_file TEXT NOT NULL, source_anchor TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS changelog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, event TEXT NOT NULL, detail TEXT NOT NULL
        );
        """
    )
    conn.commit()


# --- snapshots -------------------------------------------------------------

def get_snapshot_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    """Return {relative_path: sha256} for every doc the agent has already seen."""
    rows = conn.execute("SELECT path, sha256 FROM doc_snapshots").fetchall()
    return {row["path"]: row["sha256"] for row in rows}


def upsert_snapshot(conn: sqlite3.Connection, path: str, sha256: str, content: str) -> None:
    """Record (or refresh) the last-seen hash + content of one doc."""
    conn.execute(
        "INSERT INTO doc_snapshots (path, sha256, content, updated_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, "
        "content=excluded.content, updated_at=excluded.updated_at",
        (path, sha256, content, _now()),
    )
    conn.commit()


def delete_snapshot(conn: sqlite3.Connection, path: str) -> None:
    """Forget a doc that no longer exists on disk."""
    conn.execute("DELETE FROM doc_snapshots WHERE path = ?", (path,))
    conn.commit()


# --- faq --------------------------------------------------------------------

def faqs_for_source(conn: sqlite3.Connection, source_file: str) -> list[sqlite3.Row]:
    """Return active FAQ entries whose answer is grounded in a given doc."""
    return conn.execute(
        "SELECT * FROM faq WHERE source_file = ? AND status = 'active' ORDER BY id",
        (source_file,),
    ).fetchall()


def active_faqs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return every live FAQ entry, ordered by source then id (for rendering)."""
    return conn.execute(
        "SELECT * FROM faq WHERE status = 'active' ORDER BY source_file, id"
    ).fetchall()


def add_faq(conn: sqlite3.Connection, question: str, answer: str,
            source_file: str, source_anchor: str) -> int:
    """Insert a new FAQ entry and return its id."""
    now = _now()
    cur = conn.execute(
        "INSERT INTO faq (question, answer, source_file, source_anchor, status, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, 'active', ?, ?)",
        (question, answer, source_file, source_anchor, now, now),
    )
    conn.commit()
    return int(cur.lastrowid)


def update_faq(conn: sqlite3.Connection, faq_id: int, question: str,
               answer: str, source_anchor: str) -> None:
    """Overwrite an existing FAQ entry after its source drifted."""
    conn.execute(
        "UPDATE faq SET question=?, answer=?, source_anchor=?, updated_at=? WHERE id=?",
        (question, answer, source_anchor, _now(), faq_id),
    )
    conn.commit()


def mark_stale(conn: sqlite3.Connection, faq_id: int) -> None:
    """Retire a FAQ entry the sources no longer support (kept for history)."""
    conn.execute(
        "UPDATE faq SET status='stale', updated_at=? WHERE id=?", (_now(), faq_id)
    )
    conn.commit()


# --- changelog --------------------------------------------------------------

def log_action(conn: sqlite3.Connection, event: str, detail: str) -> None:
    """Append one line to the audit trail of autonomous actions."""
    conn.execute(
        "INSERT INTO changelog (ts, event, detail) VALUES (?, ?, ?)",
        (_now(), event, detail),
    )
    conn.commit()


def recent_changelog(conn: sqlite3.Connection, limit: int = 200) -> list[sqlite3.Row]:
    """Return the most recent audit-trail rows, newest first."""
    return conn.execute(
        "SELECT * FROM changelog ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
