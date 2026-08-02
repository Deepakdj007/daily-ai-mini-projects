"""Durable state for notes and reminders — the facts the agent recalls between messages.

Inputs:  a sqlite3 connection (see connect()).
Outputs: CRUD helpers. This DB survives restarts, which is what lets a reminder or a
saved note still be there after the bot process is killed and started again.

Tables:
  notes(id, text, tags, created_at)
  reminders(id, chat_id, text, due_at, status, created_at)   -- status: pending|fired|cancelled
"""

import sqlite3
from datetime import datetime
from pathlib import Path


def _now() -> str:
    """Return an ISO timestamp to the second, for stamping rows."""
    return datetime.now().isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the database with dict-like row access and ensure the schema exists.

    check_same_thread=False because LangChain runs sync @tool functions in a
    threadpool executor, so this connection is opened on the event-loop thread
    but used from whichever worker thread the executor picks for each call.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables on first use. Safe to call every run."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL, text TEXT NOT NULL, due_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()


# --- notes -------------------------------------------------------------------

def add_note(conn: sqlite3.Connection, text: str, tags: str = "") -> int:
    """Save a note and return its id."""
    cur = conn.execute(
        "INSERT INTO notes (text, tags, created_at) VALUES (?, ?, ?)",
        (text, tags, _now()),
    )
    conn.commit()
    return int(cur.lastrowid)


def search_notes(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[sqlite3.Row]:
    """Return notes whose text or tags contain the query (case-insensitive substring)."""
    return conn.execute(
        "SELECT * FROM notes WHERE text LIKE ? OR tags LIKE ? ORDER BY id DESC LIMIT ?",
        (f"%{query}%", f"%{query}%", limit),
    ).fetchall()


def recent_notes(conn: sqlite3.Connection, limit: int = 5) -> list[sqlite3.Row]:
    """Return the most recently saved notes, newest first."""
    return conn.execute(
        "SELECT * FROM notes ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()


# --- reminders -----------------------------------------------------------------

def add_reminder(conn: sqlite3.Connection, chat_id: int, text: str, due_at: str) -> int:
    """Save a pending reminder and return its id."""
    cur = conn.execute(
        "INSERT INTO reminders (chat_id, text, due_at, status, created_at) "
        "VALUES (?, ?, ?, 'pending', ?)",
        (chat_id, text, due_at, _now()),
    )
    conn.commit()
    return int(cur.lastrowid)


def pending_reminders(conn: sqlite3.Connection, chat_id: int) -> list[sqlite3.Row]:
    """Return every reminder still waiting to fire for this chat, soonest first."""
    return conn.execute(
        "SELECT * FROM reminders WHERE chat_id = ? AND status = 'pending' ORDER BY due_at",
        (chat_id,),
    ).fetchall()


def all_pending_reminders(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return every pending reminder across all chats — used to rehydrate jobs on startup."""
    return conn.execute(
        "SELECT * FROM reminders WHERE status = 'pending' ORDER BY due_at"
    ).fetchall()


def mark_fired(conn: sqlite3.Connection, reminder_id: int) -> None:
    """Mark a reminder as delivered so it is never rehydrated again."""
    conn.execute("UPDATE reminders SET status='fired' WHERE id=?", (reminder_id,))
    conn.commit()


def cancel_reminder(conn: sqlite3.Connection, chat_id: int, reminder_id: int) -> bool:
    """Cancel a pending reminder that belongs to this chat. Returns True if one was cancelled."""
    cur = conn.execute(
        "UPDATE reminders SET status='cancelled' WHERE id=? AND chat_id=? AND status='pending'",
        (reminder_id, chat_id),
    )
    conn.commit()
    return cur.rowcount > 0
