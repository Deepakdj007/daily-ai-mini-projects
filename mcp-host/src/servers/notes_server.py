"""
MCP server exposing a persistent sqlite notebook.

Notes outlive the host process, which is what lets the assistant recall
something you told it in an earlier session. Standalone by design: it imports
nothing from src/, so the host can launch it with no PYTHONPATH.

Inputs: --db <path>, then MCP requests on stdin.
Outputs: save_note, search_notes, list_notes tools on stdout.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mcp.server import MCPServer

mcp = MCPServer("notes")

# Replaced in main() before the server starts serving.
DB_PATH: Path = Path("notes.db")

_COLUMNS = ("id", "text", "tag", "created_at")


def _connect() -> sqlite3.Connection:
    """Open the notes database, creating the table on first use."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS notes ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  text TEXT NOT NULL,"
        "  tag TEXT NOT NULL DEFAULT '',"
        "  created_at TEXT NOT NULL)"
    )
    return conn


def _fetch(sql: str, params: tuple) -> list[dict]:
    """Run a SELECT and return rows as dicts keyed by _COLUMNS."""
    conn = _connect()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(zip(_COLUMNS, row)) for row in rows]


@mcp.tool()
def save_note(text: str, tag: str = "") -> str:
    """Save a note so it can be recalled in a later session. Tag groups related notes."""
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = _connect()
    try:
        cursor = conn.execute(
            "INSERT INTO notes (text, tag, created_at) VALUES (?, ?, ?)",
            (text, tag, created_at),
        )
        conn.commit()
        note_id = cursor.lastrowid
    finally:
        conn.close()

    suffix = f" tagged {tag!r}" if tag else ""
    return f"saved note #{note_id}{suffix}"


@mcp.tool()
def search_notes(query: str, limit: int = 10) -> list[dict]:
    """Find saved notes whose text or tag contains the query, newest first."""
    like = f"%{query}%"
    return _fetch(
        "SELECT id, text, tag, created_at FROM notes "
        "WHERE text LIKE ? OR tag LIKE ? ORDER BY id DESC LIMIT ?",
        (like, like, limit),
    )


@mcp.tool()
def list_notes(limit: int = 20) -> list[dict]:
    """List the most recently saved notes, newest first."""
    return _fetch(
        "SELECT id, text, tag, created_at FROM notes ORDER BY id DESC LIMIT ?",
        (limit,),
    )


def main() -> None:
    """Parse --db, pin the database path, and serve MCP over stdio."""
    global DB_PATH
    parser = argparse.ArgumentParser(description="Persistent notes MCP server")
    parser.add_argument("--db", required=True, help="path to the sqlite database file")
    args = parser.parse_args()

    DB_PATH = Path(args.db).resolve()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
