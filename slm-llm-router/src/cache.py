"""On-disk response cache.

Inputs:  a call signature (tier, model, prompt version, query, sample index).
Outputs: the stored call result, or None.

Two jobs. First, it makes re-runs free, which matters because Groq's free tier
counts 200,000 tokens per day across your whole account - one uncached
benchmark loop can lock you out of every other project until tomorrow. Second,
it makes the experiment reproducible: every strategy replays the same stored
answers, so differences between strategies come from routing decisions and not
from re-sampling the model.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from src.config import CACHE_PATH, PROMPT_VERSION

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    key     TEXT PRIMARY KEY,
    payload TEXT NOT NULL
)
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(_SCHEMA)
    return conn


def make_key(tier: str, model: str, query: str, sample: int = 0) -> str:
    """Build the cache key.

    PROMPT_VERSION is part of the key on purpose. Prompts change during a
    build, and a stale cached answer generated under an older system prompt
    does not announce itself - it just quietly shifts a score and looks like a
    real effect. Bumping the version string retires every old entry at once.
    """
    raw = f"{tier}|{model}|{PROMPT_VERSION}|{sample}|{query}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(key: str) -> dict[str, Any] | None:
    """Return a cached call result, or None on a miss."""
    with _connect() as conn:
        row = conn.execute("SELECT payload FROM calls WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def put(key: str, payload: dict[str, Any]) -> None:
    """Store a call result."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO calls (key, payload) VALUES (?, ?)",
            (key, json.dumps(payload)),
        )


def stats() -> dict[str, int]:
    """Row count, for the run summary."""
    with _connect() as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM calls").fetchone()
    return {"entries": count}


if __name__ == "__main__":
    key = make_key("test", "demo-model", "what is 2+2")
    put(key, {"answer": "4", "in_tok": 10, "out_tok": 2})
    print(f"key      : {key[:16]}...")
    print(f"round trip: {get(key)}")
    print(f"miss      : {get('nonexistent-key')}")
    print(f"stats     : {stats()}")
