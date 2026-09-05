"""Replay cache for model responses, plus the local spend ledger.

Inputs:  a cache key and a response payload
Outputs: replayed responses, and a per-(day, model) record of tokens spent

Two separate sqlite files with two different jobs.

The cache exists because the ablation runs the same probes through seven arms.
The extraction call depends only on the turn, never on the store, so every arm
after the first replays it for free. Without that the free tier's per-minute
limit becomes the experiment instead of the subject.

The ledger exists because Groq publishes no remaining-tokens-per-day header.
x-ratelimit-remaining-tokens is per minute. Daily headroom can only be tracked
by counting locally, and a run that starts without knowing it has 40k left is
a run that dies two thirds of the way through.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date
from typing import Any

from src import config

_CACHE_SCHEMA = "CREATE TABLE IF NOT EXISTS responses (k TEXT PRIMARY KEY, payload TEXT NOT NULL)"
_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
    day      TEXT NOT NULL,
    model    TEXT NOT NULL,
    tokens   INTEGER NOT NULL DEFAULT 0,
    requests INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, model)
)
"""


def _connect(path, schema: str) -> sqlite3.Connection:
    """Open a sqlite file and make sure its table exists."""
    conn = sqlite3.connect(str(path), timeout=config.SQLITE_TIMEOUT)
    conn.execute(schema)
    conn.commit()
    return conn


def key(model: str, payload: Any) -> str:
    """Hash everything that could change the answer.

    The caller passes an explicit dict rather than the rendered messages, so a
    forgotten field is a visible omission in one place instead of an invisible
    collision between two runs that were never comparable.
    """
    blob = json.dumps([model, payload], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def fingerprint(*parts: str) -> str:
    """Short hash of prompt text, to be folded into a cache key.

    A hand-maintained PROMPT_VERSION only retires the cache when somebody
    remembers to bump it. This was not hypothetical: an edit to the extraction
    prompt during the build replayed the old answer and looked like the edit had
    done nothing. Hashing the prompt itself makes forgetting impossible.
    """
    blob = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def get(cache_key: str) -> dict | None:
    """Return a stored response, or None. Marks the copy as a replay."""
    conn = _connect(config.CACHE_PATH, _CACHE_SCHEMA)
    try:
        row = conn.execute("SELECT payload FROM responses WHERE k = ?", (cache_key,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    data = json.loads(row[0])
    data["replayed"] = True
    # A replay spent nothing, so it cannot report a cache-hit token count.
    # Reporting the original call's number again would fabricate statistics.
    if isinstance(data.get("usage"), dict):
        data["usage"]["cached_tokens"] = None
    return data


def put(cache_key: str, payload: dict) -> None:
    """Store a response. Overwrites, so a re-run after a fix replaces cleanly."""
    conn = _connect(config.CACHE_PATH, _CACHE_SCHEMA)
    try:
        conn.execute(
            "INSERT INTO responses (k, payload) VALUES (?, ?) "
            "ON CONFLICT(k) DO UPDATE SET payload = excluded.payload",
            (cache_key, json.dumps(payload, ensure_ascii=False)),
        )
        conn.commit()
    finally:
        conn.close()


def rows() -> int:
    """How many responses are cached."""
    conn = _connect(config.CACHE_PATH, _CACHE_SCHEMA)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])
    finally:
        conn.close()


def record_spend(model: str, tokens: int, *, day: str = "") -> None:
    """Add one request's tokens to today's ledger for this model."""
    today = day or date.today().isoformat()
    conn = _connect(config.LEDGER_PATH, _LEDGER_SCHEMA)
    try:
        conn.execute(
            "INSERT INTO spend (day, model, tokens, requests) VALUES (?, ?, ?, 1) "
            "ON CONFLICT(day, model) DO UPDATE SET "
            "tokens = tokens + excluded.tokens, requests = requests + 1",
            (today, model, int(tokens)),
        )
        conn.commit()
    finally:
        conn.close()


def spent_today(model: str = "", *, day: str = "") -> tuple[int, int]:
    """(tokens, requests) spent today, for one model or all of them.

    Models whose name starts with '__' are excluded: a smoke test must not be
    able to inflate the figure a real run's pre-flight check depends on.
    """
    today = day or date.today().isoformat()
    conn = _connect(config.LEDGER_PATH, _LEDGER_SCHEMA)
    try:
        if model:
            sql = "SELECT COALESCE(SUM(tokens),0), COALESCE(SUM(requests),0) FROM spend WHERE day=? AND model=?"
            row = conn.execute(sql, (today, model)).fetchone()
        else:
            sql = ("SELECT COALESCE(SUM(tokens),0), COALESCE(SUM(requests),0) FROM spend "
                   "WHERE day=? AND model NOT LIKE '\\_\\_%' ESCAPE '\\'")
            row = conn.execute(sql, (today,)).fetchone()
    finally:
        conn.close()
    return int(row[0]), int(row[1])


def ledger_rows(*, day: str = "") -> list[dict]:
    """Today's ledger, for the results manifest."""
    today = day or date.today().isoformat()
    conn = _connect(config.LEDGER_PATH, _LEDGER_SCHEMA)
    try:
        cur = conn.execute(
            "SELECT day, model, tokens, requests FROM spend WHERE day = ? ORDER BY model", (today,)
        )
        return [{"day": r[0], "model": r[1], "tokens": r[2], "requests": r[3]} for r in cur.fetchall()]
    finally:
        conn.close()


if __name__ == "__main__":
    k1 = key("m", {"b": 2, "a": 1})
    assert k1 == key("m", {"a": 1, "b": 2}), "key must not depend on dict order"
    assert k1 != key("m", {"a": 1, "b": 3}), "a changed field must change the key"
    assert k1 != key("other", {"a": 1, "b": 2}), "the model is part of the key"

    probe = key("__smoke", {"cache": "v1"})
    put(probe, {"text": "ok", "usage": {"prompt_tokens": 5, "cached_tokens": 11}})
    hit = get(probe)
    assert hit and hit["text"] == "ok" and hit["replayed"] is True
    assert hit["usage"]["cached_tokens"] is None, "a replay must not report cached tokens"
    assert get(key("__smoke", {"cache": "absent"})) is None

    before_all, _ = spent_today()
    record_spend("__smoke", 1234)
    smoke_tokens, smoke_reqs = spent_today("__smoke")
    after_all, _ = spent_today()
    assert smoke_tokens >= 1234 and smoke_reqs >= 1
    assert after_all == before_all, "a __smoke model must not move the real total"
    print(f"OK - {rows()} cached responses; real spend today {after_all:,} tokens")
