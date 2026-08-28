"""Replay cache for model responses, and the local daily-token ledger.

Inputs:  a cache key built from everything that can change an answer
Outputs: cached completions (with their usage), and today's token/request spend

Two things live here because both are sqlite and both exist for the same
reason: the free tier's daily ceiling.

The cache stores the RAW completion and its usage object, never a grade. A
grader bug then costs zero requests to fix, which matters because a re-run
would otherwise burn the day's RPD as well as its TPD.

The ledger exists because Groq publishes no remaining-TPD header -
x-ratelimit-remaining-tokens is per-minute and x-ratelimit-limit-requests is
per-day requests - so daily spend can only be tracked locally.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from src.config import CACHE_PATH, LEDGER_PATH

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    k       TEXT PRIMARY KEY,
    payload TEXT NOT NULL
)
"""

_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
    day      TEXT NOT NULL,
    model    TEXT NOT NULL,
    tokens   INTEGER NOT NULL DEFAULT 0,
    requests INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, model)
)
"""


def _connect(path: Path, schema: str) -> sqlite3.Connection:
    """Open a database, creating its table on first use."""
    conn = sqlite3.connect(path)
    conn.execute(schema)
    return conn


def key(model: str, payload: Any) -> str:
    """Hash everything that can change an answer into one stable key.

    The caller passes the full tuple - arm, probe, model, temperature, token
    budgets, reasoning effort, prompt and assembler versions, tokenizer, and
    the scenario and summary-fixture hashes. Leaving any of them out means an
    edit somewhere else silently returns a stale answer that still looks fresh.
    """
    blob = json.dumps([model, payload], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get(k: str) -> dict[str, Any] | None:
    """Return a cached completion, marked as replayed.

    `cached_tokens` is forced to None on the way out, not on the way in. The
    original call's cache-hit figure describes a request that is not being made
    now; reporting it again would fabricate cache statistics for a replay.
    """
    with _connect(CACHE_PATH, _CACHE_SCHEMA) as conn:
        row = conn.execute("SELECT payload FROM responses WHERE k = ?", (k,)).fetchone()
    if row is None:
        return None
    data: dict[str, Any] = json.loads(row[0])
    data["replayed"] = True
    usage = data.get("usage")
    if isinstance(usage, dict):
        usage["cached_tokens"] = None
    return data


def put(k: str, value: Mapping[str, Any]) -> None:
    """Store one raw completion and its usage."""
    with _connect(CACHE_PATH, _CACHE_SCHEMA) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO responses (k, payload) VALUES (?, ?)",
            (k, json.dumps(dict(value), ensure_ascii=False)),
        )


def rows() -> int:
    """How many responses the cache holds."""
    with _connect(CACHE_PATH, _CACHE_SCHEMA) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])


# --- The daily ledger --------------------------------------------------------


def record_spend(model: str, tokens: int, requests: int = 1) -> None:
    """Add one call's uncached tokens to today's running total."""
    with _connect(LEDGER_PATH, _LEDGER_SCHEMA) as conn:
        conn.execute(
            "INSERT INTO spend (day, model, tokens, requests) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(day, model) DO UPDATE SET "
            "tokens = tokens + excluded.tokens, requests = requests + excluded.requests",
            (date.today().isoformat(), model, tokens, requests),
        )


def spent_today() -> tuple[int, int]:
    """Return (tokens, requests) spent today across every model.

    Across every model on purpose: the daily ceiling is account-wide, so a
    per-model figure would report headroom that does not exist. Models named
    with a leading double underscore are excluded, so this module's own smoke
    test cannot inflate the figure the pre-flight checks against.
    """
    with _connect(LEDGER_PATH, _LEDGER_SCHEMA) as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(tokens), 0), COALESCE(SUM(requests), 0) "
            "FROM spend WHERE day = ? AND substr(model, 1, 2) <> '__'",
            (date.today().isoformat(),),
        ).fetchone()
    return int(row[0]), int(row[1])


if __name__ == "__main__":
    k1 = key("openai/gpt-oss-120b", {"arm": "full", "probe": "f07", "v": "v1"})
    k2 = key("openai/gpt-oss-120b", {"probe": "f07", "arm": "full", "v": "v1"})
    k3 = key("openai/gpt-oss-120b", {"arm": "full", "probe": "f07", "v": "v2"})
    assert k1 == k2, "key must not depend on dict ordering"
    assert k1 != k3, "a version bump must invalidate the entry"

    assert get(k1 + "-miss") is None, "a miss returns None, not a crash"

    put(k1, {
        "text": "Answer: 4,712",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 3000, "completion_tokens": 9, "cached_tokens": 2800},
    })
    hit = get(k1)
    assert hit is not None and hit["text"] == "Answer: 4,712"
    assert hit["replayed"] is True
    assert hit["usage"]["cached_tokens"] is None, (
        "a replayed row must report n/a, never a stale cache figure"
    )
    print(f"round trip ok; cache holds {rows()} rows")

    before_tok, before_req = spent_today()
    record_spend("__smoke__", 1234)
    after_tok, after_req = spent_today()
    assert (after_tok, after_req) == (before_tok, before_req), (
        "a smoke-test write must not move the figure the pre-flight trusts"
    )
    with _connect(LEDGER_PATH, _LEDGER_SCHEMA) as conn:
        wrote = conn.execute(
            "SELECT tokens FROM spend WHERE day = ? AND model = '__smoke__'",
            (date.today().isoformat(),),
        ).fetchone()
    assert wrote and wrote[0] >= 1234, "but it must still have been written"
    print(f"ledger: {after_tok:,} real tok / {after_req} requests today "
          f"(smoke rows excluded)")
