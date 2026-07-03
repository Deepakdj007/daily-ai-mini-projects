"""Append-only audit trail.

Every node records what happened to output/audit_log.jsonl, one JSON object per
line. Append-only by design: we never rewrite earlier lines, so the file reads as a
tamper-evident chronological record of who approved which irreversible action and when.
This is the compliance artifact a reviewer in finance/healthcare actually wants.
"""

import json
from datetime import datetime, timezone

from src.config import AUDIT_LOG, OUTPUT_DIR


def record_event(thread_id: str, event: str, actor: str, details: dict) -> None:
    """Append one event to the audit log.

    thread_id: the run this event belongs to.
    event:     short label, e.g. "drafted", "validated", "executed".
    actor:     "agent" for machine steps, "human" for the approval decision.
    details:   any extra context (draft summary, errors, decision feedback).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "event": event,
        "actor": actor,
        "details": details,
    }
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
