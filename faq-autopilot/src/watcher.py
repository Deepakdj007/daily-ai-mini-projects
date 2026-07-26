"""Drift detector: scan the docs folder and classify what changed since last time.

Inputs:  the docs directory and the {path: sha256} map of what the agent last saw.
Outputs: a list of DocEvent(kind, rel_path, content) for added / modified / removed docs.

Change detection is a plain content hash, not a timestamp — a doc that is touched but
not edited produces no event, so the agent never does pointless LLM work.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocEvent:
    """One detected change to a source doc.

    kind:     'added', 'modified', or 'removed'.
    rel_path: path relative to the docs dir, e.g. 'pricing.md' (used as the citation key).
    content:  the doc's current text ('' for a removed doc).
    """

    kind: str
    rel_path: str
    content: str


def _sha256(text: str) -> str:
    """Return the hex SHA-256 of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan(docs_dir: Path, known_hashes: dict[str, str]) -> list[DocEvent]:
    """Compare the docs on disk to what we last saw and return the drift events.

    Args:
      docs_dir:     folder of source markdown files to watch.
      known_hashes: {rel_path: sha256} recorded from the previous scan.
    Returns:
      One DocEvent per doc that was added, modified, or removed. Unchanged docs
      produce nothing.
    """
    events: list[DocEvent] = []
    seen: set[str] = set()

    for path in sorted(docs_dir.glob("*.md")):
        rel = path.name
        seen.add(rel)
        text = path.read_text(encoding="utf-8")
        digest = _sha256(text)
        if rel not in known_hashes:
            events.append(DocEvent("added", rel, text))
        elif known_hashes[rel] != digest:
            events.append(DocEvent("modified", rel, text))

    for rel in known_hashes:
        if rel not in seen:
            events.append(DocEvent("removed", rel, ""))

    return events


def current_hash(text: str) -> str:
    """Public helper so callers can store the hash of content they just processed."""
    return _sha256(text)
