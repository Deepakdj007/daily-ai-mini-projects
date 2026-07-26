"""Render the database into the two human-facing artifacts: FAQ.md and CHANGELOG.md.

Inputs:  a sqlite3 connection and the output paths.
Outputs: FAQ.md (grouped by source doc, every answer carrying a citation) and
         CHANGELOG.md (the newest-first audit trail of what the agent did and why).

The DB is the source of truth; these files are always regenerated from it.
"""

import sqlite3
from pathlib import Path

from src.state import active_faqs, recent_changelog


def render_faq(conn: sqlite3.Connection, faq_path: Path) -> None:
    """Write FAQ.md from the active FAQ rows, grouped by source document."""
    rows = active_faqs(conn)
    lines: list[str] = [
        "# FAQ",
        "",
        "_Maintained automatically by FAQ Autopilot. Every answer is grounded in a "
        "source document and updated whenever that document changes._",
        "",
    ]

    current_source = None
    for row in rows:
        if row["source_file"] != current_source:
            current_source = row["source_file"]
            lines.append(f"## From `{current_source}`")
            lines.append("")
        lines.append(f"**Q: {row['question']}**")
        lines.append("")
        lines.append(row["answer"])
        lines.append("")
        lines.append(f"> Source: `{row['source_file']}` → {row['source_anchor']}")
        lines.append("")

    if not rows:
        lines.append("_No FAQ entries yet._")
        lines.append("")

    faq_path.write_text("\n".join(lines), encoding="utf-8")


def render_changelog(conn: sqlite3.Connection, changelog_path: Path) -> None:
    """Write CHANGELOG.md — the audit trail of every autonomous action, newest first."""
    rows = recent_changelog(conn)
    lines: list[str] = [
        "# Changelog — FAQ Autopilot",
        "",
        "_Every line is an action the agent took on its own, with the reason it fired._",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['ts']}` **{row['event']}** — {row['detail']}")

    if not rows:
        lines.append("_No actions recorded yet._")

    lines.append("")
    changelog_path.write_text("\n".join(lines), encoding="utf-8")
