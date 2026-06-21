"""Synthesizer (fan-in): merge all worker findings into one match-day briefing.

Runs once after every dispatched worker has reported. It dedupes findings (a
retry pass can produce two entries per agent), feeds the successful sections to
the heavy model, and writes the final `briefing`.

Inputs:  AnalystState (uses `query`, `team_name`, `next_match`, `opponent_name`,
         `findings`).
Outputs: partial state with `briefing`.
"""

from __future__ import annotations

from app.agents.llm import safe_analyse
from app.config import agent_model, heavy_model
from app.state import AnalystState, Finding, latest_findings

_SYSTEM = (
    "You are a World Cup match-day editor writing a PREVIEW of an upcoming "
    "fixture. Weave the section notes into a single cohesive briefing. Open with "
    "one punchy line naming the fixture (focus team vs opponent, with the date) "
    "that blends the hard numbers (form, standings) with the News & Storylines "
    "narrative — injuries and momentum should colour the read, not sit in a silo. "
    "Then keep each section's heading as a bold line followed by 2-3 sentences, "
    "framed toward how it shapes that next match. Do not invent facts beyond the "
    "notes; if the next opponent or a section is unknown, say so briefly."
)


def _assemble(findings: list[Finding]) -> str:
    """Lay out the deduped sections as labelled notes for the editor model."""
    blocks = []
    for finding in latest_findings(findings):
        status = "" if finding.ok else "  [DATA UNAVAILABLE]"
        blocks.append(f"## {finding.title}{status}\n{finding.content}")
    return "\n\n".join(blocks)


def _fallback(query: str, findings: list[Finding]) -> str:
    """Plain concatenation used when the editor LLM call itself fails."""
    head = f"Match-day briefing for: {query}\n"
    body = "\n\n".join(f"**{f.title}**\n{f.content}" for f in latest_findings(findings))
    return head + "\n" + body


async def synthesizer_node(state: AnalystState) -> dict:
    """Fan worker outputs back in and produce the final briefing text."""
    findings = state.get("findings") or []
    if not findings:
        return {"briefing": "No analysis was produced — every worker failed."}

    fixture = state.get("next_match") or "next fixture unknown"
    user = (f"User question: {state['query']}\n"
            f"Focus team: {state.get('team_name') or 'none'}\n"
            f"Next match: {fixture}\n"
            f"Opponent: {state.get('opponent_name') or 'unknown'}\n\n"
            f"Section notes:\n{_assemble(findings)}")

    # Prefer 70b prose; if its daily budget is spent, write on 8b; else concat.
    text, ok = await safe_analyse(heavy_model(), _SYSTEM, user)
    if not ok:
        text, ok = await safe_analyse(agent_model(), _SYSTEM, user)
    return {"briefing": text if ok else _fallback(state["query"], findings)}
