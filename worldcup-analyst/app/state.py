"""Shared graph state and the reducers that make parallel writes safe.

The worker agents run concurrently and each appends one `Finding`. A plain
TypedDict key would clobber on concurrent writes, so `findings` and `missing`
use `Annotated[list, operator.add]` reducers — LangGraph merges every worker's
contribution instead of keeping only the last.

The supervisor resolves the focus team's id and next fixture once and shares them
through the read-only context fields (`team_id`, `opponent_*`, `next_match`) so
workers don't each re-resolve the team.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict


@dataclass
class Finding:
    """One worker's result: what it produced and whether it had real data."""

    agent: str            # node name, e.g. "standings_agent"
    title: str            # short heading for the briefing section
    content: str          # the worker's analytical write-up (LLM-generated)
    ok: bool              # False when the underlying data call failed/was empty
    transient: bool = False  # True when a failed finding is worth a retry


class AnalystState(TypedDict, total=False):
    """State threaded through the graph from supervisor to synthesizer."""

    query: str                                  # the user's question
    team_name: str | None                       # focus team resolved by router
    team_id: int | None                         # resolved once by the supervisor
    opponent_name: str | None                   # next-match opponent label
    opponent_id: int | None                     # next-match opponent team id
    next_match: str | None                      # human fixture line (vs / date / venue)
    jobs: list[str]                             # worker node names to dispatch
    findings: Annotated[list[Finding], operator.add]  # workers append here
    missing: Annotated[list[str], operator.add]       # agents reporting no data
    retries: int                                # fan-out retry counter (cap 1)
    briefing: str | None                        # final synthesized briefing


def latest_findings(findings: list[Finding]) -> list[Finding]:
    """Keep only the most recent Finding per agent (dedupes after a retry pass)."""
    by_agent: dict[str, Finding] = {}
    for finding in findings:
        by_agent[finding.agent] = finding  # later entry wins
    return list(by_agent.values())
