# state.py
# ─────────────────────────────────────────────────────────────
# State schema definitions for the parallel news analyst graph.
# Defines two state types:
#   - TopicState: the isolated state for each parallel agent
#   - OverallState: the shared graph state with reducer for merging results
# ─────────────────────────────────────────────────────────────

import operator
from typing import Annotated, TypedDict

class TopicState(TypedDict):
    """
    Isolated state for a single news analyst agent.

    Each parallel agent gets its own TopicState. It doesn't see
    the other agents' states. This isolation is what makes
    parallel execution safe — no shared mutable data.

    Fields:
        topic: The news topic this agent is researching.
    """
    topic: str

class OverallState(TypedDict):
    """
    Shared state for the entire graph, including the reduce phase.

    The `briefs` field uses Annotated with operator.add as a reducer.
    This is REQUIRED for parallel nodes. Without it, LangGraph raises
    InvalidUpdateError when multiple agents try to write to the same key.

    The reducer tells LangGraph: 'when parallel agents all return a list
    for this key, concatenate the lists together instead of overwriting.'

    Fields:
        topics: Input list of topics to research.
        briefs: Accumulated list of news briefs, one per topic.
                The Annotated reducer makes this safe for parallel writes.
        report: Final assembled report string, set by report_assembler.
    """
    topics: list[str]
    briefs: Annotated[list[str], operator.add]
    report: str

