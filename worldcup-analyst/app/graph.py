"""Build and compile the LangGraph: supervisor -> parallel workers -> synthesizer.

Topology:
    START -> supervisor --(Send fan-out)--> [3 tool-using agents in parallel] -> synthesizer
    synthesizer --(conditional)--> supervisor (retry once) | END

Each agent is a real ReAct tool loop (see app/agents/runner.py). `Send` from
langgraph.types dispatches them concurrently in one superstep; they all feed the
single synthesizer node, which LangGraph runs only after every dispatched agent
has finished. The retry loop fires only for transient failures (timeout / rate
limit), not permanent ones (bad token, data not published).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from app.agents.matchup import matchup_node
from app.agents.news import news_node
from app.agents.player import player_node
from app.agents.supervisor import ALL_JOBS, supervisor_node
from app.agents.synthesizer import synthesizer_node
from app.state import AnalystState, latest_findings

_WORKERS = {
    "matchup_agent": matchup_node,
    "player_agent": player_node,
    "news_agent": news_node,
}


def _fan_out(state: AnalystState) -> list[Send]:
    """Dispatch one Send per requested job — this is the parallel fan-out."""
    return [Send(job, state) for job in state.get("jobs", []) if job in _WORKERS]


def _after_synthesis(state: AnalystState) -> str:
    """Loop back to the supervisor once for a TRANSIENT failure, else END.

    Permanent failures (bad token, team not found, data not published) are not
    retried — a second identical call would just waste latency and API quota.
    """
    findings = latest_findings(state.get("findings") or [])
    transient_failure = any(not f.ok and f.transient for f in findings)
    if transient_failure and state.get("retries", 0) < 1:
        return "supervisor"
    return END


def build_graph() -> CompiledStateGraph:
    """Wire the nodes and edges and return the compiled, runnable graph."""
    graph = StateGraph(AnalystState)

    graph.add_node("supervisor", supervisor_node)
    for name, node in _WORKERS.items():
        graph.add_node(name, node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "supervisor")
    # Conditional fan-out: the third arg lists every possible Send target.
    graph.add_conditional_edges("supervisor", _fan_out, list(ALL_JOBS))
    for name in _WORKERS:
        graph.add_edge(name, "synthesizer")
    graph.add_conditional_edges(
        "synthesizer", _after_synthesis, ["supervisor", END]
    )
    return graph.compile()
