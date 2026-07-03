"""Wire the nodes into a StateGraph and compile it with a checkpointer.

The checkpointer is injected (not created here) so main.py can own the SqliteSaver
context manager. A checkpointer is REQUIRED for interrupt() to work — without one
the graph has nowhere to freeze its state.
"""

from langgraph.graph import END, START, StateGraph

from src.nodes import (
    approval_gate_node,
    cancel_node,
    execute_node,
    planner_node,
    route_after_approval,
    route_after_validation,
    validator_node,
)
from src.state import ApprovalState


def build_graph(checkpointer):
    """Construct and compile the human-in-the-loop approval graph."""
    g = StateGraph(ApprovalState)

    g.add_node("planner", planner_node)
    g.add_node("validator", validator_node)
    g.add_node("approval", approval_gate_node)
    g.add_node("execute", execute_node)
    g.add_node("cancel", cancel_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "validator")

    # Validator decides: send to the human, re-draft, or give up.
    g.add_conditional_edges(
        "validator",
        route_after_validation,
        {"approval": "approval", "planner": "planner", "cancel": "cancel"},
    )
    # Human decision routes the run after the interrupt resumes.
    g.add_conditional_edges(
        "approval",
        route_after_approval,
        {"execute": "execute", "planner": "planner", "cancel": "cancel"},
    )

    g.add_edge("execute", END)
    g.add_edge("cancel", END)

    return g.compile(checkpointer=checkpointer)
