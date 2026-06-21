"""Wire the supervisor and specialists into a LangGraph StateGraph.

The shape is a star: START -> supervisor, the supervisor conditionally routes to
one specialist, and every specialist hands control straight back to the
supervisor. The supervisor is the only node that can route to END.

Inputs:  none.
Outputs: a compiled graph ready for .invoke().
"""

from langgraph.graph import END, START, StateGraph

from src.agents import editor_node, researcher_node, seo_node, writer_node
from src.state import ContentState
from src.supervisor import route_decision, supervisor_node

# The specialist node names, kept in one place so edges stay in sync.
WORKERS = ["researcher", "writer", "editor", "seo"]


def build_graph():
    """Construct and compile the supervisor content-team graph."""
    graph = StateGraph(ContentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("editor", editor_node)
    graph.add_node("seo", seo_node)

    graph.add_edge(START, "supervisor")

    # The supervisor's decision string maps to a worker node or to END.
    routing = {name: name for name in WORKERS}
    routing["FINISH"] = END
    graph.add_conditional_edges("supervisor", route_decision, routing)

    # Every specialist returns control to the supervisor for the next decision.
    for name in WORKERS:
        graph.add_edge(name, "supervisor")

    return graph.compile()
