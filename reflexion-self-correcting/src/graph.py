"""Wire the generator -> critic loop into a cyclical LangGraph StateGraph.

The shape is a cycle with one exit. START goes to the generator; the critic
grades and conditionally routes either back to the generator (another revision)
or forward to the adjudicator, which gives the final verdict and ends the run.

Inputs:  none.
Outputs: a compiled graph ready for .invoke()/.stream().
"""

from langgraph.graph import END, START, StateGraph

from src.nodes import (
    adjudicator_node,
    critic_node,
    generator_node,
    route_after_critic,
)
from src.state import ReflexionState


def build_graph():
    """Construct and compile the Reflexion self-correction graph."""
    graph = StateGraph(ReflexionState)

    graph.add_node("generator", generator_node)
    graph.add_node("critic", critic_node)
    graph.add_node("adjudicator", adjudicator_node)

    graph.add_edge(START, "generator")
    graph.add_edge("generator", "critic")

    # The critic either sends the draft back for revision or ends the loop by
    # handing off to the one-shot adjudicator.
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"generator": "generator", "adjudicator": "adjudicator"},
    )

    graph.add_edge("adjudicator", END)

    return graph.compile()
