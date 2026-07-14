"""Wire the plan-execute nodes into a StateGraph.

Shape: START -> planner -> executor -> replanner, with the replanner looping back
to the executor until it declares the goal answered (or the step cap trips).

The graph is built per run around a Metrics object so every LLM/tool call made
during that run is counted for the side-by-side comparison.
"""

from functools import partial

from langgraph.graph import END, START, StateGraph

from src.nodes import executor_node, planner_node, replanner_node, should_end
from src.react_loop import Metrics
from src.state import PlanExecuteState


def build_graph(metrics: Metrics):
    """Construct and compile the plan-and-execute graph for one run."""
    graph = StateGraph(PlanExecuteState)

    # partial binds the shared metrics counter into each async node.
    graph.add_node("planner", partial(planner_node, metrics=metrics))
    graph.add_node("executor", partial(executor_node, metrics=metrics))
    graph.add_node("replanner", partial(replanner_node, metrics=metrics))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "replanner")

    # The replanner either sends the run back for another step or ends it.
    graph.add_conditional_edges(
        "replanner", should_end, {"executor": "executor", END: END}
    )

    return graph.compile()
