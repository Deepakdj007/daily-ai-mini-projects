# graph.py
# ─────────────────────────────────────────────────────────────
# Graph construction for the parallel news analyst.
# Wires state, nodes, and routing into a compiled LangGraph.
# Key pattern: topic_dispatcher returns List[Send] to launch
# parallel agents — this is the map step of map-reduce.
# ─────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from state import OverallState
from agents import news_analyst, report_assembler

def topic_dispatcher(state: OverallState) -> list[Send]:
    """
    Routing function that fans out parallel work using the Send API.

    This is NOT a node — it's a conditional edge routing function.
    LangGraph calls it after START and uses its return value to determine
    what to execute next. By returning a list of Send objects, we tell
    LangGraph to launch one news_analyst instance per topic, all in
    the same superstep (simultaneously).

    Each Send object has two arguments:
    - The target node name as a string: "news_analyst"
    - The state to pass to that node instance: {"topic": topic_string}

    Args:
        state: OverallState containing the topics list.

    Returns:
        List of Send objects, one per topic. LangGraph launches them all
        in parallel in the next superstep.
    """
    return [
        Send("news_analyst", {"topic": topic})
        for topic in state["topics"]
    ]

def build_graph():
    """
    Construct and compile the parallel news analyst graph.

    Graph structure:
    START → topic_dispatcher (routing) → news_analyst (×N, parallel)
                                       ↓ (all complete)
                                  report_assembler → END

    The conditional edge from START uses topic_dispatcher to fan out.
    Each news_analyst instance edges to report_assembler.
    LangGraph waits for all parallel instances before running
    report_assembler.

    Returns:
        Compiled LangGraph ready for .invoke() calls.
    """
    builder = StateGraph(OverallState)

    # Add the two worker nodes.
    # news_analyst runs N times in parallel; report_assembler runs once.
    builder.add_node("news_analyst", news_analyst)
    builder.add_node("report_assembler", report_assembler)

    # Connect START to the routing function.
    # add_conditional_edges reads the return value of topic_dispatcher
    # (a list of Send objects) and uses it to determine what runs next.
    builder.add_conditional_edges(START, topic_dispatcher)

    # Every parallel news_analyst instance, when it finishes,
    # edges to report_assembler.
    # LangGraph only executes report_assembler after ALL parallel
    # instances have completed — this is the fan-in behavior.
    builder.add_edge("news_analyst", "report_assembler")

    # report_assembler completes the graph.
    builder.add_edge("report_assembler", END)

    # Compile validates the graph structure (no orphaned nodes,
    # no missing edges) and returns the runnable object.
    return builder.compile()