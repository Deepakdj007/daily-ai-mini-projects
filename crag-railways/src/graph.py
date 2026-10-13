"""The CRAG graph: retrieve, grade, then refine the KB, search the web, or both.

Inputs:  a question
Outputs: the final state - labels, verdict, the knowledge used and a cited answer
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from src.evaluator import decide, grade
from src.kb import Doc, retrieve
from src.llm import ask_text
from src.refine import refine
from src.web import rewrite_query, web_search

ANSWER_SYSTEM = """You are a railway help desk assistant.
Answer only from the knowledge given. Each line ends with its source tag, such as [KB:tdr] or [WEB:https://...].
Cite the tags you used at the end of the sentence that uses them, copied exactly in square brackets.
If the knowledge does not answer the question, say you don't know. Keep it under 100 words."""


class CragState(TypedDict, total=False):
    """Everything the graph knows about one question."""

    question: str
    docs: list[Doc]
    labels: dict[str, str]
    verdict: str
    # refine and web can both write here in the same step, so their lists are added together.
    knowledge: Annotated[list[str], operator.add]
    web_query: str
    answer: str


async def answer(question: str, knowledge: list[str]) -> str:
    """Write the final answer from whatever knowledge made it through."""
    lines = "\n".join(knowledge) or "(no knowledge)"
    return await ask_text(ANSWER_SYSTEM, f"Knowledge:\n{lines}\n\nQuestion: {question}")


async def retrieve_node(state: CragState) -> dict:
    """Fetch the closest KB passages."""
    return {"docs": retrieve(state["question"])}


async def evaluate_node(state: CragState) -> dict:
    """Grade the passages and pick a CRAG action."""
    labels = await grade(state["question"], state["docs"])
    return {"labels": labels, "verdict": decide(labels)}


async def refine_node(state: CragState) -> dict:
    """Keep only the KB sentences that help."""
    return {"knowledge": await refine(state["question"], state["docs"])}


async def web_node(state: CragState) -> dict:
    """Rewrite the question for a search engine and fetch web snippets."""
    query = await rewrite_query(state["question"])
    return {"web_query": query, "knowledge": await web_search(query)}


async def generate_node(state: CragState) -> dict:
    """Answer from the corrected knowledge."""
    return {"answer": await answer(state["question"], state.get("knowledge", []))}


def route(state: CragState) -> list[str]:
    """The corrective step: the verdict decides where the knowledge comes from."""
    return {
        "CORRECT": ["refine"],
        "INCORRECT": ["web"],
        "AMBIGUOUS": ["refine", "web"],
    }[state["verdict"]]


def build_graph():
    """Wire the nodes. The only branching is the conditional edge after evaluate."""
    graph = StateGraph(CragState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("refine", refine_node)
    graph.add_node("web", web_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_conditional_edges("evaluate", route, ["refine", "web"])
    graph.add_edge("refine", "generate")
    graph.add_edge("web", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
