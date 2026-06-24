"""LangGraph flow that wires the pipeline: question -> retrieve -> answer.

Two nodes, one edge between them. Keeping it in a graph makes the data flow
explicit and gives you a clean place to add steps later (reranking, multi-hop,
guardrails) without rewriting the call sites.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.answerer import answer as gemini_answer
from src.embedder import embed_query
from src.vector_store import search


class RAGState(TypedDict):
    """State passed between nodes as the query flows through the graph."""

    question: str
    retrieved: list[dict]
    answer: str


def retrieve_node(state: RAGState) -> dict:
    """Embed the question visually and pull the top matching pages from Qdrant."""
    query_vec = embed_query(state["question"])
    hits = search(query_vec)
    return {"retrieved": hits}


def answer_node(state: RAGState) -> dict:
    """Send the retrieved page images to Gemini and capture the answer."""
    text = gemini_answer(state["question"], state["retrieved"])
    return {"answer": text}


def build_graph():
    """Compile the retrieve -> answer graph."""
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)
    return builder.compile()
