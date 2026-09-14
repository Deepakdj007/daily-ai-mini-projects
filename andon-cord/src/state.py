"""The shapes that move through the graph.

Inputs:  none
Outputs: Plan, Finding (what the model returns); ResearchState, ResearchTask
         (what the graph carries).

Strict-mode rule for every pydantic model here: every field is required, with no
defaults, no Optional, and no numeric constraints. Groq's constrained decoding
rejects the `default`, `minimum` and `maximum` keywords those produce. So where
a field is sometimes empty, say that in the description and let the model send
an empty string.
"""

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """The research lead's reading of the brief, plus its one chance to ask."""

    interpretation: str = Field(
        description="one sentence stating how you are reading this brief, "
        "including the assumption you had to make if anything was unclear"
    )
    sub_questions: list[str] = Field(
        description="research questions, each answerable by one web search, "
        "each covering a different angle"
    )
    needs_human: bool = Field(
        description="true ONLY when the brief is ambiguous about something "
        "private to this client - which of their products, which audience, "
        "which internal system - and guessing wrong would send the whole team "
        "down the wrong path. If a web search could settle it, this is false"
    )
    question: str = Field(
        description="the single question to ask the human. Empty string when "
        "needs_human is false"
    )


class Finding(BaseModel):
    """One researcher's answer to one sub-question."""

    summary: str = Field(
        description="3 to 5 sentences answering the question from the search "
        "results, citing each claim as [1], [2] and so on"
    )
    needs_human: bool = Field(
        description="true ONLY when the missing fact is private to this "
        "client - their own number, their own system, their own choice - and "
        "exists nowhere on the public web. Thin or unhelpful search results "
        "are NOT a reason: set this false and say so in the summary instead"
    )
    question: str = Field(
        description="the single question to ask the human. Empty string when "
        "needs_human is false"
    )


class ResearchTask(TypedDict):
    """What one researcher receives when the lead dispatches it.

    This is the researcher node's whole input. A node reached by Send does not
    see the shared state — it sees this dict and nothing else, which is why the
    brief and the interpretation are copied in here rather than read from state.
    """

    index: int
    question: str
    brief: str
    interpretation: str
    clarifications: list[dict]
    ask_policy: str


class ResearchState(TypedDict, total=False):
    """The shared state. total=False so every node can return a partial dict.

    Two fields carry an `operator.add` reducer because two researchers write to
    them in the same superstep. Without a reducer, LangGraph treats concurrent
    writes to one key as a conflict and raises InvalidUpdateError.

    A finding dict looks like:
        {"index": 1, "question": str, "summary": str, "needs_human": bool,
         "ask": str, "sources": [{"title": str, "url": str, "snippet": str}]}

    A clarification dict looks like:
        {"id": "lead" | "r1", "asked_by": str, "text": str, "answer": str}
    """

    brief: str
    ask_policy: str
    interpretation: str
    sub_questions: list[str]
    needs_human: bool
    question: str
    findings: Annotated[list[dict], operator.add]
    clarifications: Annotated[list[dict], operator.add]
    report: str
