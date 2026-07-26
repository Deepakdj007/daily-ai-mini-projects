"""Every event and data model that moves through the workflow.

LlamaIndex Workflows routes work by event type: a step declares which event it
accepts, and the engine delivers matching events to it. So these classes are the
wiring diagram of the whole app.

All of them live at module level on purpose. Workflows resolves step type hints
at import time, and events defined inside a function or behind
`from __future__ import annotations` fail to resolve.

Inputs:  none (pure declarations).
Outputs: Event subclasses consumed by src/workflow.py, pydantic models used for
         structured LLM output and for the final report.
"""

from pydantic import BaseModel, Field
from workflows.events import Event


# --- Data models. The first two are what the LLM is asked to return; the last
# two are plain containers we build ourselves from search results.
class Plan(BaseModel):
    """The planner's decomposition of the user's topic."""

    sub_questions: list[str] = Field(
        description="Focused, independently searchable research questions."
    )


class Critique(BaseModel):
    """The reflector's verdict on whether the findings answer the topic."""

    covered: bool = Field(
        description="True if the findings together answer the topic well enough to write."
    )
    reasoning: str = Field(description="One or two sentences justifying the verdict.")
    gaps: list[str] = Field(
        default_factory=list,
        description="New sub-questions that would close the gaps. Empty when covered is True.",
    )


class Source(BaseModel):
    """One web result behind a finding."""

    title: str
    url: str
    snippet: str


class Finding(BaseModel):
    """What a single researcher learned about a single sub-question."""

    question: str
    summary: str
    sources: list[Source]
    round: int


# --- Events. One class per hop in the graph.
class SubQuestionEvent(Event):
    """A research assignment. The planner and the reflector both emit these."""

    question: str
    index: int
    round: int


class FindingEvent(Event):
    """A completed piece of research, sent back for collection."""

    finding: Finding


class ReflectEvent(Event):
    """Fires once per round, after every finding for that round has arrived."""

    round: int


class WriteEvent(Event):
    """Signals that research is done and the report can be written."""


class ProgressEvent(Event):
    """A human-readable status line. Never routed to a step -- it goes straight
    to the event stream so the CLI can show the team working in real time."""

    agent: str
    msg: str
    style: str = "cyan"
