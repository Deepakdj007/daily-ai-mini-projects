"""Shared state for the Reflexion loop and the critic's structured verdict.

Two shapes live here. `Critique` is what the critic LLM must return — a scored
rubric plus actionable feedback — enforced via structured output. `ReflexionState`
is the graph's running memory: the current draft, the latest feedback, and the
score history that the diminishing-returns plot is built from.

Inputs:  none.
Outputs: Critique (LLM schema) + ReflexionState (graph state) used by the nodes.
"""

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class Critique(BaseModel):
    """The critic's grade of one draft, returned via structured output.

    The sub-scores force the critic to grade each rubric dimension separately
    instead of guessing one vague number, which makes the feedback specific.
    """

    hook: int = Field(description="Opening line grabs attention (0-10)")
    specificity: int = Field(description="Concrete, personalized, not generic (0-10)")
    clarity: int = Field(description="Easy to read, one clear idea (0-10)")
    cta: int = Field(description="Single, low-friction call to action (0-10)")
    brevity: int = Field(description="Tight; no filler, under ~120 words (0-10)")
    score: int = Field(description="Overall quality, holistic (0-10)")
    passed: bool = Field(description="True if the email is send-ready as-is")
    feedback: str = Field(
        description="The single most important fix for the next revision. "
        "Concrete and actionable. Empty string if passed."
    )


class ReflexionState(TypedDict, total=False):
    """Running state threaded through the generator -> critic loop.

    `scores` uses an additive reducer so each critic pass appends one entry;
    that ordered list is exactly the y-axis of the diminishing-returns curve.
    """

    task: str  # the brief the generator must satisfy
    draft: str  # current email draft
    feedback: str  # critic's fix to apply on the next revision
    score: int  # latest overall score
    scores: Annotated[list[int], operator.add]  # one score per revision -> plot
    revision: int  # how many drafts produced so far
    passed: bool  # critic's latest pass/fail
    final_verdict: str  # adjudicator's one-shot closing verdict
    verdict_model: str  # which model actually judged (pro, or free fallback)
    history: Annotated[list[str], operator.add]  # human-readable handoff log
