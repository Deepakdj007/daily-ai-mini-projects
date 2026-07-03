"""Graph state and the planner's structured-output schema.

PlannedAction is what the LLM must produce (validated by Pydantic).
ApprovalState is the dict that flows through every node in the graph.
"""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class PlannedAction(BaseModel):
    """The irreversible action the agent wants to take, drafted by the LLM.

    The four content fields are generic on purpose so one schema covers all
    three action types (email / calendar / ticket) without branching.
    """

    action_type: Literal["email", "calendar", "ticket"] = Field(
        description="Which kind of irreversible action this is."
    )
    recipient: str = Field(
        description="Email to-address, calendar attendee, or ticket assignee."
    )
    subject: str = Field(
        description="Email subject, event title, or ticket title."
    )
    body: str = Field(
        description="Email body, event details, or ticket description."
    )
    summary: str = Field(
        description="One short human-readable line describing the action."
    )


class ApprovalState(TypedDict, total=False):
    """State carried through the graph. total=False so nodes return partial dicts."""

    request: str                                    # the user's natural-language ask
    thread_id: str                                  # carried in state for audit entries
    action: dict                                    # PlannedAction.model_dump()
    decision: str                                   # approve | edit | reject
    feedback: str                                   # edit instructions / rejection reason
    revision: int                                   # how many times we've drafted
    valid: bool                                     # validator verdict
    validation_errors: list[str]                    # why the draft was bounced
    status: str                                     # pending | sent | cancelled
    result: str                                     # confirmation / cancellation message
    history: Annotated[list[str], operator.add]     # append-only handoff log
