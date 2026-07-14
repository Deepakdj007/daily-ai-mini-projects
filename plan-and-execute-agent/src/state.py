"""Graph state and the planner/replanner structured-output schemas.

Plan is what the planner produces. Act is what the replanner produces after each
executed step — a flat schema (not a Plan|Response union) because gpt-oss returns
it reliably through json_schema mode. PlanExecuteState flows through every node.
"""

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """An ordered list of steps that will achieve the goal."""

    steps: list[str] = Field(
        description="Ordered, self-contained steps. Each step is one clear action."
    )


class Act(BaseModel):
    """The replanner's decision after a step runs: finish, or keep going.

    A flat schema on purpose. gpt-oss returns a discriminated union
    (Plan | Response) unreliably, but sets these three fields cleanly.
    """

    done: bool = Field(
        description="True if the collected results already answer the goal."
    )
    answer: str = Field(
        default="",
        description="The final answer to the user. Fill this only when done=True.",
    )
    remaining_steps: list[str] = Field(
        default_factory=list,
        description="The steps still left to do. Fill this only when done=False.",
    )


class PlanExecuteState(TypedDict, total=False):
    """State carried through the graph. total=False so nodes return partial dicts."""

    task: str                                              # the user's goal
    plan: list[str]                                        # steps still to execute
    # (step, result) pairs, appended by the executor. The operator.add reducer lets
    # each executor return just its one new pair without clobbering the history.
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    response: str                                          # final answer when done
