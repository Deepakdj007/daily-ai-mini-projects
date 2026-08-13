"""The shapes that move through the system.

Inputs:  none
Outputs: Item, Verdict, TriageBatch, Draft, ScoutState

Strict-mode rule for every pydantic model here: each field is required, no
defaults, no Optional, and no numeric constraints. Groq's constrained decoding
rejects the `default`, `minimum` and `maximum` keywords that those produce, so a
range like 0-10 goes in the description and is clamped in Python afterwards.
"""

import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class Item:
    """One polled event, normalized so HN, arXiv and RSS look identical downstream.

    ts exists because the three sources disagree about time formats: HN sends a unix
    integer, arXiv sends ISO 8601, RSS sends RFC 822 ("Wed, 31 May 2023 00:00:00
    GMT"). Sorting those as strings puts every RSS entry ahead of everything else,
    because "W" > "2". So each fetcher converts to unix seconds, and ts is the only
    field anything sorts or filters on.
    """

    item_id: str      # "<source>:<native id>" — stable across nights
    source: str       # hn | arxiv | rss
    title: str
    url: str
    snippet: str
    ts: float         # unix seconds, UTC. 0.0 when the source gave no usable date
    created_at: str   # ISO 8601 for display, derived from ts


class Verdict(BaseModel):
    """One triage score."""

    item_id: str = Field(description="the exact id given in the list, copied verbatim")
    score: int = Field(description="0 to 10, how well this matches the interest profile")
    reason: str = Field(description="one short sentence, max 15 words")


class TriageBatch(BaseModel):
    """Every fresh item scored in a single call."""

    verdicts: list[Verdict] = Field(description="one entry per item, same order as given")


class Draft(BaseModel):
    """The reading-list entry a human is asked to approve."""

    headline: str = Field(description="rewrite the title as a plain-English claim")
    why_it_matters: str = Field(description="two sentences, addressed to the reader")
    key_points: list[str] = Field(description="2 to 4 bullets, each under 20 words")
    tags: list[str] = Field(description="1 to 3 lowercase topic tags")


class ScoutState(TypedDict, total=False):
    """State for one item's graph run. total=False so nodes return partial dicts."""

    item: dict                                   # Item, as a plain dict
    score: int
    reason: str
    night_id: str
    detail: str                                  # fetched article text, or the snippet
    draft: dict                                  # Draft.model_dump()
    decision: str                                # approve | edit | reject
    feedback: str                                # edit instructions from the human
    revision: int                                # how many times we have drafted
    status: str                                  # committed | discarded
    result: str                                  # what happened, for the CLI
    history: Annotated[list[str], operator.add]  # append-only trace
