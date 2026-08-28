"""The vocabulary every other module speaks: messages, turns, facts, results.

Inputs:  nothing - this is a leaf module with no src/ imports
Outputs: the dataclasses and type aliases used across the whole project

Split into two halves on purpose. The domain half (Message, Turn) travels with
the reusable layer; the measurement half (Probe, ProbeResult, Usage) belongs to
the experiment. Keeping them in one leaf file avoids a circular import without
pretending they are the same kind of thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, Sequence

# --- Domain: what the layer moves around -------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class Message:
    """One message in a conversation. `name` labels a tool call's source."""

    role: Role
    content: str
    name: str = ""

    def as_api(self) -> dict[str, str]:
        """Render to the shape the chat-completions API expects.

        Tool results are sent as user messages carrying a labelled block. The
        real tool-call protocol would work too, but it makes the assembled
        prompt harder to diff, and diffing prompts is how prefix stability is
        measured here.
        """
        if self.role == "tool":
            return {"role": "user", "content": f"[tool:{self.name}]\n{self.content}"}
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class Turn:
    """One exchange: a user message, an assistant reply, and any tool output.

    The turn is the unit the window keeps and the retriever returns. That is a
    simplification worth naming: for a 3,000-token tool result the turn is far
    too coarse a unit, and the README says so.
    """

    index: int
    messages: tuple[Message, ...]

    def text(self) -> str:
        """Flatten to plain text, for BM25 and for the fact-presence grep."""
        return "\n".join(m.content for m in self.messages)


class TokenCounter(Protocol):
    """Anything that can price a string in tokens."""

    def __call__(self, text: str) -> int: ...


SummarizeFn = Callable[[str, int], str]
"""Takes the dropped-middle text and a token budget, returns a summary."""


# --- Measurement: what the experiment records --------------------------------

Zone = Literal["recent", "mid", "deep", "negative", "doc"]
Carrier = Literal["prose", "tool_head", "tool_body", "none"]

Status = Literal[
    "correct",
    "wrong",
    "abstained",
    "unparseable",
    "truncated",
    "oversized",
    "rate_limited",
    "api_error",
]
"""Assigned in precedence order, so exactly one applies.

`contaminated` is deliberately absent. It is not a status but one cell of a 2x2
against `fact_present`, and folding it in here would make the other three cells
unrecoverable. report.py derives it.
"""

NON_SCORED: frozenset[str] = frozenset(
    {"truncated", "oversized", "rate_limited", "api_error", "unparseable"}
)
"""Outcomes that are not the model's answer. Counted as failures for pairing -
mcnemar_exact zips positionally, so dropping a row misaligns every later pair -
but reported separately so a service failure never reads as a memory result.
"""


@dataclass(frozen=True, slots=True)
class Fact:
    """One planted value, and the question that asks for it."""

    id: str
    turn: int
    zone: Zone
    carrier: Carrier
    value: str
    probe: str
    answer_space: int = 1000
    schema: str = ""
    field_name: str = ""
    distractor: bool = False


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one call.

    `cached_tokens` is None rather than 0 when it cannot be known - a replayed
    row, or an arm whose prefix is below the model's minimum cacheable length.
    Writing 0 there would report a fabricated cache statistic.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int | None = None
    latency_s: float = 0.0

    @property
    def uncached_tokens(self) -> int:
        """What actually counted against the rate limit."""
        return self.prompt_tokens - (self.cached_tokens or 0)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """One (arm, probe) cell of the results grid."""

    arm: str
    model: str
    probe_id: str
    zone: Zone
    carrier: Carrier
    status: Status
    answer: str
    gold: str
    fact_present: bool | None
    assembled_tokens: int
    turns_in_context: int
    retrieval_hit: bool | None
    usage: Usage
    replayed: bool = False
    error: str = ""
    layer_trace: tuple[dict[str, object], ...] = field(default_factory=tuple)

    @property
    def contaminated(self) -> bool:
        """Answered correctly from a context the fact demonstrably was not in.

        Any non-zero count here means a probe is answerable some other way and
        the whole scorecard is suspect - so it voids the run rather than
        quietly inflating an arm.
        """
        return self.fact_present is False and self.status == "correct"

    @property
    def scored(self) -> bool:
        """False for service failures, which are reported but never graded."""
        return self.status not in NON_SCORED


def as_api_messages(messages: Sequence[Message]) -> list[dict[str, str]]:
    """Render a message sequence into an API payload."""
    return [m.as_api() for m in messages]


if __name__ == "__main__":
    turn = Turn(
        index=3,
        messages=(
            Message("user", "why did the payout retry?"),
            Message("tool", '{"incident": "INC-4471", "p99_latency_ms": 4712}', "ledger"),
            Message("assistant", "It hit the retry ceiling."),
        ),
    )
    print("turn text:", turn.text().replace("\n", " | "))
    print("api roles:", [m["role"] for m in as_api_messages(turn.messages)])

    u = Usage(prompt_tokens=3000, completion_tokens=90, cached_tokens=2800)
    print(f"uncached: {u.uncached_tokens} of {u.prompt_tokens}")
    assert Usage(prompt_tokens=500).uncached_tokens == 500, "None cached must not crash"

    hit = ProbeResult(
        arm="full", model="m", probe_id="f07", zone="deep", carrier="tool_body",
        status="correct", answer="4,712", gold="4,712", fact_present=False,
        assembled_tokens=2900, turns_in_context=5, retrieval_hit=False, usage=u,
    )
    assert hit.contaminated, "absent + correct must flag as contaminated"
    assert hit.scored
    miss = ProbeResult(
        arm="raw", model="m", probe_id="f07", zone="deep", carrier="prose",
        status="oversized", answer="", gold="4,712", fact_present=None,
        assembled_tokens=14000, turns_in_context=30, retrieval_hit=None, usage=Usage(),
    )
    assert not miss.contaminated, "a negative/None fact_present must never contaminate"
    assert not miss.scored, "oversized is a service outcome, not a wrong answer"
    print("contamination + scoring rules hold")
