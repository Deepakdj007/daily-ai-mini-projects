"""The immutable carrier every layer receives and returns.

Inputs:  turns, a query, a budget
Outputs: Context, Block, LayerTrace - and the rendered API payload

Imports nothing from config.py, so this travels with the reusable layer.

Blocks render in a fixed order, and that order is the cache design. Everything
up to the retrieved block is prefix-stable across queries at the same history
depth; only the retrieved block and the question vary. Reordering these is not
a style choice - it decides how much of each request the provider can serve
from cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Sequence

from src.types import Message, TokenCounter, Turn

BlockKind = Literal["system", "pinned", "summary", "window", "retrieved"]

RENDER_ORDER: tuple[BlockKind, ...] = (
    "system",
    "pinned",
    "summary",
    "window",
    "retrieved",
)
"""Stable prefix first, per-query varying content last.

`retrieved` sits after `window` on purpose. Injecting recalled turns in
chronological position would put per-query varying content in front of the
window and invalidate every cached token after it.
"""


@dataclass(frozen=True, slots=True)
class Block:
    """A contiguous, independently priced region of the assembled prompt."""

    kind: BlockKind
    messages: tuple[Message, ...] = ()
    tokens: int = 0
    turn_ids: tuple[int, ...] = ()
    note: str = ""


@dataclass(frozen=True, slots=True)
class LayerTrace:
    """One row of the per-assembly audit trail.

    A disabled layer still records a row with ran=False, so every arm's trace
    has the same shape and the report can align them without special cases.
    """

    layer: str
    ran: bool
    tokens_before: int
    tokens_after: int
    turns_before: int
    turns_after: int
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        """Flatten for the results JSON."""
        return {
            "layer": self.layer,
            "ran": self.ran,
            "tokens": [self.tokens_before, self.tokens_after],
            "turns": [self.turns_before, self.turns_after],
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class Context:
    """What flows through the pipeline. Every layer returns a new one."""

    query: str
    instruction: str = ""
    system_prompt: str = ""
    pins: tuple[str, ...] = ()
    budget: int = 0
    include_history: bool = True

    system: Block | None = None
    pinned: Block | None = None
    summary: Block | None = None
    window: Block | None = None
    retrieved: Block | None = None

    pool: tuple[Turn, ...] = ()
    dropped: tuple[Turn, ...] = ()
    # The turns as they arrived, before `cap` shortened anything. `retrieve`
    # re-injects from here: handing back a turn whose middle cap had already
    # elided would make retrieval structurally unable to recover the 40% of
    # facts that live in that middle, and the ceiling would look like a
    # retrieval-quality result when it was really an ordering bug.
    originals: tuple[Turn, ...] = ()
    traces: tuple[LayerTrace, ...] = field(default_factory=tuple)

    def blocks(self) -> tuple[Block, ...]:
        """Present blocks, in render order."""
        by_kind = {
            "system": self.system,
            "pinned": self.pinned,
            "summary": self.summary,
            "window": self.window,
            "retrieved": self.retrieved,
        }
        return tuple(b for k in RENDER_ORDER if (b := by_kind[k]) is not None)

    def used(self) -> int:
        """Tokens the assembled blocks currently occupy."""
        return sum(b.tokens for b in self.blocks())

    def remaining(self) -> int:
        """Budget still unspent. Never negative."""
        return max(0, self.budget - self.used())

    def original(self, index: int) -> Turn | None:
        """The uncapped turn at `index`, if the history included it."""
        for turn in self.originals:
            if turn.index == index:
                return turn
        return None

    def turns_in_context(self) -> int:
        """How many distinct turns are present verbatim in the prompt.

        The summary block is excluded even though it carries the ids of every
        turn it stands in for. Those turns are represented, not present - and
        counting them would let a one-line summary claim ninety turns of
        context, which is exactly the overclaim this project exists to measure.
        """
        ids: set[int] = set()
        for block in self.blocks():
            if block.kind == "summary":
                continue
            ids.update(block.turn_ids)
        return len(ids)

    def turns_summarised(self) -> int:
        """How many turns the summary stands in for, reported separately."""
        return len(self.summary.turn_ids) if self.summary else 0

    def messages(self) -> list[dict[str, str]]:
        """The actual API payload.

        The question and the answer instruction always come last, in the same
        shape, whatever the arms did upstream - so an arm never differs from
        another in how the task is framed, only in what history it saw.
        """
        out: list[dict[str, str]] = []
        for block in self.blocks():
            out.extend(m.as_api() for m in block.messages)
        tail = self.query if not self.instruction else f"{self.query}\n\n{self.instruction}"
        out.append({"role": "user", "content": tail})
        return out

    def text(self) -> str:
        """Everything the model will see, flattened.

        Used for the fact-presence grep. It must agree with messages(); the
        probe runner asserts that, because a silent divergence would turn a
        present fact into a phantom contamination event.
        """
        return "\n".join(m["content"] for m in self.messages())

    def prefix_text(self) -> str:
        """The part of the prompt that does not vary with the query.

        Everything except the retrieved block and the trailing question. This
        is what a provider's prefix cache can actually reuse between two
        queries at the same history depth.
        """
        stable = [b for b in self.blocks() if b.kind != "retrieved"]
        return "\n".join(m.content for b in stable for m in b.messages)

    def traced(
        self,
        *,
        before: Context,
        layer: str,
        ran: bool = True,
        note: str = "",
    ) -> Context:
        """Append a trace row describing what one layer just did."""
        row = LayerTrace(
            layer=layer,
            ran=ran,
            tokens_before=before.used(),
            tokens_after=self.used(),
            turns_before=len(before.pool),
            turns_after=len(self.pool),
            note=note,
        )
        return replace(self, traces=self.traces + (row,))


def new_context(
    turns: Sequence[Turn],
    query: str,
    *,
    budget: int,
    system_prompt: str = "",
    pins: Sequence[str] = (),
    instruction: str = "",
    include_history: bool = True,
) -> Context:
    """Start an assembly. `include_history=False` is the null-context arm."""
    return Context(
        query=query,
        instruction=instruction,
        system_prompt=system_prompt,
        pins=tuple(pins),
        budget=budget,
        include_history=include_history,
        pool=tuple(turns) if include_history else (),
        originals=tuple(turns) if include_history else (),
    )


def block_from_turns(
    kind: BlockKind, turns: Sequence[Turn], count: TokenCounter
) -> Block | None:
    """Pack whole turns into one block, or None if there are none."""
    if not turns:
        return None
    messages = tuple(m for t in turns for m in t.messages)
    return Block(
        kind=kind,
        messages=messages,
        tokens=sum(count(m.content) for m in messages),
        turn_ids=tuple(t.index for t in turns),
    )


if __name__ == "__main__":
    from src import tokens

    count: TokenCounter = tokens.count_text
    turns = tuple(
        Turn(i, (Message("user", f"q{i} " + "pad " * 10),
                 Message("assistant", f"a{i} " + "pad " * 10)))
        for i in range(6)
    )
    ctx = new_context(turns, "what was q2?", budget=400, system_prompt="you are a desk")
    ctx = replace(
        ctx,
        system=Block("system", (Message("system", ctx.system_prompt),), count(ctx.system_prompt)),
        window=block_from_turns("window", turns[-2:], count),
        retrieved=block_from_turns("retrieved", turns[2:3], count),
    )

    kinds = [b.kind for b in ctx.blocks()]
    print("render order:", kinds)
    assert kinds == ["system", "window", "retrieved"], "retrieved must render last"
    assert ctx.messages()[-1]["content"].startswith("what was q2?")
    print(f"used {ctx.used()} of {ctx.budget}, turns in context {ctx.turns_in_context()}")

    other = replace(ctx, query="what was q3?")
    assert ctx.prefix_text() == other.prefix_text(), (
        "changing only the query must not move the cacheable prefix"
    )
    assert "q2" in ctx.text() and "q2" not in ctx.prefix_text().split("q4")[-1]
    print("prefix is query-invariant")

    traced = ctx.traced(before=new_context(turns, "x", budget=400), layer="window")
    assert traced.traces[0].layer == "window" and traced.traces[0].ran
    print("trace:", traced.traces[0].as_dict())
