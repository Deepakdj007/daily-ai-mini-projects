"""Chain the enabled layers, and expose the one function worth copying.

Inputs:  a conversation history, a query, a budget
Outputs: an assembled Context - or, via build_messages, a plain API payload

`build_messages` is the whole deliverable. Everything else in this repo exists
to test it. It takes a list of Messages and a token budget and hands back
something you can pass straight to a chat-completions call, so lifting this
into another agent means copying src/layers/, context.py, tokens.py, types.py
and this file - and nothing that knows about the experiment.

Layer order is fixed and it matters:
  cap       shrink messages in place, before anything measures a budget
  pin       claim the non-negotiable space off the top
  retrieve  look ahead at what the window will drop, reserve a suffix for it
  window    fill what is left with the most recent turns
  summarize stand in for whatever the window still had to lose
"""

from __future__ import annotations

from typing import Sequence

from src.context import Context, new_context
from src.layers import cap, pin, retrieve, summarize, window
from src.layers.base import LayerDeps, PipelineConfig
from src.tokens import PER_MESSAGE_OVERHEAD, common_prefix_tokens, count_text
from src.types import Message, SummarizeFn, Turn

LAYER_ORDER: tuple[tuple[str, object], ...] = (
    ("cap", cap),
    ("pin", pin),
    ("retrieve", retrieve),
    ("window", window),
    ("summarize", summarize),
)


def make_deps(*, summarize_fn: SummarizeFn | None = None, fudge: float = 1.0) -> LayerDeps:
    """Bundle the capabilities the layers are allowed to use.

    The counter prices a string as a MESSAGE, framing included. Every string a
    layer measures becomes exactly one message in the payload, and
    split_by_budget already counts turns that way - a counter that priced raw
    text would put the blocks and the window on two different scales and
    quietly enforce a budget nobody chose.
    """
    return LayerDeps(
        count=lambda text: count_text(text, fudge) + PER_MESSAGE_OVERHEAD,
        summarize=summarize_fn,
    )


async def assemble(
    turns: Sequence[Turn],
    query: str,
    cfg: PipelineConfig,
    deps: LayerDeps,
    *,
    system_prompt: str = "",
    pins: Sequence[str] = (),
    instruction: str = "",
) -> Context:
    """Run every layer in order and return the assembled context."""
    ctx = new_context(
        turns,
        query,
        budget=cfg.context_budget,
        system_prompt=system_prompt,
        pins=pins,
        instruction=instruction,
        include_history=cfg.enabled("history"),
    )
    for _, module in LAYER_ORDER:
        ctx = await module.apply(ctx, cfg, deps)  # type: ignore[attr-defined]
    return ctx


async def build_messages(
    history: Sequence[Message],
    budget: int,
    *,
    query: str = "",
    system: str = "",
    pinned: Sequence[str] = (),
    summarize_fn: SummarizeFn | None = None,
    instruction: str = "",
) -> list[dict[str, str]]:
    """Assemble a chat payload that fits `budget` tokens. The public entry point.

    `history` is grouped into turns on each user message, which is the usual
    shape of an agent loop's message list. If your history is already turn
    structured, call `assemble` directly and keep the grouping you have.
    """
    turns: list[Turn] = []
    current: list[Message] = []
    for msg in history:
        if msg.role == "user" and current:
            turns.append(Turn(len(turns), tuple(current)))
            current = []
        current.append(msg)
    if current:
        turns.append(Turn(len(turns), tuple(current)))

    cfg = PipelineConfig(context_budget=budget)
    ctx = await assemble(
        turns, query, cfg, make_deps(summarize_fn=summarize_fn),
        system_prompt=system, pins=pinned, instruction=instruction,
    )
    return ctx.messages()


def prefix_stability(contexts: Sequence[Context]) -> float:
    """Mean fraction of each prompt that the previous one already contained.

    The whole cache measurement, at zero API cost. A provider caches a prefix,
    so this is exactly how much of a request could be served from cache -
    exact rather than sampled, and computable for every arm rather than the one
    a live placement experiment could afford.
    """
    if len(contexts) < 2:
        return 1.0
    fractions: list[float] = []
    for previous, current in zip(contexts, contexts[1:]):
        rendered = current.prefix_text()
        shared = common_prefix_tokens(previous.prefix_text(), rendered)
        total = max(1, count_text(rendered))
        fractions.append(min(1.0, shared / total))
    return sum(fractions) / len(fractions)


if __name__ == "__main__":
    import asyncio
    from dataclasses import replace as _replace

    def _fake_summary(text: str, budget: int) -> str:
        return f"({len(text.split())} words of earlier troubleshooting)"

    turns = tuple(
        Turn(i, (
            Message("user", f"turn {i}: what happened at step {i}? " + "pad " * 4),
            Message("assistant", f"step {i} completed " + "pad " * 6),
            Message("tool", '{"step": %d, "rows": [%s]}' % (i, "1," * 120), "ledger"),
        ))
        for i in range(40)
    )
    deps = make_deps(summarize_fn=_fake_summary)
    full = PipelineConfig(context_budget=900)

    async def _smoke() -> None:
        ctx = await assemble(
            turns, "what happened at step 7?", full, deps,
            system_prompt="You are the PaySetu incident desk.",
            pins=("merchant M-88213 is on the Growth plan",),
            instruction="Answer in one line.",
        )
        print(f"{'layer':<10} {'ran':<6} {'tokens':>14} {'note'}")
        for tr in ctx.traces:
            print(f"{tr.layer:<10} {str(tr.ran):<6} "
                  f"{tr.tokens_before:>6} -> {tr.tokens_after:<5} {tr.note}")
        print(f"\nassembled {ctx.used()} of {ctx.budget} tok, "
              f"{ctx.turns_in_context()} turns, blocks "
              f"{[b.kind for b in ctx.blocks()]}")

        assert len(ctx.traces) == 5, "every layer must leave a row, enabled or not"
        assert ctx.used() <= ctx.budget, f"{ctx.used()} over budget {ctx.budget}"

        win = set(ctx.window.turn_ids) if ctx.window else set()
        ret = set(ctx.retrieved.turn_ids) if ctx.retrieved else set()
        assert not (win & ret), f"turn in both window and retrieved: {win & ret}"

        # The reservation must not move with the hit list.
        assert ctx.retrieved is None or ctx.retrieved.tokens == full.retrieve_block_tokens

        off = _replace(full, retrieve=False, summarize=False)
        lean = await assemble(turns, "what happened at step 7?", off, deps,
                              system_prompt="You are the PaySetu incident desk.")
        print(f"cap+pin+window keeps {lean.turns_in_context()} turns; "
              f"full keeps {ctx.turns_in_context()}")

        no_cap = _replace(off, cap=False)
        raw = await assemble(turns, "q", no_cap, deps, system_prompt="desk")
        print(f"window alone keeps {raw.turns_in_context()} turns; "
              f"capping raises that to {lean.turns_in_context()}")
        assert lean.turns_in_context() > raw.turns_in_context(), (
            "capping must buy more turns at the same budget"
        )

        # Prefix stability across a growing conversation, for two designs.
        async def curve(cfg: PipelineConfig) -> float:
            snaps = [
                await assemble(turns[:n], "same question", cfg, deps,
                               system_prompt="desk")
                for n in range(8, 20)
            ]
            return prefix_stability(snaps)

        evicting = await curve(_replace(full, retrieve=False, summarize=False))
        appending = await curve(
            _replace(full, retrieve=False, summarize=False, context_budget=99_000)
        )
        print(f"\nprefix stability - evicting window {evicting:.0%}, "
              f"append-only (huge budget) {appending:.0%}")
        assert appending > evicting, (
            "an append-only history must keep more cacheable prefix than an "
            "evicting window - this is the project's whole claim"
        )

    asyncio.run(_smoke())
