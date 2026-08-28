"""Layer 1: truncate over-long individual messages, head and tail.

Inputs:  a Context whose pool holds raw turns
Outputs: the same Context with every over-long message shortened

Fat tool output is what actually kills agents, and this is the only layer that
is free on both axes: it shrinks the context AND keeps the prefix cacheable.

The reason is that `_cap_message` is a pure function of the message. It never
looks at the budget, the position in the conversation, or how many turns came
after. So a message renders byte-identically on turn 100 as it did on turn 4,
and a provider's prefix cache still matches. Any cap that adapted to remaining
budget would rewrite history on every turn and forfeit the cache entirely -
which is the trap the sliding window falls into two layers down.
"""

from __future__ import annotations

from dataclasses import replace

from src.context import Context
from src.layers.base import LayerDeps, PipelineConfig
from src.tokens import elide_middle
from src.types import Message, Turn


def _cap_message(msg: Message, cfg: PipelineConfig, deps: LayerDeps) -> tuple[Message, bool]:
    """Shorten one message if it is over the ceiling. Pure in the message."""
    if deps.count(msg.content) <= cfg.cap_max_message_tokens:
        return msg, False
    shortened = elide_middle(
        msg.content, head=cfg.cap_head_tokens, tail=cfg.cap_tail_tokens
    )
    return replace(msg, content=shortened), True


def _cap_turn(turn: Turn, cfg: PipelineConfig, deps: LayerDeps) -> tuple[Turn, int]:
    """Cap every message in a turn, reporting how many were shortened."""
    capped: list[Message] = []
    hits = 0
    for msg in turn.messages:
        new_msg, was_capped = _cap_message(msg, cfg, deps)
        capped.append(new_msg)
        hits += int(was_capped)
    return replace(turn, messages=tuple(capped)), hits


async def apply(ctx: Context, cfg: PipelineConfig, deps: LayerDeps) -> Context:
    """Cap every message in the pool."""
    if not cfg.enabled("cap") or not ctx.pool:
        return ctx.traced(before=ctx, layer="cap", ran=False)
    capped: list[Turn] = []
    total = 0
    for turn in ctx.pool:
        new_turn, hits = _cap_turn(turn, cfg, deps)
        capped.append(new_turn)
        total += hits
    out = replace(ctx, pool=tuple(capped))
    return out.traced(
        before=ctx, layer="cap", note=f"shortened {total} messages"
    )
