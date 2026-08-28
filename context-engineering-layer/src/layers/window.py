"""Layer 4: keep the most recent turns that fit what budget is left.

Inputs:  a Context whose earlier layers have already claimed their space
Outputs: a window block, and everything it could not keep moved to `dropped`

This is the layer everybody reaches for first, and it is the one that quietly
breaks prefix caching. A window that evicts its oldest turn changes the front
of the prompt, so every token after the eviction point has to be recomputed.
Capping shrinks a message in place and stays cacheable; evicting does not. The
prefix-stability figure that plot.py draws is what makes that visible.

Window also wins over retrieve. Retrieval predicts what will be dropped, but it
spends budget too, so a turn it predicted would be dropped can end up kept -
and then rendered twice, out of order. Overlapping turns are stripped from the
retrieved block; the tokens are NOT returned to the window, because a window
whose size depended on retrieval results would vary per query and destroy the
stable prefix that the whole suffix layout exists to protect.
"""

from __future__ import annotations

from dataclasses import replace

from src.context import Context, block_from_turns
from src.layers.base import LayerDeps, PipelineConfig
from src.tokens import split_by_budget


def _dedupe(ctx: Context, kept_ids: set[int]) -> Context:
    """Drop retrieved turns the window already kept. Budget is not refunded."""
    block = ctx.retrieved
    if block is None:
        return ctx
    overlap = [i for i in block.turn_ids if i in kept_ids]
    if not overlap:
        return ctx
    survivors = tuple(i for i in block.turn_ids if i not in kept_ids)
    if not survivors:
        return replace(ctx, retrieved=replace(block, messages=(), turn_ids=(),
                                              note="all hits already in window"))
    # Messages are matched back to their turn by the "[turn N]" prefix the
    # retriever writes, so an overlapping hit is genuinely removed from the
    # payload rather than merely dropped from the id list.
    kept_messages = [block.messages[0]] if block.messages else []
    for msg in block.messages[1:]:
        if any(msg.content.startswith(f"[turn {i}]") for i in survivors):
            kept_messages.append(msg)
    return replace(
        ctx,
        retrieved=replace(
            block,
            messages=tuple(kept_messages),
            turn_ids=survivors,
            note=f"dropped {len(overlap)} already in window",
        ),
    )


async def apply(ctx: Context, cfg: PipelineConfig, deps: LayerDeps) -> Context:
    """Fill the remaining budget with the most recent turns."""
    if not ctx.pool:
        return ctx.traced(before=ctx, layer="window", ran=False)

    if not cfg.enabled("window"):
        # No windowing means unbounded history, not empty history. This is the
        # arm that demonstrates the break, and it can only do that if it
        # actually tries to send all 15,000 tokens.
        everything = block_from_turns("window", ctx.pool, deps.count)
        out = replace(ctx, window=everything, dropped=())
        return out.traced(
            before=ctx, layer="window", ran=False,
            note=f"unbounded - all {len(ctx.pool)} turns",
        )

    kept, dropped = split_by_budget(
        ctx.pool,
        ctx.remaining(),
        min_turns=cfg.window_min_turns,
    )
    block = block_from_turns("window", kept, deps.count)
    out = replace(ctx, window=block, dropped=dropped)
    out = _dedupe(out, {t.index for t in kept})
    reach = kept[0].index if kept else -1
    return out.traced(
        before=ctx,
        layer="window",
        note=f"kept {len(kept)} turns, back to turn {reach}",
    )
