"""Layer 5: a running summary of the middle the window had to drop.

Inputs:  a Context whose window has already chosen what to keep
Outputs: a summary block, if one is needed and a summariser was injected

This is the layer everyone reaches for, and it is the most expensive thing you
can do to a prefix-caching provider: a summary rewrites the FRONT of the
prompt, so every token after it is recomputed. Capping shrinks a message in
place and stays cacheable; summarising does not. It amortises - one rewrite
holds for many turns - but the rewrite is never free.

`deps.summarize` is injected rather than imported, which is what keeps this
package free of network code. With no summariser the layer records ran=False
and returns unchanged, so the whole pipeline still assembles offline.
"""

from __future__ import annotations

from dataclasses import replace

from src.context import Block, Context
from src.layers.base import LayerDeps, PipelineConfig
from src.types import Message, Turn

SUMMARY_HEADER = "Summary of earlier turns:"


def _should_run(ctx: Context, cfg: PipelineConfig) -> bool:
    """Only summarise when something was actually lost."""
    return bool(ctx.dropped)


def dropped_text(dropped: tuple[Turn, ...]) -> str:
    """The material the summary has to stand in for."""
    return "\n".join(f"[turn {t.index}] {t.text()}" for t in dropped)


def _evict_for(ctx: Context, block: Block, deps: LayerDeps) -> Context:
    """Make room by dropping the oldest window turns.

    A summary that pushes the prompt past its budget is worse than no summary,
    so if it does not fit the window gives ground rather than the budget being
    quietly exceeded.
    """
    window = ctx.window
    if window is None:
        return ctx
    overflow = (ctx.used() + block.tokens) - ctx.budget
    if overflow <= 0:
        return ctx
    messages = list(window.messages)
    ids = list(window.turn_ids)
    tokens = window.tokens
    while overflow > 0 and len(ids) > 1:
        # Each turn contributes a run of messages; drop from the oldest end.
        per = max(1, len(messages) // max(len(ids), 1))
        removed, messages = messages[:per], messages[per:]
        freed = sum(deps.count(m.content) for m in removed)
        tokens -= freed
        overflow -= freed
        ids.pop(0)
    return replace(
        ctx,
        window=replace(
            window, messages=tuple(messages), tokens=max(0, tokens),
            turn_ids=tuple(ids), note="trimmed to make room for the summary",
        ),
    )


async def apply(ctx: Context, cfg: PipelineConfig, deps: LayerDeps) -> Context:
    """Write a running summary of everything the window could not keep."""
    if not cfg.enabled("summarize"):
        return ctx.traced(before=ctx, layer="summarize", ran=False)
    if deps.summarize is None:
        return ctx.traced(
            before=ctx, layer="summarize", ran=False, note="no summariser injected"
        )
    if not _should_run(ctx, cfg):
        return ctx.traced(
            before=ctx, layer="summarize", ran=False, note="nothing was dropped"
        )

    body = deps.summarize(dropped_text(ctx.dropped), cfg.summary_budget_tokens)
    if not body.strip():
        return ctx.traced(
            before=ctx, layer="summarize", ran=False, note="summariser returned nothing"
        )

    text = f"{SUMMARY_HEADER}\n{body.strip()}"
    block = Block(
        kind="summary",
        messages=(Message("system", text),),
        tokens=deps.count(text),
        turn_ids=tuple(t.index for t in ctx.dropped),
        note=f"stands in for {len(ctx.dropped)} turns",
    )
    out = _evict_for(ctx, block, deps)
    out = replace(out, summary=block)
    return out.traced(
        before=ctx,
        layer="summarize",
        note=f"{block.tokens} tok for {len(ctx.dropped)} dropped turns",
    )
