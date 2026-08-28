"""Layer 2: the system prompt, and a small never-evictable facts block.

Inputs:  a Context carrying a system prompt and pinned facts
Outputs: the same Context with those two blocks costed off the top

One rule here decides whether the ablation means anything: when `pin` is
OFF, the system prompt is still emitted. Only the pinned-facts block is
withheld. Dropping the system prompt too would change how the model behaves for
reasons that have nothing to do with context engineering, and every downstream
comparison would be measuring two things at once.
"""

from __future__ import annotations

from dataclasses import replace

from src.context import Block, Context
from src.layers.base import LayerDeps, PipelineConfig
from src.tokens import elide_middle
from src.types import Message

PIN_HEADER = "Facts to keep in mind for this session:"


async def apply(ctx: Context, cfg: PipelineConfig, deps: LayerDeps) -> Context:
    """Emit the system block always; the pinned block only when enabled."""
    system_block: Block | None = None
    if ctx.system_prompt:
        msg = Message("system", ctx.system_prompt)
        system_block = Block("system", (msg,), deps.count(ctx.system_prompt))

    pinned_block: Block | None = None
    note = "system only"
    if cfg.enabled("pin") and ctx.pins:
        body = PIN_HEADER + "\n" + "\n".join(f"- {p}" for p in ctx.pins)
        if deps.count(body) > cfg.pin_max_tokens:
            body = elide_middle(body, head=cfg.pin_max_tokens - 8, tail=6)
        msg = Message("system", body)
        pinned_block = Block("pinned", (msg,), deps.count(body))
        note = f"pinned {len(ctx.pins)} facts"

    out = replace(ctx, system=system_block, pinned=pinned_block)
    return out.traced(
        before=ctx, layer="pin", ran=cfg.enabled("pin"), note=note
    )
