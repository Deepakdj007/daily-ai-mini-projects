"""Layer 3: BM25 over the turns the window is about to drop.

Inputs:  a Context whose pool is capped and whose overheads are already costed
Outputs: a retrieved block holding the top-k UNCAPPED turns, rendered as a suffix

Two decisions carry this layer.

It re-injects the UNCAPPED original. `cap` has already run, so pulling a turn
back out of the pool would return it with its middle elided - and 40% of the
planted facts live in exactly that middle. Retrieval would then be structurally
unable to recover them, and the ceiling would read as a retrieval-quality
result when it was really an ordering bug.

It reserves a FIXED block, always the same size, and it renders after the
window rather than in chronological position. Both protect the same thing: the
prompt in front of the retrieved block must not move when the query changes,
or the provider's prefix cache has nothing to match.

The candidate pool is the turns the window will not keep - computed by asking
the same split_by_budget the window itself will use, so this is a real
lookahead rather than a guess.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Sequence

from rank_bm25 import BM25Okapi

from src.context import Block, Context
from src.layers.base import LayerDeps, PipelineConfig
from src.tokens import elide_middle, split_by_budget
from src.types import Message, Turn

RETRIEVED_HEADER = "Earlier in this conversation:"
_WORD = re.compile(r"[a-z0-9][a-z0-9\-]*")


def tokenize(text: str) -> list[str]:
    """Lowercase word split. BM25Okapi does no preprocessing of its own."""
    return _WORD.findall(text.lower())


def at_risk(ctx: Context, cfg: PipelineConfig) -> tuple[Turn, ...]:
    """The turns the window will not be able to keep.

    Uses the budget that will actually be left after this layer takes its
    reservation, so the lookahead matches what the window then does.
    """
    after_reservation = max(0, ctx.remaining() - cfg.retrieve_block_tokens)
    _, dropped = split_by_budget(
        ctx.pool, after_reservation, min_turns=cfg.window_min_turns
    )
    return dropped


def rank(query: str, turns: Sequence[Turn], k: int) -> list[tuple[Turn, float]]:
    """Top-k turns by BM25 against the query, best first."""
    if not turns:
        return []
    corpus = [tokenize(t.text()) for t in turns]
    if not any(corpus):
        return []
    scores = BM25Okapi(corpus).get_scores(tokenize(query))
    ordered = sorted(zip(turns, scores), key=lambda p: p[1], reverse=True)
    return ordered[:k]


def _render(
    hits: Sequence[tuple[Turn, float]], cfg: PipelineConfig, deps: LayerDeps
) -> Block:
    """Pack the hits into one suffix block, capped at the fixed reservation.

    The block is always claimed at retrieve_block_tokens even when it holds
    less, because a reservation that shrank with the hit list would hand tokens
    back to the window and change how many turns it keeps per query.
    """
    messages: list[Message] = [Message("user", RETRIEVED_HEADER)]
    ids: list[int] = []
    spent = deps.count(RETRIEVED_HEADER)
    for turn, score in hits:
        if score <= cfg.retrieve_min_score:
            continue
        body = f"[turn {turn.index}] {turn.text()}"
        cost = deps.count(body)
        room = cfg.retrieve_block_tokens - spent
        if cost > room:
            if ids:
                break
            # The single best hit is worth carrying even when it does not fit
            # whole, but it is elided to the reservation rather than allowed to
            # overrun it - the whole point of a fixed block is that the prompt
            # in front of it never moves.
            # Shrink until it fits. The elision marker itself costs tokens,
            # so one pass at room-minus-a-guess can still overrun; two or three
            # halvings of the head always converge and cost nothing.
            head = max(8, room - 24)
            while head >= 8:
                body = elide_middle(f"[turn {turn.index}] {turn.text()}",
                                    head=head, tail=6)
                cost = deps.count(body)
                if cost <= room:
                    break
                head //= 2
            if cost > room:
                break
        messages.append(Message("user", body))
        ids.append(turn.index)
        spent += cost
    return Block(
        kind="retrieved",
        messages=tuple(messages) if ids else (),
        tokens=cfg.retrieve_block_tokens,
        turn_ids=tuple(ids),
        note=f"{len(ids)} hits, {spent} tok used of {cfg.retrieve_block_tokens} reserved",
    )


async def apply(ctx: Context, cfg: PipelineConfig, deps: LayerDeps) -> Context:
    """Recall the most relevant turns the window is about to lose."""
    if not cfg.enabled("retrieve") or not ctx.pool:
        return ctx.traced(before=ctx, layer="retrieve", ran=False)

    candidates = at_risk(ctx, cfg)
    if not candidates:
        return ctx.traced(
            before=ctx, layer="retrieve", note="nothing at risk of being dropped"
        )

    hits = rank(ctx.query, candidates, cfg.retrieve_k)
    # Swap each hit for its pre-cap original before rendering.
    uncapped = [(ctx.original(t.index) or t, s) for t, s in hits]
    block = _render(uncapped, cfg, deps)
    out = replace(ctx, retrieved=block)
    return out.traced(
        before=ctx,
        layer="retrieve",
        note=f"pool {len(candidates)}, {block.note}",
    )
