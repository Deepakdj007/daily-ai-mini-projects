"""Two figures, both from data that costs nothing to produce.

Inputs:  output/results-<model>.json, plus an offline replay of the assembler
Outputs: output/context_cost.png and output/recall_by_zone.png

The cost figure is the one that carries the project. It plots two lines per
design: what the layer assembled, and what the provider actually charges for.
For an append-only history those two diverge - the prompt grows but almost all
of it is a prefix that was already sent. For a sliding window they do not
diverge at all, because every eviction rewrites the front and the whole thing
is billed again. The gap between a design's two lines is what prefix stability
is worth, and it is computable without a single API call.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config, summarizer  # noqa: E402
from src.harness import load  # noqa: E402
from src.ladder import BY_NAME  # noqa: E402
from src.pipeline import assemble, make_deps  # noqa: E402
from src.scenario import build  # noqa: E402
from src.tokens import common_prefix_tokens, count_text  # noqa: E402

INK = "#1f2933"
GRID = "#d9dee3"
COLOURS = {"raw": "#c1121f", "window": "#e08b00", "full": "#1c6dd0"}
WALL = "#c1121f"


async def _cost_series(arm_name: str, max_turn: int) -> tuple[list[int], list[int], list[int]]:
    """Replay one arm over a growing conversation, offline.

    Returns turn numbers, assembled tokens, and the tokens a prefix cache could
    NOT serve - which is what the provider bills and rate-limits on.
    """
    turns, _, spec = build()
    fixture = summarizer.load() or {"text": ""}
    deps = make_deps(summarize_fn=summarizer.as_fn(fixture), fudge=config.TOKENIZER_FUDGE)
    cfg = BY_NAME[arm_name].config

    xs: list[int] = []
    assembled: list[int] = []
    uncached: list[int] = []
    previous = ""
    for n in range(2, max_turn + 1):
        ctx = await assemble(
            turns[:n], "what is the current state?", cfg, deps,
            system_prompt=spec["system_prompt"], pins=spec["pinned_facts"],
            instruction=config.ANSWER_INSTRUCTION,
        )
        # The WHOLE prompt, not just the stable part. Pricing only the prefix
        # would let a design hide its per-query varying block - which is
        # exactly the block retrieval adds, and exactly what has to be paid
        # for. The common-prefix calculation then naturally stops where the
        # varying content starts, which is the honest model of a prefix cache.
        rendered = ctx.text()
        total = count_text(rendered, config.TOKENIZER_FUDGE)
        shared = common_prefix_tokens(previous, rendered) if previous else 0
        xs.append(n)
        assembled.append(ctx.used())
        uncached.append(max(0, total - shared))
        previous = rendered
    return xs, assembled, uncached


def cost_figure(max_turn: int = 60) -> None:
    """Assembled versus billable tokens, per design, against the wall."""
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    totals: dict[str, int] = {}

    for arm in ("raw", "window", "full"):
        xs, assembled, uncached = asyncio.run(_cost_series(arm, max_turn))
        colour = COLOURS[arm]
        ax.plot(xs, assembled, color=colour, lw=2.0, label=f"{arm} - assembled")
        ax.plot(xs, uncached, color=colour, lw=1.4, ls="--", alpha=0.85,
                label=f"{arm} - billable")
        totals[arm] = sum(uncached)

    ceiling = config.EFFECTIVE_PROMPT_CEILING
    ax.axhline(ceiling, color=WALL, lw=1.2, ls=":")
    ax.text(max_turn * 0.30, ceiling * 1.10,
            f"effective prompt ceiling {ceiling:,} tok",
            color=WALL, fontsize=9)

    xs, assembled, _ = asyncio.run(_cost_series("raw", max_turn))
    crossing = next((x for x, a in zip(xs, assembled) if a > ceiling), None)
    if crossing:
        ax.axvline(crossing, color=WALL, lw=0.9, alpha=0.5)
        ax.annotate(
            f"raw history stops running\nat turn {crossing}",
            xy=(crossing, ceiling), xytext=(crossing + 2, ceiling * 0.55),
            fontsize=9, color=WALL,
            arrowprops={"arrowstyle": "->", "color": WALL, "lw": 0.9},
        )

    ax.set_yscale("log")
    ax.set_xlabel("turns in the conversation")
    ax.set_ylabel("tokens (log scale)")
    ax.set_title("What the layer builds, and what you actually pay for", color=INK)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.14))

    lines = [
        "  |  ".join(f"{k}  {v:,} billable" for k, v in totals.items())
        + f"   over {max_turn} turns",
        f"The sliding window is the most expensive design here: "
        f"{totals['window'] / totals['raw'] - 1:.0%} more than sending everything,",
        "because every eviction rewrites the prefix.",
    ]
    fig.text(0.5, 0.015, "\n".join(lines), ha="center", va="bottom",
             fontsize=9, color=INK, linespacing=1.6)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(config.COST_PLOT_PATH, dpi=150)
    plt.close(fig)


def recall_figure(model: str = "") -> None:
    """Recall by zone, one group per arm."""
    from src.report import _rows_by_arm, recall

    _, rows = load(model)
    by_arm = _rows_by_arm(rows)
    order = [a for a in ("none", "window", "cap", "pin", "summarize", "full")
             if a in by_arm]
    zones = ("recent", "mid", "deep")
    palette = {"recent": "#7fb069", "mid": "#e08b00", "deep": "#1c6dd0"}

    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.26
    for i, zone in enumerate(zones):
        xs, ys = [], []
        for j, arm in enumerate(order):
            hit, n = recall(by_arm[arm], (zone,))
            xs.append(j + (i - 1) * width)
            ys.append(100.0 * hit / n if n else 0.0)
        ax.bar(xs, ys, width=width, label=zone, color=palette[zone])

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("recall %")
    ax.set_ylim(0, 105)
    ax.set_title("Recall by how far back the fact was planted", color=INK)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9, frameon=False, title="zone")
    fig.text(0.5, 0.005,
             "recent = inside every arm's window (the control). "
             "mid and deep are the headline.",
             ha="center", fontsize=8.5, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(config.RECALL_PLOT_PATH, dpi=150)
    plt.close(fig)


def main(model: str = "") -> None:
    """Write both figures."""
    cost_figure()
    print(f"wrote {config.COST_PLOT_PATH.name}")
    recall_figure(model)
    print(f"wrote {config.RECALL_PLOT_PATH.name}")


if __name__ == "__main__":
    main()
