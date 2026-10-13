"""Turn two replay runs into the numbers and the chart.

Inputs: the Records from a run without the cache and a run with it.
Outputs: a printed totals table and cache_report.png (cost + latency).
"""

import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNG files, never open a window
import matplotlib.pyplot as plt

from src.agent import Record

NO_CACHE = "#eb6834"
CACHE = "#2a78d6"
INK = "#52514e"
GRID = "#e4e3df"


def totals(records: list[Record]) -> dict:
    """Sum up one run: LLM calls, dollars, and median latency."""
    return {
        "calls": sum(not r.hit for r in records),
        "cost": sum(r.cost for r in records),
        "p50_ms": statistics.median(r.latency_ms for r in records),
        "hit_p50_ms": statistics.median([r.latency_ms for r in records if r.hit] or [0]),
    }


def print_totals(plain: list[Record], cached: list[Record]) -> float:
    """Print the side-by-side table; return the percentage of cost saved."""
    a, b = totals(plain), totals(cached)
    saved = 100 * (1 - b["cost"] / a["cost"])
    per_100k = 100_000 / len(plain)
    print(f"\n{'':22}{'no cache':>14}{'with cache':>14}")
    print(f"{'LLM calls':22}{a['calls']:>14}{b['calls']:>14}")
    print(f"{'cost (paid tier)':22}{'$' + format(a['cost'], '.5f'):>14}{'$' + format(b['cost'], '.5f'):>14}")
    print(f"{'cost per 100k questions':22}{'$' + format(a['cost'] * per_100k, '.2f'):>14}"
          f"{'$' + format(b['cost'] * per_100k, '.2f'):>14}")
    print(f"{'median latency':22}{a['p50_ms']:>12.0f}ms{b['p50_ms']:>12.0f}ms")
    print(f"{'median cache hit':22}{'-':>14}{b['hit_p50_ms']:>12.0f}ms")
    print(f"\nCost saved: {saved:.0f}%   ({len(plain) - b['calls']} of {len(plain)} answered from cache)")
    return saved


def _style(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Recessive grid and axes, left-aligned title."""
    ax.set_title(title, loc="left", fontsize=12, color="#0b0b0b")
    ax.set_xlabel("question # (arrival order)", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_cost(ax: plt.Axes, plain: list[Record], cached: list[Record]) -> None:
    """Cumulative dollars spent, request by request, with and without the cache."""
    xs = range(1, len(plain) + 1)
    for records, color, label in ((plain, NO_CACHE, "no cache"), (cached, CACHE, "with cache")):
        running, total = [], 0.0
        for r in records:
            total += r.cost
            running.append(total * 1000)  # milli-dollars read better than 0.00x
        ax.plot(xs, running, color=color, linewidth=2, label=label)
        ax.annotate(f"{label}  ${total:.4f}", (len(records), running[-1]),
                    xytext=(-4, 6), textcoords="offset points", ha="right", color=INK)
    _style(ax, "Cumulative LLM cost", "cost (milli-dollars, paid tier)")
    ax.legend(frameon=False, loc="upper left")


def plot_latency(ax: plt.Axes, plain: list[Record], cached: list[Record]) -> None:
    """Latency of every request; cache hits sit two orders of magnitude lower."""
    xs = range(1, len(plain) + 1)
    ax.scatter(xs, [r.latency_ms for r in plain], s=18, color=NO_CACHE, label="no cache")
    ax.scatter(xs, [r.latency_ms for r in cached], s=18, color=CACHE, label="with cache")
    ax.set_yscale("log")
    _style(ax, "Latency per question", "milliseconds (log scale)")
    ax.legend(frameon=False, loc="center right")


def save_chart(plain: list[Record], cached: list[Record], saved: float, path: Path) -> None:
    """Draw both panels side by side and write the PNG."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#fcfcfb")
    plot_cost(left, plain, cached)
    plot_latency(right, plain, cached)
    fig.suptitle(f"Semantic cache: {saved:.0f}% cheaper on {len(plain)} support questions",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
