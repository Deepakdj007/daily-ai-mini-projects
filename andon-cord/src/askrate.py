"""How often does the lead pull the cord on the same brief?

A single run cannot answer that. At temperature 0.0 this model flags a brief on
one sample and not the next, because "temperature 0" is not a promise of
determinism on batched hosted inference. So a single yes/no is an anecdote, and
the claim worth checking — asks on briefs that hide something, stays quiet on
briefs that do not — is a rate.

Only the lead is sampled. That decision is one model call made before any
searching, which is what makes a real rate affordable on a free tier.

Inputs:  data/briefs.json
Outputs: a terminal table and output/ask-rate.md
"""

from rich.console import Console
from rich.table import Table

from src import config
from src.compare import load_briefs
from src.nodes import lead_node


def _sample_one(brief: dict, samples: int) -> int:
    """Put one brief to the lead `samples` times and count the cord pulls."""
    asked = 0
    for _ in range(samples):
        plan = lead_node({"brief": brief["brief"], "clarifications": []})
        asked += bool(plan["needs_human"] and plan["question"])
    return asked


def ask_rate(samples: int = 5, use_all: bool = False) -> None:
    """Report, per brief, how often the lead stops to ask.

    Args:
        samples: how many times to put each brief to the lead.
        use_all: include the two briefs that are off by default.
    """
    config.require_keys()
    console = Console()
    briefs = load_briefs(use_all)

    table = Table(
        title=f"How often does the lead pull the cord? ({samples} samples each)"
    )
    for column in ("brief", "kind", "asked", "rate"):
        table.add_column(column)

    lines = [
        "# andon-cord ask rate",
        "",
        f"Model `{config.MODEL}`, temperature 0.0, {samples} samples per brief. "
        "Same prompt every time.",
        "",
        "| brief | kind | asked | rate |",
        "|---|---|---|---|",
    ]

    for brief in briefs:
        asked = _sample_one(brief, samples)
        rate = f"{asked / samples:.0%}"
        console.print(f"[dim]{brief['id']}: {asked}/{samples}[/dim]")
        table.add_row(brief["id"], brief["kind"], f"{asked}/{samples}", rate)
        lines.append(f"| {brief['id']} | {brief['kind']} | {asked}/{samples} | {rate} |")

    console.print(table)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "ask-rate.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    console.print(f"\nWritten to {config.OUTPUT_DIR / 'ask-rate.md'}")
