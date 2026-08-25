"""Render the leaderboard to the terminal and to output/leaderboard.md.

Inputs:  the dict from src/evaluate.build_rows
Outputs: a Rich table on stdout and a committed markdown artifact

The markdown file is tracked in git on purpose, so a reader can diff their run
against a known-good one before concluding they broke something.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from src.config import LEADERBOARD_PATH, LLM_MODEL, slm_label

console = Console()

# Declared before any run. Quoting a different operating point after seeing the
# results would be choosing the threshold to fit the headline.
CRITERION = (
    "Supported iff the lower bound of the 95% CI on quality retention is "
    ">= 0.90 while the LLM-call rate is <= 0.20."
)


def _rows_for_table(data: dict) -> list[dict]:
    return [data["baseline"], data["all_slm"], data["random"], data["oracle"], data["headline"]]


def render_terminal(data: dict) -> None:
    table = Table(
        title=f"SLM+LLM router  ·  n={data['n']}  ·  {data['stamp']}",
        pad_edge=False,
    )
    table.add_column("strategy", style="bold", no_wrap=True)
    table.add_column("acc", justify="right")
    table.add_column("95% CI", justify="right", no_wrap=True)
    table.add_column("cost $", justify="right")
    table.add_column("save", justify="right")
    table.add_column("LLM", justify="right")
    table.add_column("reten", justify="right")
    table.add_column("p50s", justify="right")

    for row in _rows_for_table(data):
        ci = "—" if row["ci"] is None else f"{row['ci'][0]:.0%}-{row['ci'][1]:.0%}"
        table.add_row(
            row["name"],
            f"{row['accuracy']:.1%}",
            ci,
            f"{row['cost_usd']:.5f}",
            f"{row['savings']:.0%}",
            f"{row['llm_rate']:.0%}",
            f"{row['retention']:.3f}",
            f"{row['p50']:.1f}" if row["p50"] else "—",
        )
    console.print(table)

    sweep = Table(title="deferral sweep · verifier strictness")
    sweep.add_column("strictness", justify="right")
    sweep.add_column("LLM rate", justify="right")
    sweep.add_column("accuracy", justify="right")
    sweep.add_column("cost", justify="right")
    sweep.add_column("verifier precision", justify="right")
    sweep.add_column("verifier recall", justify="right")
    for point in data["curve"]:
        conf = data["confusions"][point["strictness"]]
        sweep.add_row(
            str(point["strictness"]),
            f"{point['llm_rate']:.0%}",
            f"{point['accuracy']:.1%}",
            f"${point['cost_usd']:.5f}",
            f"{conf['precision']:.0%}",
            f"{conf['recall']:.0%}",
        )
    console.print(sweep)

    mix = Table(title="what sets the savings ceiling · by traffic slice")
    mix.add_column("traffic slice")
    mix.add_column("n", justify="right")
    mix.add_column("SLM acc", justify="right")
    mix.add_column("must escalate", justify="right")
    mix.add_column("max savings", justify="right")
    mix.add_column("cascade saves", justify="right")
    for row in data["mix"]:
        mix.add_row(
            row["slice"], str(row["n"]), f"{row['slm_accuracy']:.0%}",
            f"{row['must_escalate']:.0%}", f"{row['max_savings']:.0%}",
            f"{row['cascade_savings']:.0%}",
        )
    console.print(mix)

    buckets = Table(title="accuracy by difficulty")
    buckets.add_column("bucket")
    buckets.add_column("n", justify="right")
    buckets.add_column("SLM", justify="right")
    buckets.add_column("LLM", justify="right")
    buckets.add_column("gap", justify="right")
    for level in ("easy", "medium", "hard"):
        stats = data["by_category"].get(level)
        if stats:
            buckets.add_row(
                level, str(stats["n"]), f"{stats['slm']:.0%}",
                f"{stats['llm']:.0%}", f"{stats['llm'] - stats['slm']:+.0%}",
            )
    console.print(buckets)


def verdict(data: dict) -> tuple[bool, str]:
    """Apply the pre-registered criterion to the headline operating point."""
    row = data["headline"]
    retention_lo = row["retention_ci"][0]
    passed = retention_lo >= 0.90 and row["llm_rate"] <= 0.20
    detail = (
        f"retention {row['retention']:.3f} "
        f"(95% CI {row['retention_ci'][0]:.3f}-{row['retention_ci'][1]:.3f}), "
        f"LLM-call rate {row['llm_rate']:.0%}"
    )
    return passed, detail


def render_markdown(data: dict) -> None:
    passed, detail = verdict(data)
    lines: list[str] = [
        "# SLM+LLM router leaderboard",
        "",
        f"- items: **{data['n']}**",
        f"- dataset: `{data['stamp']}`",
        f"- small tier: `{slm_label()}`",
        f"- large tier: `{LLM_MODEL}`",
        f"- headline operating point: verifier strictness "
        f"**{data['headline_strictness']}**",
        "",
        "## Pre-registered criterion",
        "",
        f"> {CRITERION}",
        "",
        f"**{'MET' if passed else 'NOT MET'}** — {detail}",
        "",
        "## Strategies",
        "",
        "| strategy | accuracy | 95% CI | cost (USD) | savings | LLM rate | retention | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _rows_for_table(data):
        ci = "—" if row["ci"] is None else f"{row['ci'][0]:.0%}–{row['ci'][1]:.0%}"
        p = "—" if row["mcnemar_p"] is None else f"{row['mcnemar_p']:.3f}"
        lines.append(
            f"| {row['name']} | {row['accuracy']:.1%} | {ci} | "
            f"${row['cost_usd']:.5f} | {row['savings']:.1%} | {row['llm_rate']:.0%} | "
            f"{row['retention']:.3f} | {p} |"
        )
    lines.append("")
    lines.append(
        "The random row is an expectation over "
        "1,000 draws, not a realised per-item vector, so a confidence interval "
        "and a paired p-value are not defined for it."
    )

    lines += [
        "",
        "## Deferral sweep",
        "",
        "| strictness | LLM rate | accuracy | cost (USD) | verifier precision | verifier recall |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for point in data["curve"]:
        conf = data["confusions"][point["strictness"]]
        lines.append(
            f"| {point['strictness']} | {point['llm_rate']:.0%} | "
            f"{point['accuracy']:.1%} | ${point['cost_usd']:.5f} | "
            f"{conf['precision']:.0%} | {conf['recall']:.0%} |"
        )

    lines += [
        "",
        "## What sets the savings ceiling",
        "",
        "A router that holds baseline quality has to escalate at least as often "
        "as the small model is wrong — and the items it escalates are the "
        "expensive ones, so they eat a bigger share of the bill than of the "
        "traffic. `max savings` is the most that perfect routing could save "
        "**while still matching baseline accuracy**. It is a property of your "
        "traffic and your small model, not of your router.",
        "",
        "The cascade can exceed `max savings` on a slice, as it does on easy-only. "
        "That is not a better router — it is a router escalating less than it "
        "should and buying the difference in accuracy.",
        "",
        "| traffic slice | n | SLM accuracy | must escalate | max savings | cascade savings | cascade accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["mix"]:
        lines.append(
            f"| {row['slice']} | {row['n']} | {row['slm_accuracy']:.0%} | "
            f"{row['must_escalate']:.0%} | {row['max_savings']:.0%} | "
            f"{row['cascade_savings']:.0%} | {row['cascade_accuracy']:.0%} |"
        )

    lines += [
        "",
        "## Accuracy by difficulty",
        "",
        "| bucket | n | SLM | LLM | gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for level in ("easy", "medium", "hard"):
        stats = data["by_category"].get(level)
        if stats:
            lines.append(
                f"| {level} | {stats['n']} | {stats['slm']:.0%} | "
                f"{stats['llm']:.0%} | {stats['llm'] - stats['slm']:+.0%} |"
            )

    lines += [
        "",
        "## Reading this table",
        "",
        "`savings` is not a property of the router. With a local small tier at "
        "$0, savings is `1 - LLM-call-rate`, so the random control reports the "
        "same savings as the cascade at the same rate. What the cascade has to "
        "earn is the **accuracy** at that rate — the gap between its row and "
        "the random row, bounded above by the oracle row.",
        "",
    ]

    LEADERBOARD_PATH.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]wrote[/green] {LEADERBOARD_PATH}")


if __name__ == "__main__":
    from src.evaluate import build_rows
    from src.matrix import load

    data = build_rows(load())
    render_terminal(data)
    render_markdown(data)
