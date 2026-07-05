"""Aggregates experiment results into a leaderboard and renders it two ways.

Inputs: the list of ExperimentResult objects returned by run_all_configs().
Outputs: a Rich table printed to the terminal and a markdown table written
to leaderboard.md, both ranked by the llm_judge accuracy score.
"""

import statistics
from dataclasses import dataclass
from typing import Any

from langfuse import get_client
from rich.console import Console
from rich.table import Table

from src.config import LEADERBOARD_PATH


@dataclass
class ConfigStats:
    """Aggregated metrics for a single config's experiment run."""

    name: str
    accuracy_pct: float
    exact_match_pct: float
    avg_cost_usd: float
    total_cost_usd: float
    avg_latency_s: float
    p50_latency_s: float
    sample_trace_url: str | None


def _scores_for(item_results: list[Any], eval_name: str) -> list[float]:
    """Pull every numeric score with a given evaluation name across all items."""
    return [
        evaluation.value
        for result in item_results
        for evaluation in result.evaluations
        if evaluation.name == eval_name
    ]


def summarize(experiment_result: Any) -> ConfigStats:
    """Reduce one ExperimentResult down to the metrics shown on the leaderboard.

    Local (non-dataset) experiments have no `dataset_run_url`, so a link to the
    first item's trace is used instead as a jumping-off point into the Langfuse UI.
    """
    item_results = experiment_result.item_results
    accuracy = _scores_for(item_results, "llm_judge")
    exact_match = _scores_for(item_results, "exact_match")
    cost = _scores_for(item_results, "cost_usd")
    latency = _scores_for(item_results, "latency_s")

    sample_trace_url = None
    if item_results and item_results[0].trace_id:
        sample_trace_url = get_client().get_trace_url(trace_id=item_results[0].trace_id)

    return ConfigStats(
        name=experiment_result.name,
        accuracy_pct=100 * sum(accuracy) / len(accuracy) if accuracy else 0.0,
        exact_match_pct=100 * sum(exact_match) / len(exact_match) if exact_match else 0.0,
        avg_cost_usd=sum(cost) / len(cost) if cost else 0.0,
        total_cost_usd=sum(cost),
        avg_latency_s=sum(latency) / len(latency) if latency else 0.0,
        p50_latency_s=statistics.median(latency) if latency else 0.0,
        sample_trace_url=sample_trace_url,
    )


def _build_rich_table(stats: list[ConfigStats]) -> Table:
    """Render the leaderboard as a Rich table, ranked by judge accuracy."""
    table = Table(title="Agent Eval Arena Leaderboard")
    table.add_column("Config")
    table.add_column("Accuracy (judge)", justify="right")
    table.add_column("Exact match", justify="right")
    table.add_column("Avg cost/task", justify="right")
    table.add_column("Total cost", justify="right")
    table.add_column("Avg latency", justify="right")
    table.add_column("p50 latency", justify="right")

    for s in sorted(stats, key=lambda s: s.accuracy_pct, reverse=True):
        table.add_row(
            s.name,
            f"{s.accuracy_pct:.1f}%",
            f"{s.exact_match_pct:.1f}%",
            f"${s.avg_cost_usd:.6f}",
            f"${s.total_cost_usd:.6f}",
            f"{s.avg_latency_s:.2f}s",
            f"{s.p50_latency_s:.2f}s",
        )
    return table


def _build_markdown(stats: list[ConfigStats]) -> str:
    """Render the same leaderboard as a markdown table for leaderboard.md."""
    lines = [
        "# Agent Eval Arena Leaderboard",
        "",
        "| Config | Accuracy (judge) | Exact match | Avg cost/task | Total cost | Avg latency | p50 latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in sorted(stats, key=lambda s: s.accuracy_pct, reverse=True):
        lines.append(
            f"| {s.name} | {s.accuracy_pct:.1f}% | {s.exact_match_pct:.1f}% | "
            f"${s.avg_cost_usd:.6f} | ${s.total_cost_usd:.6f} | "
            f"{s.avg_latency_s:.2f}s | {s.p50_latency_s:.2f}s |"
        )
    lines.append("")
    for s in stats:
        if s.sample_trace_url:
            lines.append(f"- [{s.name} sample trace]({s.sample_trace_url})")
    return "\n".join(lines) + "\n"


def publish_leaderboard(experiment_results: list[Any]) -> None:
    """Print the leaderboard to the terminal and persist it to leaderboard.md."""
    stats = [summarize(result) for result in experiment_results]
    Console().print(_build_rich_table(stats))
    LEADERBOARD_PATH.write_text(_build_markdown(stats), encoding="utf-8")
