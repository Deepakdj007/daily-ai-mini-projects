"""Render a Trace in the terminal with Rich: step tree, passage table, diagnosis, answer.

Inputs:  a Trace (and optionally the judge-vs-hand-label agreement)
Outputs: coloured terminal output

The terminal view exists for the demo. The Trace is plain JSON on disk,
so the same data could feed a web UI without changing the pipeline.
"""

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from src.tracer import Trace

console = Console()

ROW_STYLE = {"HIT": "green", "NOISE": "red", "MISS": "bold yellow", "CUT": "dim", "UNJUDGED": "dim italic"}
HEADLINE_STYLE = {"HEALTHY": "green", "NOISY_CONTEXT": "red", "RETRIEVAL_MISS": "yellow",
                  "CORPUS_GAP": "magenta", "JUDGE_UNAVAILABLE": "dim"}


def step_tree(trace: Trace) -> Tree:
    """The timeline: each step, its duration and what it recorded."""
    total = sum(step.ms for step in trace.steps)
    tree = Tree(f"[bold]trace[/] {total:.0f} ms  [dim]{escape(trace.query)}[/]")
    for step in trace.steps:
        facts = "  ".join(f"{key}={value}" for key, value in step.summary.items())
        tree.add(f"[cyan]{step.name:<9}[/] {step.ms:>7.1f} ms  [dim]{escape(facts)}[/]")
    return tree


def passage_table(trace: Trace) -> Table:
    """One row per passage in the pool, coloured by its cell."""
    table = Table(show_lines=False, header_style="bold", expand=True)
    table.add_column("#", justify="right", width=2)
    table.add_column("passage", ratio=3, overflow="fold")
    table.add_column("sim", justify="right", width=5)
    table.add_column("prompt?", width=7)
    table.add_column("retriever why", ratio=1, overflow="fold")
    table.add_column("judge", width=9)
    table.add_column("judge why", ratio=2, overflow="fold")
    table.add_column("cell", width=8)
    # Passage text, judge reasons and answers are data, not Rich markup: a "[/b]" in any of them
    # would otherwise raise MarkupError, and "[see policy]" would silently vanish.
    for row in trace.rows:
        snippet = row["text"] if len(row["text"]) <= 70 else row["text"][:67] + "..."
        table.add_row(
            str(row["rank"]), f"[bold]{row['pid']}[/] {escape(snippet)}", f"{row['similarity']:.2f}",
            "yes" if row["in_prompt"] else "no", row["retriever_why"], row["label"],
            escape(row["judge_why"]), row["cell"], style=ROW_STYLE[row["cell"]],
        )
    return table


def score(value: float | None) -> str:
    """Show a metric to two places, or n/a when it can't be computed."""
    return "n/a" if value is None else f"{value:.2f}"


def diagnosis_panel(trace: Trace, agreement: dict | None = None) -> Panel:
    """The headline, what it means, the two context metrics and where the trace was saved."""
    diag, m = trace.diagnosis, trace.diagnosis["metrics"]
    lines = [
        f"[bold {HEADLINE_STYLE[diag['headline']]}]{diag['headline']}[/]  {diag['detail']}",
        f"context precision {score(m['precision'])}   context recall {score(m['recall'])}   "
        f"HIT {m['hit']}  NOISE {m['noise']}  MISS {m['miss']}",
    ]
    if agreement:
        extra = f"  (disagreed on {', '.join(agreement['disagreed_on'])})" if agreement["disagreed_on"] else ""
        lines.append(f"judge vs hand labels: {agreement['agree']}/{agreement['total']} agree{extra}")
    lines.append(f"[dim]trace saved to {escape(trace.saved_to)}[/]")
    return Panel("\n".join(lines), title="diagnosis", border_style=HEADLINE_STYLE[diag["headline"]])


def render(trace: Trace, agreement: dict | None = None) -> None:
    """Print the full debugger view for one query."""
    console.print(step_tree(trace))
    console.print(passage_table(trace))
    console.print(diagnosis_panel(trace, agreement))
    console.print(Panel(escape(trace.answer), title="what the pipeline answered", border_style="blue"))
