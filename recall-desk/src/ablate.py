"""The ablation harness — five configs, four probes, a scorecard.

Method: ablate the READ, not the write.

Every config starts from an identical seeded store and differs only in which
tiers the recall step is allowed to look at. Ablating writes instead would let
the configs drift apart through many uncontrolled paths; ablating reads moves
exactly one variable, and makes the whole run reproducible and cheap.

Each probe runs on a fresh thread_id with the checkpointer wired and working. So
there is never any thread history to fall back on, and a probe failure cannot be
explained away as "it fell out of the context window."

Usage:
    PYTHONPATH=. uv run python -m src.ablate                 # 2 repeats
    PYTHONPATH=. uv run python -m src.ablate --repeats 3
    PYTHONPATH=. uv run python -m src.ablate --config "all on"
    PYTHONPATH=. uv run python -m src.ablate --probe A   # one probe, all configs

The last two exist because Groq's free tier is 200k tokens/day org-wide, and a
full grid is not free. Narrow the run while iterating; widen it to publish.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from rich.console import Console
from rich.table import Table

from src.agent import ask_desk, build_graph, open_checkpointer
from src.config import ABLATION_CONFIGS, MemoryConfig
from src.probes import PROBES, Probe, context_leak
from src.seed import seed
from src.store import open_store

console = Console()

# Rough per-call cost, measured from real runs during the build. Used only for
# the pre-flight estimate; actual usage is reported from the API afterwards.
EST_TOKENS_PER_CALL = 1_700


def _tier_enabled(memory: MemoryConfig, tier: str) -> bool:
    """Whether the tier a probe measures is switched on in this config."""
    return {
        "semantic": memory.semantic,
        "episodic": memory.episodic,
        "procedural": memory.procedural,
        "none": True,
    }[tier]


def run_probe(graph, probe: Probe, memory: MemoryConfig, run_index: int) -> dict[str, Any]:
    """Run one probe under one config.

    Returns:
        A record with the checks, the reply, token usage, and any context leak.
    """
    state = ask_desk(
        graph,
        probe.customer_id,
        probe.ticket,
        # Fresh thread every single time. Nothing carries over but the store.
        thread_id=f"ablate-{memory.label}-{probe.key}-{run_index}",
        memory=memory,
        learn=False,  # the store is pre-seeded; this run only reads
    )
    checks = probe.score(state)
    tokens = (state.get("trace") or {}).get("tokens", {})
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "reply": state["reply"],
        "tokens": tokens.get("input", 0) + tokens.get("output", 0),
        "leak": context_leak(state, probe) if not _tier_enabled(memory, probe.measures) else [],
    }


def run(repeats: int = 2, only: str | None = None, probe_key: str | None = None) -> dict:
    """Run the grid and print the scorecard.

    Args:
        repeats: runs per config/probe pair. Pass rates come from these.
        only: restrict to one config label, to stay inside a token budget.
        probe_key: restrict to one probe, for iterating on a single assertion.
    """
    configs = [c for c in ABLATION_CONFIGS if only is None or c.label == only]
    if not configs:
        labels = ", ".join(f"{c.label!r}" for c in ABLATION_CONFIGS)
        raise SystemExit(f"no config named {only!r}. Options: {labels}")

    probes = [p for p in PROBES if probe_key is None or p.key == probe_key]
    if not probes:
        raise SystemExit(f"no probe named {probe_key!r}. Options: {[p.key for p in PROBES]}")

    calls = len(configs) * len(probes) * repeats
    console.print(
        f"[bold]{len(configs)} configs x {len(probes)} probes x {repeats} repeats "
        f"= {calls} calls[/bold]"
    )
    console.print(
        f"rough estimate {calls * EST_TOKENS_PER_CALL // 1000}k tokens. "
        "Groq's free tier is 200k/day org-wide.\n"
    )

    console.print("[dim]seeding an identical store for every config...[/dim]")
    seed(fresh=True)
    store = open_store()
    # The checkpointer is deliberately wired even though every probe uses a new
    # thread. Its presence is what rules out the context window as an excuse.
    graph = build_graph(store, open_checkpointer())

    results: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    errors: list[str] = []
    total_tokens = 0

    for memory in configs:
        for probe in probes:
            for i in range(repeats):
                try:
                    record = run_probe(graph, probe, memory, i)
                except Exception as exc:  # a 429 or a bad generation
                    errors.append(f"{memory.label} / {probe.key} / run {i}: {exc}")
                    continue
                results[memory.label][probe.key].append(record)
                total_tokens += record["tokens"]
        done = sum(len(v) for v in results[memory.label].values())
        console.print(f"  {memory.label:22} {done} runs")

    _scorecard(results, configs, probes, repeats)
    _leaks(results)
    _detail(results, configs, probes)

    console.print(f"\n[dim]actual tokens used: {total_tokens:,}[/dim]")
    if errors:
        console.print(f"\n[bold red]{len(errors)} run(s) errored[/bold red]")
        for err in errors:
            console.print(f"  {err}")
    return results


def _rate(records: list[dict]) -> str:
    """Pass rate across repeats, shown as a fraction rather than a verdict."""
    if not records:
        return "-"
    passed = sum(1 for r in records if r["passed"])
    return f"{passed}/{len(records)}"


def _scorecard(results, configs, probes, repeats: int) -> None:
    """The headline table: configs down the side, probes across the top."""
    table = Table(title=f"\nAblation scorecard ({repeats} repeats, pass rate)")
    table.add_column("config", style="bold")
    for probe in probes:
        table.add_column(f"{probe.key}\n({probe.measures})", justify="center")

    for memory in configs:
        row = [memory.label]
        for probe in probes:
            records = results[memory.label][probe.key]
            cell = _rate(records)
            expected_fail = not _tier_enabled(memory, probe.measures)
            if cell == "-":
                row.append(cell)
            elif expected_fail:
                # Failing here is the point, so a failure is the good outcome.
                row.append(f"[green]{cell}[/green]" if cell.startswith("0/") else f"[yellow]{cell}[/yellow]")
            else:
                ok = cell.split("/")[0] == cell.split("/")[1]
                row.append(f"[green]{cell}[/green]" if ok else f"[red]{cell}[/red]")
        table.add_row(*row)

    console.print(table)
    console.print(
        "[dim]For a probe whose tier is switched off, a LOW score is the expected "
        "result.\nGreen means the harness behaved as designed, not that the agent "
        "answered well.[/dim]"
    )


def _leaks(results) -> None:
    """Report any probe that could be answered from a tier that was off."""
    found = [
        (label, key, r["leak"])
        for label, probes in results.items()
        for key, records in probes.items()
        for r in records
        if r["leak"]
    ]
    if found:
        console.print("\n[bold red]BROKEN PROBES — answer leaked into context[/bold red]")
        for label, key, leak in found:
            console.print(f"  {label} / probe {key}: {leak}")
    else:
        console.print(
            "\n[green]no context leaks[/green] — with a tier off, its answer tokens "
            "were absent from the assembled prompt"
        )


def _detail(results, configs, probes) -> None:
    """Which individual checks failed, so degradation is legible per rule."""
    console.print("\n[bold]Per-check detail[/bold]")
    for probe in probes:
        console.print(f"\n  probe {probe.key} ({probe.measures})")
        for memory in configs:
            records = results[memory.label][probe.key]
            if not records:
                continue
            tally: dict[str, int] = defaultdict(int)
            for record in records:
                for name, ok in record["checks"].items():
                    tally[name] += int(ok)
            bits = [f"{name} {n}/{len(records)}" for name, n in tally.items()]
            console.print(f"    {memory.label:22} " + " | ".join(bits))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate each memory tier and score the result.")
    parser.add_argument("--repeats", type=int, default=2, help="runs per config/probe pair")
    parser.add_argument("--config", type=str, default=None, help="run only this config label")
    parser.add_argument("--probe", type=str, default=None, help="run only this probe (A/B/C/control)")
    args = parser.parse_args()
    run(repeats=args.repeats, only=args.config, probe_key=args.probe)


if __name__ == "__main__":
    main()
