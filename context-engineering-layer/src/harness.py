"""Run the ladder, pace it against the daily ceiling, write the results.

Inputs:  a profile name and a model
Outputs: output/results-<model>.json, with a manifest of what produced it

The estimate comes first and it is not decorative. On a free tier where a warm
probe costs a full prefix, an unchecked run exhausts the day's tokens somewhere
in the middle of the fourth arm and leaves a half-finished scorecard. So the
harness prices the whole run against the local ledger, prints it, and stops
unless the headroom is there.

Everything is replayed from sqlite when it can be, so an interrupted run
resumes for free and a re-grade costs nothing at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from typing import Sequence

from rich.console import Console
from rich.table import Table

from src import cache, config, summarizer
from src.ladder import ALL_ARMS, BY_NAME, PROFILES, Arm
from src.layers.base import LayerDeps
from src.pipeline import make_deps
from src.probes import cache_key, run_probe
from src.scenario import assign_zones, build, spec_hash, window_reach_async
from src.types import Fact, ProbeResult

console = Console()

SAMPLE_PROBES = 3


def probes_for(arm: Arm, facts: Sequence[Fact]) -> list[Fact]:
    """Which probes this arm runs.

    `raw` runs a sample: it is rejected before dispatch every time, so more
    rows would add cost and no information. `no-pin` runs the doc probes only,
    which is the whole point of that control.
    """
    if arm.probes == "doc":
        return [f for f in facts if f.zone == "doc"]
    if arm.probes == "sample":
        return list(facts)[:SAMPLE_PROBES]
    return list(facts)


def estimate(arms: Sequence[Arm], facts: Sequence[Fact]) -> tuple[int, int]:
    """Price the run in tokens and requests, before spending any.

    Priced at FULL prefix per probe, because that is what the pre-flight
    measured: cached tokens are not exempt from the rate limit on this account,
    so the marginal-question arithmetic the plan was built on does not hold.
    """
    tokens = requests = 0
    for arm in arms:
        n = len(probes_for(arm, facts))
        if not arm.config.enabled("window") and arm.config.enabled("history"):
            continue  # rejected before dispatch, so it costs nothing
        prefix = arm.config.context_budget if arm.config.enabled("history") else 300
        tokens += int(n * (prefix * config.TOKENIZER_FUDGE + 40))
        requests += n
    return tokens, requests


async def run(
    profile: str = "full",
    model: str = "",
    *,
    only: Sequence[str] = (),
    limit: int = 0,
    yes: bool = False,
) -> list[ProbeResult]:
    """Run every arm in the profile and write the results."""
    model = model or config.CHAT_MODEL
    turns, facts, spec = build()
    facts = assign_zones(facts, await window_reach_async(turns))
    if limit:
        facts = facts[:limit]

    names = only or PROFILES.get(profile, PROFILES["full"])
    arms = [BY_NAME[n] for n in names]

    gold = [f.value for f in facts if f.zone not in ('negative', 'doc')]
    fixture = summarizer.load(gold)
    if fixture is None:
        raise SystemExit(
            "No summary fixture. Run: PYTHONPATH=. uv run python -m src.summarizer"
        )
    deps: LayerDeps = make_deps(
        summarize_fn=summarizer.as_fn(fixture), fudge=config.TOKENIZER_FUDGE
    )

    predicted, requests = estimate(arms, facts)
    spent, used_requests = cache.spent_today()
    headroom = config.TOKENS_PER_DAY - spent
    cached_rows = sum(
        1
        for arm in arms
        for f in probes_for(arm, facts)
        if cache.get(cache_key(arm, f, model, spec_hash(), str(fixture.get("hash", ""))))
    )

    console.print(
        f"[bold]{profile}[/bold] on {model}: {len(arms)} arms, {len(facts)} probes\n"
        f"  estimate     {predicted:,} tok / {requests} requests "
        f"(priced at full prefix - cached tokens are not exempt here)\n"
        f"  already run  {cached_rows} rows replay from sqlite at zero cost\n"
        f"  spent today  {spent:,} of {config.TOKENS_PER_DAY:,} tok, "
        f"{used_requests} of {config.REQUESTS_PER_DAY} requests\n"
        f"  headroom     {headroom:,} tok"
    )
    if predicted > headroom and not yes:
        raise SystemExit(
            f"\nThis run needs {predicted:,} tok and {headroom:,} remain today.\n"
            "  Use a smaller profile (--profile lite), --limit N, or wait for the\n"
            "  daily refill. Nothing already cached will be re-spent."
        )
    if not yes:
        console.print("  [dim]--yes to proceed[/dim]")
        return []

    results: list[ProbeResult] = []
    for arm in arms:
        arm_facts = probes_for(arm, facts)
        console.print(f"\n[bold]{arm.name}[/bold] ({arm.label}) - {len(arm_facts)} probes")
        for fact in arm_facts:
            result = await run_probe(
                turns, fact, arm, deps, model=model,
                system_prompt=spec["system_prompt"], pins=spec["pinned_facts"],
                scenario_hash=spec_hash(), summary_hash=str(fixture.get("hash", "")),
            )
            results.append(result)
            flag = "R" if result.replayed else " "
            console.print(
                f"  {flag} {result.probe_id:<4} {result.zone:<8} "
                f"present={str(result.fact_present):<5} {result.status}",
                highlight=False,
            )

    write(results, model, spec, fixture)
    return results


def write(results: Sequence[ProbeResult], model: str, spec: dict,
          fixture: dict) -> None:
    """Persist results plus everything needed to trace them back."""
    payload = {
        "manifest": {
            "date": date.today().isoformat(),
            "model": model,
            "scenario": spec_hash(),
            "summary_fixture": fixture.get("hash"),
            "facts_in_summary": fixture.get("facts_in_summary"),
            "context_budget": config.CONTEXT_BUDGET_TOKENS,
            "max_completion_tokens": config.MAX_COMPLETION_TOKENS,
            "temperature": config.TEMPERATURE,
            "tokenizer": config.TOKENIZER_NAME,
            "tokenizer_fudge": config.TOKENIZER_FUDGE,
            "prompt_version": config.PROMPT_VERSION,
            "assembler_version": config.ASSEMBLER_VERSION,
            "cached_tokens_rate_limit_exempt": config.CACHED_TOKENS_ARE_RATE_LIMIT_EXEMPT,
        },
        "results": [
            {**asdict(r), "usage": asdict(r.usage), "contaminated": r.contaminated}
            for r in results
        ],
    }
    path = config.results_path(model)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    console.print(f"\nwrote {len(results)} rows to {path.name}")


def load(model: str = "") -> tuple[dict, list[dict]]:
    """Read a results file back, for report.py and plot.py."""
    path = config.results_path(model or config.CHAT_MODEL)
    if not path.exists():
        raise SystemExit(f"No results at {path}. Run: python -m src.main run --yes")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["manifest"], data["results"]


if __name__ == "__main__":
    from src.scenario import window_reach

    turns, facts, spec = build()
    facts = assign_zones(facts, window_reach(turns))
    table = Table(title="what each profile would cost, priced at full prefix")
    for col in ("profile", "arms", "probe rows", "tokens", "requests"):
        table.add_column(col, justify="right")
    for name, arm_names in PROFILES.items():
        arms = [BY_NAME[n] for n in arm_names]
        tok, req = estimate(arms, facts)
        rows = sum(len(probes_for(a, facts)) for a in arms)
        table.add_row(name, str(len(arms)), str(rows), f"{tok:,}", str(req))
    console.print(table)
    spent, reqs = cache.spent_today()
    console.print(f"spent today {spent:,} tok / {reqs} requests; "
                  f"headroom {config.TOKENS_PER_DAY - spent:,} tok")
