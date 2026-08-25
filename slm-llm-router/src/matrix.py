"""Build the answer matrix: every item run through every tier, once.

Inputs:  data/queries.json
Outputs: output/results.json - one record per item holding both tiers' answers,
         their token counts, costs, latencies, and correctness.

This is the only module that spends money, and it is the reason the rest of the
project is free. Once each item has been answered by both tiers and by the
self-consistency samples, every routing strategy is just a different way of
choosing which stored answer to keep. Strategies, controls, and the whole
strictness sweep then run offline in milliseconds and cost nothing.

That design also removes a confound: every strategy sees the identical answer
for a given item, so differences between strategies come from routing decisions
alone and never from re-sampling a model.
"""

from __future__ import annotations

import json

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from src.config import QUERIES_PATH, RESULTS_PATH, SELF_CONSISTENCY_K
from src.grader import grade
from src.tiers import call_llm, call_slm

console = Console()


def load_items(limit: int | None = None) -> list[dict]:
    """Load the frozen eval set, optionally truncated for a smoke run.

    A truncated run samples each difficulty bucket in proportion, because the
    point of a smoke run is to exercise the routing logic - and routing only
    happens when some items are hard. Taking the first N, or even every Nth,
    returns easy items only and makes a broken cascade look healthy.
    """
    with QUERIES_PATH.open(encoding="utf-8") as handle:
        items = json.load(handle)
    if limit is None or limit >= len(items):
        return items

    levels = ("easy", "medium", "hard")
    buckets = {lv: [i for i in items if i["difficulty"] == lv] for lv in levels}

    # Guarantee at least one item per bucket, then hand the remaining budget out
    # in proportion. Without the guarantee a small limit spends itself entirely
    # on easy items and the cascade never has anything to escalate.
    shares = {lv: 1 for lv in levels if buckets[lv]}
    remaining = limit - sum(shares.values())
    for level in levels:
        if remaining <= 0:
            break
        extra = min(
            remaining,
            round(limit * len(buckets[level]) / len(items)),
            len(buckets[level]) - shares.get(level, 0),
        )
        shares[level] = shares.get(level, 0) + max(0, extra)
        remaining = limit - sum(shares.values())

    picked: list[dict] = []
    for level in levels:
        bucket = buckets[level]
        share = min(shares.get(level, 0), len(bucket))
        if share:
            stride = max(1, len(bucket) // share)
            picked.extend(bucket[::stride][:share])
    return picked


def build_record(item: dict) -> dict:
    """Run one item through both tiers plus the self-consistency samples."""
    slm = call_slm(item["query"])
    slm_correct = grade(slm["answer"], item["gold"])

    # Extra draws at a higher temperature. The first sample IS the main answer,
    # so only k-1 additional calls are made.
    samples = [slm["answer"]]
    sample_cost = 0.0
    sample_latency = 0.0
    for index in range(1, SELF_CONSISTENCY_K):
        extra = call_slm(item["query"], sample=index)
        samples.append(extra["answer"])
        sample_cost += extra["cost_usd"]
        sample_latency += extra["latency_s"]

    llm = call_llm(item["query"])
    llm_correct = grade(llm["answer"], item["gold"])

    return {
        "id": item["id"],
        "category": item["category"],
        "difficulty": item["difficulty"],
        "gold": item["gold"],
        "numeric": item.get("numeric", False),
        "choices": item.get("choices"),
        "source": item.get("source", "authored"),
        "slm_answer": slm["answer"],
        "slm_correct": slm_correct,
        "slm_cost": slm["cost_usd"],
        "slm_latency": slm["latency_s"],
        "slm_in_tok": slm["in_tok"],
        "slm_out_tok": slm["out_tok"],
        "samples": samples,
        "sample_cost": sample_cost,
        "sample_latency": sample_latency,
        "llm_answer": llm["answer"],
        "llm_correct": llm_correct,
        "llm_cost": llm["cost_usd"],
        "llm_latency": llm["latency_s"],
        "llm_in_tok": llm["in_tok"],
        "llm_out_tok": llm["out_tok"],
    }


def build(limit: int | None = None) -> list[dict]:
    """Build the full matrix, showing progress because this loop is slow."""
    items = load_items(limit)
    calls_per_item = SELF_CONSISTENCY_K + 1
    console.print(
        f"[bold]{len(items)} items[/bold] x {calls_per_item} calls "
        f"= {len(items) * calls_per_item} model calls "
        f"({len(items)} of them billed to Groq).\n"
        "Cached calls return instantly; a cold run on CPU can take 20-40 minutes."
    )

    records: list[dict] = []
    columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=console) as progress:
        task = progress.add_task("answering", total=len(items))
        for item in items:
            records.append(build_record(item))
            progress.advance(task)

    return records


def save(records: list[dict]) -> None:
    RESULTS_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load() -> list[dict]:
    if not RESULTS_PATH.exists():
        raise SystemExit(
            f"No answer matrix at {RESULTS_PATH}.\n"
            "Build it first:  PYTHONPATH=. uv run python -m src.main run"
        )
    with RESULTS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    records = build(limit=6)
    save(records)
    slm_acc = sum(r["slm_correct"] for r in records) / len(records)
    llm_acc = sum(r["llm_correct"] for r in records) / len(records)
    print(f"\nsmoke run: {len(records)} items")
    print(f"  slm accuracy : {slm_acc:.0%}")
    print(f"  llm accuracy : {llm_acc:.0%}")
    print(f"  llm cost     : ${sum(r['llm_cost'] for r in records):.6f}")
