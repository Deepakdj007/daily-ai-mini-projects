"""Calibration gate: is there actually a capability gap to route across?

Inputs:  data/queries.json, a sample size, optionally --slm-only
Outputs: per-difficulty accuracy for each tier, and the gap between them

Run this BEFORE authoring or trusting anything else. A router is only worth
building when the small model fails on a bucket the big model handles. If the
two tiers score the same on the hard items, there is no gap to exploit and
every downstream number would be measuring noise. Better to discover that on
15 items than after a full run.
"""

from __future__ import annotations

import argparse
import json

from src.config import QUERIES_PATH, slm_label, LLM_MODEL, GROQ_API_KEY
from src.grader import grade
from src.tiers import call_llm, call_slm, preflight


def load_items() -> list[dict]:
    with QUERIES_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def sample(items: list[dict], per_bucket: int) -> list[dict]:
    """Take the first N items of each difficulty - deterministic, no seed."""
    out: list[dict] = []
    for level in ("easy", "medium", "hard"):
        bucket = [i for i in items if i["difficulty"] == level]
        out.extend(bucket[:per_bucket])
    return out


def run(items: list[dict], slm_only: bool) -> dict[str, dict[str, list[bool]]]:
    """Score both tiers on every item, grouped by difficulty."""
    scores: dict[str, dict[str, list[bool]]] = {
        "slm": {"easy": [], "medium": [], "hard": []},
        "llm": {"easy": [], "medium": [], "hard": []},
    }

    for index, item in enumerate(items, 1):
        level = item["difficulty"]
        slm = call_slm(item["query"])
        slm_ok = grade(slm["answer"], item["gold"])
        scores["slm"][level].append(slm_ok)

        llm_ok = None
        if not slm_only:
            llm = call_llm(item["query"])
            llm_ok = grade(llm["answer"], item["gold"])
            scores["llm"][level].append(llm_ok)

        mark = "OK " if slm_ok else "MISS"
        extra = "" if llm_ok is None else f"  llm={'OK ' if llm_ok else 'MISS'}"
        print(
            f"[{index:2d}/{len(items)}] {level:6s} {item['id']:10s} "
            f"slm={mark}{extra}  ({slm['latency_s']:.1f}s)"
        )

    return scores


def report(scores: dict[str, dict[str, list[bool]]], slm_only: bool) -> None:
    def pct(values: list[bool]) -> str:
        return f"{sum(values) / len(values):5.0%}" if values else "   - "

    print(f"\n{'bucket':8s} {'slm':>7s} {'llm':>7s} {'gap':>7s}")
    print("-" * 32)
    for level in ("easy", "medium", "hard"):
        slm_vals = scores["slm"][level]
        llm_vals = scores["llm"][level]
        gap = "   - "
        if slm_vals and llm_vals:
            gap = f"{(sum(llm_vals) / len(llm_vals)) - (sum(slm_vals) / len(slm_vals)):+5.0%}"
        print(f"{level:8s} {pct(slm_vals):>7s} {pct(llm_vals):>7s} {gap:>7s}")

    if slm_only:
        print("\n(slm-only run: set GROQ_API_KEY to measure the gap)")
        return

    hard_slm = scores["slm"]["hard"]
    hard_llm = scores["llm"]["hard"]
    if hard_slm and hard_llm:
        gap = (sum(hard_llm) / len(hard_llm)) - (sum(hard_slm) / len(hard_slm))
        print()
        if gap < 0.15:
            print(
                f"GATE FAILED: hard-bucket gap is only {gap:+.0%}. The small model "
                "keeps up with the big one here, so there is nothing for a router "
                "to exploit. Replace the hard bucket before going further."
            )
        else:
            print(f"GATE PASSED: hard-bucket gap is {gap:+.0%}. Enough to route across.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-bucket", type=int, default=5)
    parser.add_argument("--slm-only", action="store_true")
    args = parser.parse_args()

    slm_only = args.slm_only or not GROQ_API_KEY
    preflight()

    items = sample(load_items(), args.per_bucket)
    print(f"slm: {slm_label()}")
    print(f"llm: {LLM_MODEL if not slm_only else '(skipped)'}")
    print(f"items: {len(items)}\n")

    report(run(items, slm_only), slm_only)
