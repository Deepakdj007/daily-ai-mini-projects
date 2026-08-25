"""Score every strategy and assemble the leaderboard.

Inputs:  output/results.json (the answer matrix)
Outputs: a dict of computed rows, plus the dataset provenance stamp

Kept separate from rendering so the numbers can be inspected without going
through a table.
"""

from __future__ import annotations

import hashlib
import subprocess

from src.config import HEADLINE_STRICTNESS, QUERIES_PATH, STRICTNESS_LEVELS
from src.stats import mcnemar_exact, paired_bootstrap_ratio, bootstrap_ci
from src.strategies import (
    Result,
    all_slm,
    baseline,
    cascade,
    oracle_router,
    random_router,
    verifier_confusion,
)


def dataset_stamp() -> str:
    """Identify exactly which eval set produced these numbers.

    A leaderboard that does not name its dataset version is unfalsifiable: any
    later edit to the questions silently rewrites history. The content hash is
    always available; the git commit is added when the file has been frozen.
    """
    digest = hashlib.sha256(QUERIES_PATH.read_bytes()).hexdigest()[:12]
    try:
        commit = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", str(QUERIES_PATH)],
            capture_output=True, text=True, timeout=10, cwd=QUERIES_PATH.parent,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        commit = ""
    return f"sha256:{digest}" + (f" git:{commit}" if commit else " git:UNCOMMITTED")


def build_rows(records: list[dict]) -> dict:
    """Compute every strategy, control, and derived statistic."""
    base = baseline(records)
    floor = all_slm(records)

    cascades = [cascade(records, level) for level in STRICTNESS_LEVELS]
    headline = cascade(records, HEADLINE_STRICTNESS)

    # Controls are drawn at the headline cascade's own routing rate, so the
    # comparison is at matched cost rather than at a flattering operating point.
    rate = headline.llm_rate
    rnd = random_router(records, rate)
    orc = oracle_router(records, rate)

    def summarise(result: Result) -> dict:
        savings = 1 - (result.cost_usd / base.cost_usd) if base.cost_usd else 0.0

        # A synthetic row carries an expected accuracy, not a realised per-item
        # vector, so a CI or a paired p-value computed from it would describe
        # arithmetic rather than evidence.
        if result.synthetic:
            return {
                "name": result.name,
                "accuracy": result.accuracy,
                "ci": None,
                "cost_usd": result.cost_usd,
                "savings": savings,
                "llm_rate": result.llm_rate,
                "retention": result.accuracy / base.accuracy if base.accuracy else 0.0,
                "retention_ci": None,
                "mcnemar_p": None,
                "discordant": None,
                "p50": result.p50_latency,
                "max": result.max_latency,
                "detail": result.detail,
                "synthetic": True,
            }

        lo, hi = bootstrap_ci(result.outcomes)
        ratio, r_lo, r_hi = paired_bootstrap_ratio(result.outcomes, base.outcomes)
        b, c, p = mcnemar_exact(result.outcomes, base.outcomes)
        return {
            "name": result.name,
            "accuracy": result.accuracy,
            "ci": (lo, hi),
            "cost_usd": result.cost_usd,
            "savings": savings,
            "llm_rate": result.llm_rate,
            "retention": ratio,
            "retention_ci": (r_lo, r_hi),
            "mcnemar_p": p,
            "discordant": (b, c),
            "p50": result.p50_latency,
            "max": result.max_latency,
            "detail": result.detail,
            "synthetic": False,
        }

    return {
        "n": len(records),
        "stamp": dataset_stamp(),
        "baseline": summarise(base),
        "all_slm": summarise(floor),
        "cascades": [summarise(c) for c in cascades],
        "headline": summarise(headline),
        "headline_strictness": HEADLINE_STRICTNESS,
        "random": summarise(rnd),
        "oracle": summarise(orc),
        "confusions": {
            level: verifier_confusion(records, level) for level in STRICTNESS_LEVELS
        },
        "curve": [
            {
                "strictness": level,
                "llm_rate": c.llm_rate,
                "accuracy": c.accuracy,
                "cost_usd": c.cost_usd,
            }
            for level, c in zip(STRICTNESS_LEVELS, cascades)
        ],
        "chord": {
            "slm": (0.0, floor.accuracy),
            "llm": (1.0, base.accuracy),
        },
        "oracle_curve": [
            {
                "llm_rate": r,
                "accuracy": oracle_router(records, r).accuracy,
            }
            for r in [i / 20 for i in range(21)]
        ],
        "random_curve": [
            {
                "llm_rate": r,
                "accuracy": floor.accuracy + r * (base.accuracy - floor.accuracy),
            }
            for r in [i / 20 for i in range(21)]
        ],
        "by_category": category_breakdown(records),
        "mix": mix_sensitivity(records),
    }


def mix_sensitivity(records: list[dict]) -> list[dict]:
    """How much the headline depends on the traffic mix, not on the router.

    The savings ceiling is set by how often the small model is right, and that
    is a property of your traffic. Two hard numbers fall out of the data:

      * a router that holds baseline quality must escalate at least as often as
        the small model is wrong, and
      * the items it escalates are the expensive ones, so they eat a larger
        share of the bill than their share of the traffic.

    Together those cap the achievable saving well below `1 - escalation rate`.
    Re-slicing by difficulty shows the cap moving, which is the clearest way to
    say: routing pays off exactly to the degree your traffic is easy.
    """
    slices = [
        ("easy only", lambda r: r["difficulty"] == "easy"),
        ("easy + medium", lambda r: r["difficulty"] in ("easy", "medium")),
        ("full mix (60/25/15)", lambda _r: True),
        ("hard only", lambda r: r["difficulty"] == "hard"),
    ]

    out: list[dict] = []
    for label, keep in slices:
        subset = [r for r in records if keep(r)]
        if not subset:
            continue
        base_cost = sum(r["llm_cost"] for r in subset)
        slm_acc = sum(r["slm_correct"] for r in subset) / len(subset)
        wrong = [r for r in subset if not r["slm_correct"]]
        must_escalate = len(wrong) / len(subset)
        ceiling = 1 - (sum(r["llm_cost"] for r in wrong) / base_cost) if base_cost else 0.0

        casc = cascade(subset, HEADLINE_STRICTNESS)
        out.append({
            "slice": label,
            "n": len(subset),
            "slm_accuracy": slm_acc,
            "must_escalate": must_escalate,
            "max_savings": ceiling,
            "cascade_savings": 1 - (casc.cost_usd / base_cost) if base_cost else 0.0,
            "cascade_accuracy": casc.accuracy,
        })
    return out


def category_breakdown(records: list[dict]) -> dict[str, dict[str, float]]:
    """Per-category accuracy for both tiers.

    A single aggregate hides the only thing that matters here: whether the
    small model fails on a bucket the big model handles.
    """
    out: dict[str, dict[str, float]] = {}
    for record in records:
        bucket = out.setdefault(
            record["difficulty"], {"n": 0, "slm": 0, "llm": 0}
        )
        bucket["n"] += 1
        bucket["slm"] += record["slm_correct"]
        bucket["llm"] += record["llm_correct"]

    return {
        key: {
            "n": value["n"],
            "slm": value["slm"] / value["n"],
            "llm": value["llm"] / value["n"],
        }
        for key, value in out.items()
    }


if __name__ == "__main__":
    from src.matrix import load

    rows = build_rows(load())
    print(f"n = {rows['n']}   dataset {rows['stamp']}")
    for key in ("baseline", "all_slm", "headline", "random", "oracle"):
        row = rows[key]
        print(
            f"{row['name']:34s} acc={row['accuracy']:6.1%} "
            f"cost=${row['cost_usd']:.6f} savings={row['savings']:6.1%} "
            f"llm={row['llm_rate']:5.1%}"
        )
