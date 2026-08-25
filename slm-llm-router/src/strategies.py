"""Routing strategies, evaluated by replaying the cached answer matrix.

Inputs:  the records built by src/matrix.py
Outputs: a Result per strategy - outcomes, cost, LLM-call rate, latency

Nothing here calls a model. Every strategy picks between answers that already
exist, which is what makes the controls affordable: the random router averaged
over a thousand draws costs nothing but arithmetic.

The three controls matter more than the cascade itself. With a free local tier,
`savings% = 1 - LLM-call-rate`, so ANY policy that sends 10% of traffic to the
big model reports "90% cheaper" - including one that picks those 10% by coin
flip. The random control is what separates a router that thinks from a router
that merely defers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import SELF_CONSISTENCY_K
from src.stats import RNG_SEED
from src.verifier import DISAGREE, should_escalate

N_RANDOM_DRAWS = 1000


@dataclass
class Result:
    """One strategy's performance over the whole eval set."""

    name: str
    outcomes: list[bool]
    cost_usd: float
    llm_calls: int
    latencies: list[float] = field(default_factory=list)
    escalations: int = 0
    detail: str = ""
    # True when `outcomes` is an expectation reconstructed from a mean rather
    # than a real per-item vector. Paired tests are meaningless on such a row,
    # so the report blanks its CI and p-value instead of printing a number that
    # looks like evidence.
    synthetic: bool = False

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def accuracy(self) -> float:
        return sum(self.outcomes) / self.n if self.n else 0.0

    @property
    def llm_rate(self) -> float:
        return self.llm_calls / self.n if self.n else 0.0

    @property
    def p50_latency(self) -> float:
        return float(np.percentile(self.latencies, 50)) if self.latencies else 0.0

    @property
    def max_latency(self) -> float:
        return max(self.latencies) if self.latencies else 0.0


def baseline(records: list[dict]) -> Result:
    """Send everything to the big model. The cost and quality ceiling."""
    return Result(
        name="baseline (LLM)",
        outcomes=[r["llm_correct"] for r in records],
        cost_usd=sum(r["llm_cost"] for r in records),
        llm_calls=len(records),
        latencies=[r["llm_latency"] for r in records],
    )


def all_slm(records: list[dict]) -> Result:
    """Send everything to the small model. The cost and quality floor.

    This is the control that decides whether the eval set is worth anything.
    If it scores near baseline, the questions are too easy and every other row
    in the table is noise.
    """
    return Result(
        name="all SLM",
        outcomes=[r["slm_correct"] for r in records],
        cost_usd=sum(r["slm_cost"] for r in records),
        llm_calls=0,
        latencies=[r["slm_latency"] for r in records],
    )


def cascade(records: list[dict], strictness: int) -> Result:
    """Answer with the SLM, check locally, escalate when the check fails.

    Escalated items are paid for twice - the small model's tokens and latency
    are already spent when the big model is called. That double payment is real
    and is included here rather than quietly dropped.
    """
    outcomes: list[bool] = []
    latencies: list[float] = []
    cost = 0.0
    llm_calls = 0

    uses_samples = strictness >= DISAGREE

    for record in records:
        item = {"numeric": record["numeric"], "choices": record["choices"]}
        samples = record["samples"] if uses_samples else []
        escalate, _reason = should_escalate(
            item, record["slm_answer"], samples, strictness
        )

        spent = record["slm_cost"]
        elapsed = record["slm_latency"]
        if uses_samples:
            spent += record["sample_cost"]
            elapsed += record["sample_latency"]

        if escalate:
            llm_calls += 1
            spent += record["llm_cost"]
            elapsed += record["llm_latency"]
            outcomes.append(record["llm_correct"])
        else:
            outcomes.append(record["slm_correct"])

        cost += spent
        latencies.append(elapsed)

    label = f"cascade s{strictness}"
    if uses_samples:
        label += f" k{SELF_CONSISTENCY_K}"

    return Result(
        name=label,
        outcomes=outcomes,
        cost_usd=cost,
        llm_calls=llm_calls,
        latencies=latencies,
        escalations=llm_calls,
    )


def random_router(records: list[dict], llm_rate: float) -> Result:
    """Send a fixed fraction to the big model, chosen at random.

    This traces a straight chord from all-SLM to baseline. Any real router must
    bow above it to have earned its complexity.
    """
    n = len(records)
    k = int(round(llm_rate * n))
    slm_ok = np.array([r["slm_correct"] for r in records], dtype=float)
    llm_ok = np.array([r["llm_correct"] for r in records], dtype=float)
    slm_cost = np.array([r["slm_cost"] for r in records], dtype=float)
    llm_cost = np.array([r["llm_cost"] for r in records], dtype=float)

    rng = np.random.default_rng(RNG_SEED)
    accs = np.empty(N_RANDOM_DRAWS)
    costs = np.empty(N_RANDOM_DRAWS)
    for draw in range(N_RANDOM_DRAWS):
        picked = rng.choice(n, size=k, replace=False) if k else np.array([], dtype=int)
        mask = np.zeros(n, dtype=bool)
        mask[picked] = True
        accs[draw] = np.where(mask, llm_ok, slm_ok).mean()
        costs[draw] = np.where(mask, llm_cost, slm_cost).sum()

    # Outcomes are an expectation, not a realised vector, so paired tests do
    # not apply to this row - it is a reference line, not a competitor.
    mean_acc = float(accs.mean())
    outcomes = [True] * int(round(mean_acc * n)) + [False] * (n - int(round(mean_acc * n)))

    return Result(
        name=f"random@{llm_rate:.0%}",
        outcomes=outcomes,
        cost_usd=float(costs.mean()),
        llm_calls=k,
        latencies=[],
        detail=f"mean of {N_RANDOM_DRAWS} draws",
        synthetic=True,
    )


def oracle_router(records: list[dict], llm_rate: float) -> Result:
    """Spend the LLM budget perfectly: escalate only what the SLM got wrong.

    Unreachable in practice - it needs the answer key - but it is the ceiling
    that says how much room any real router had to work with.
    """
    n = len(records)
    budget = int(round(llm_rate * n))

    # Best use of the budget first: items the SLM failed and the LLM fixes.
    gain = [i for i, r in enumerate(records) if not r["slm_correct"] and r["llm_correct"]]
    neutral = [i for i, r in enumerate(records) if not r["slm_correct"] and not r["llm_correct"]]
    picked = set(gain[:budget])
    if len(picked) < budget:
        picked.update(neutral[: budget - len(picked)])

    outcomes = [
        r["llm_correct"] if i in picked else r["slm_correct"] for i, r in enumerate(records)
    ]
    cost = sum(
        r["llm_cost"] if i in picked else r["slm_cost"] for i, r in enumerate(records)
    )

    return Result(
        name=f"oracle@{llm_rate:.0%}",
        outcomes=outcomes,
        cost_usd=cost,
        llm_calls=len(picked),
        latencies=[],
        detail="upper bound, needs the answer key",
    )


def verifier_confusion(records: list[dict], strictness: int) -> dict[str, float]:
    """How good is the escalation gate at spotting the SLM's actual failures?

    This is the number that decides whether a cascade can work at all, and it
    is invisible in the leaderboard's accuracy column. A verifier with high
    recall catches real errors; one with low precision burns money escalating
    answers that were already correct.
    """
    uses_samples = strictness >= DISAGREE
    tp = fp = fn = tn = 0

    for record in records:
        item = {"numeric": record["numeric"], "choices": record["choices"]}
        samples = record["samples"] if uses_samples else []
        escalate, _ = should_escalate(item, record["slm_answer"], samples, strictness)
        slm_wrong = not record["slm_correct"]

        if escalate and slm_wrong:
            tp += 1
        elif escalate and not slm_wrong:
            fp += 1
        elif not escalate and slm_wrong:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
    }


if __name__ == "__main__":
    from src.matrix import load

    records = load()
    for result in (baseline(records), all_slm(records), cascade(records, 4)):
        print(
            f"{result.name:34s} acc={result.accuracy:6.1%} "
            f"cost=${result.cost_usd:.6f} llm_rate={result.llm_rate:5.1%}"
        )
