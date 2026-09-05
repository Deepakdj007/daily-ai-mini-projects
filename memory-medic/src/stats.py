"""Paired statistics for small samples, with a fixed seed.

Inputs:  paired 0/1 outcome vectors from two arms on the same probes
Outputs: confidence intervals and an exact paired p-value

Every arm answers the identical probes, so the comparisons are paired and the
tests here are the paired ones. An unpaired test on forty probes would throw
away exactly the structure that makes forty probes enough to say anything.

McNemar is computed exactly rather than with the chi-squared approximation,
because at this size the approximation is not trustworthy and `math.comb` costs
nothing. No scipy.
"""

from __future__ import annotations

from math import comb, sqrt
from typing import Sequence

import numpy as np

RNG_SEED = 20260905
N_BOOTSTRAP = 10_000


def _rng() -> np.random.Generator:
    """Fixed seed: the same data always produces the same interval."""
    return np.random.default_rng(RNG_SEED)


def wilson_interval(passed: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval - honest at small n and at rates near 0 or 1."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054 if confidence == 0.95 else 2.5758293035489004
    rate = passed / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = z * sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_ci(outcomes: Sequence[int], confidence: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap on one arm's pass rate."""
    values = np.asarray(outcomes, dtype=float)
    if values.size == 0:
        return (0.0, 0.0)
    draws = _rng().choice(values, size=(N_BOOTSTRAP, values.size), replace=True)
    means = draws.mean(axis=1)
    tail = (1 - confidence) / 2
    return (float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail)))


def paired_bootstrap_diff(a: Sequence[int], b: Sequence[int],
                          confidence: float = 0.95) -> tuple[float, float]:
    """CI on the difference, resampling PROBE INDICES so the pairing survives."""
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    if first.size == 0 or first.size != second.size:
        return (0.0, 0.0)
    indices = _rng().integers(0, first.size, size=(N_BOOTSTRAP, first.size))
    diffs = first[indices].mean(axis=1) - second[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    return (float(np.quantile(diffs, tail)), float(np.quantile(diffs, 1 - tail)))


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> tuple[int, int, float]:
    """Exact two-sided McNemar. Returns (a-only wins, b-only wins, p).

    Only the probes the two arms disagree on carry information; the ones they
    both pass or both fail say nothing about which is better.
    """
    b_count = sum(1 for x, y in zip(a, b) if x and not y)
    c_count = sum(1 for x, y in zip(a, b) if y and not x)
    n = b_count + c_count
    if n == 0:
        return (0, 0, 1.0)
    k = min(b_count, c_count)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return (b_count, c_count, min(1.0, 2 * tail))


if __name__ == "__main__":
    # Ten probes the treatment wins and none the other way: the strongest
    # evidence ten paired probes can carry.
    after = [1] * 10
    before = [0] * 10
    b, c, p = mcnemar_exact(after, before)
    assert (b, c) == (10, 0), (b, c)
    assert p < 0.01, p
    print(f"10-0 discordant -> p={p:.5f}")

    # Eight is the smallest split that still clears p < 0.01.
    _, _, p8 = mcnemar_exact([1] * 8, [0] * 8)
    _, _, p7 = mcnemar_exact([1] * 7, [0] * 7)
    print(f"8-0 -> p={p8:.5f}   7-0 -> p={p7:.5f}")
    assert p8 < 0.01 <= p7, "eight one-way disagreements is the threshold at this size"

    assert mcnemar_exact([1, 0], [1, 0])[2] == 1.0, "no disagreement means no evidence"

    low, high = paired_bootstrap_diff(after, before)
    assert low > 0.5, (low, high)
    print(f"paired bootstrap on the same data -> {low:+.0%} to {high:+.0%}")

    assert paired_bootstrap_diff(after, before) == paired_bootstrap_diff(after, before), \
        "a fixed seed must give a reproducible interval"
    lo, hi = wilson_interval(10, 10)
    print(f"wilson 10/10 -> {lo:.2f} to {hi:.2f}")
    print("OK - exact McNemar, paired bootstrap, reproducible under the fixed seed")
