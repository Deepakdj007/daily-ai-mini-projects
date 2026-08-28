"""Confidence intervals and paired significance tests.

Inputs:  per-item boolean outcome vectors.
Outputs: bootstrap CIs and exact McNemar p-values.

Why this file exists
--------------------
At n=23 a single probe is more than four points of recall. Two arms differing
by "ten points" differ by two probes, which may be nothing at all. Reporting a
bare recall number at this sample size invites the reader to believe a
precision the experiment does not have, which is why the criterion is stated
in twenty-point steps and no slice below n=12 is printed at all.

Every arm answers the same probes, so comparisons are paired - far more
powerful than treating them as independent samples, and the only reason any
conclusion is available at this n. Power then depends on the number of
DISAGREEMENTS, not on n: a 12:1 split is significant, a 9:4 split is not.
"""

from __future__ import annotations

from math import comb

import numpy as np

RNG_SEED = 20260815
N_BOOTSTRAP = 10_000


def _rng() -> np.random.Generator:
    """Fixed seed: the same data must always produce the same interval."""
    return np.random.default_rng(RNG_SEED)


def bootstrap_ci(outcomes: list[bool], confidence: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for a single strategy's accuracy."""
    values = np.asarray(outcomes, dtype=float)
    if values.size == 0:
        return (0.0, 0.0)
    rng = _rng()
    draws = rng.choice(values, size=(N_BOOTSTRAP, values.size), replace=True)
    means = draws.mean(axis=1)
    lo = (1 - confidence) / 2 * 100
    return float(np.percentile(means, lo)), float(np.percentile(means, 100 - lo))


def paired_bootstrap_diff(
    a: list[bool], b: list[bool], confidence: float = 0.95
) -> tuple[float, float, float]:
    """CI for (mean(a) - mean(b)), resampling items rather than outcomes.

    Resampling item indices - not each vector independently - is what keeps the
    comparison paired. Break that and the interval widens for no reason.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    n = arr_a.size
    if n == 0:
        return (0.0, 0.0, 0.0)

    rng = _rng()
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    diffs = arr_a[idx].mean(axis=1) - arr_b[idx].mean(axis=1)
    lo = (1 - confidence) / 2 * 100
    return (
        float(arr_a.mean() - arr_b.mean()),
        float(np.percentile(diffs, lo)),
        float(np.percentile(diffs, 100 - lo)),
    )


def paired_bootstrap_ratio(
    a: list[bool], b: list[bool], confidence: float = 0.95
) -> tuple[float, float, float]:
    """CI for mean(a) / mean(b) - the quality-retention number."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    n = arr_a.size
    if n == 0:
        return (0.0, 0.0, 0.0)

    rng = _rng()
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    denom = arr_b[idx].mean(axis=1)
    denom[denom == 0] = np.nan
    ratios = arr_a[idx].mean(axis=1) / denom
    ratios = ratios[~np.isnan(ratios)]
    if ratios.size == 0:
        return (0.0, 0.0, 0.0)

    point = arr_a.mean() / arr_b.mean() if arr_b.mean() else 0.0
    lo = (1 - confidence) / 2 * 100
    return (
        float(point),
        float(np.percentile(ratios, lo)),
        float(np.percentile(ratios, 100 - lo)),
    )


def mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired binary outcomes.

    Only the disagreements carry information: items both strategies get right,
    or both get wrong, say nothing about which is better. Returns
    (a_only_correct, b_only_correct, p_value).
    """
    b_count = sum(1 for x, y in zip(a, b) if x and not y)
    c_count = sum(1 for x, y in zip(a, b) if y and not x)
    n = b_count + c_count
    if n == 0:
        return (0, 0, 1.0)

    k = min(b_count, c_count)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return (b_count, c_count, min(1.0, 2 * tail))


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    strong = list(rng.random(120) < 0.90)
    weak = [s and bool(rng.random() < 0.93) for s in strong]

    lo, hi = bootstrap_ci(strong)
    print(f"strong accuracy : {np.mean(strong):.1%}  95% CI [{lo:.1%}, {hi:.1%}]")
    lo, hi = bootstrap_ci(weak)
    print(f"weak   accuracy : {np.mean(weak):.1%}  95% CI [{lo:.1%}, {hi:.1%}]")

    point, lo, hi = paired_bootstrap_diff(weak, strong)
    print(f"\npaired diff     : {point:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]")
    point, lo, hi = paired_bootstrap_ratio(weak, strong)
    print(f"retention ratio : {point:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    b, c, p = mcnemar_exact(weak, strong)
    print(f"\nmcnemar         : b={b} c={c} p={p:.4f}")

    identical = mcnemar_exact(strong, strong)
    print(f"identical vectors: b={identical[0]} c={identical[1]} p={identical[2]:.4f}")
