"""Interval estimates and paired tests of design §4."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    low: float
    high: float


def wilson_interval(p_hat: float, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a proportion observed as p_hat over n trials.

    A tie-aware GTPA@1 is a mean of fractional scores; it enters as p_hat like any proportion.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= p_hat <= 1.0:
        raise ValueError("p_hat must lie in [0, 1]")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    # The bounds are exactly 0 at p_hat = 0 and 1 at p_hat = 1; centre - half leaves rounding residue there.
    low = 0.0 if p_hat == 0.0 else max(0.0, centre - half)
    high = 1.0 if p_hat == 1.0 else min(1.0, centre + half)
    return Interval(p_hat, low, high)


def bootstrap_interval(
    values: npt.ArrayLike, b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED, confidence: float = 0.95
) -> Interval:
    """Percentile interval of the mean, resampling patients with replacement."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    return bootstrap_intervals([x], b=b, seed=seed, confidence=confidence)[0]


def bootstrap_intervals(
    columns: Sequence[npt.ArrayLike], b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED, confidence: float = 0.95
) -> list[Interval]:
    """`bootstrap_interval` of each column, all from the same resampled patients.

    Each column is one per-patient quantity over the same patients in the same order; every replicate draws
    one set of patients for all of them, so each interval equals `bootstrap_interval` of its column alone.
    """
    x = np.asarray(columns, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("columns must be one or more non-empty columns of equal length")
    means = _resampled_means(x, b, seed)
    lo, hi = np.quantile(means, [0.5 - confidence / 2, 0.5 + confidence / 2], axis=0, method="linear")
    return [Interval(float(m), float(low), float(high)) for m, low, high in zip(x.mean(axis=1), lo, hi, strict=True)]


def _resampled_means(x: npt.NDArray[np.float64], b: int, seed: int) -> npt.NDArray[np.float64]:
    """(b, columns) replicate means of the rows of x, shape (columns, patients)."""
    # One draw of n indices per replicate keeps memory at O(columns x n) for n ~ 134k patients.
    rng = np.random.default_rng(seed)
    n = x.shape[1]
    means = np.empty((b, x.shape[0]), dtype=np.float64)
    for r in range(b):
        idx = rng.integers(0, n, size=n)
        # Row by row: a 2-D mean may sum in another order, and a column's interval would then depend on
        # which other columns share the call.
        for c, row in enumerate(x):
            means[r, c] = row[idx].mean()
    return means


@dataclass(frozen=True, slots=True)
class McNemar:
    only_a: int
    only_b: int
    p_value: float


def mcnemar_exact(a_correct: npt.ArrayLike, b_correct: npt.ArrayLike) -> McNemar:
    """Two-sided exact McNemar test on per-patient 0/1 outcomes of two policies."""
    x = np.asarray(a_correct)
    y = np.asarray(b_correct)
    if x.shape != y.shape:
        raise ValueError("paired outcomes must have the same shape")
    if not (np.isin(x, (0, 1)).all() and np.isin(y, (0, 1)).all()):
        raise ValueError("McNemar needs 0/1 outcomes; use fixed-order correctness, not tie-aware scores")
    xb = x.astype(bool)
    yb = y.astype(bool)
    only_a = int(np.count_nonzero(xb & ~yb))
    only_b = int(np.count_nonzero(~xb & yb))
    return McNemar(only_a, only_b, _binomial_two_sided(only_a, only_a + only_b))


def _binomial_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    m = min(k, n - k)
    term = 1
    tail = 1
    for i in range(m):
        term = term * (n - i) // (i + 1)
        tail += term
    # Exact integers: P(X <= m) = tail / 2^n, doubled for the two-sided test.
    return min(1.0, tail / (1 << (n - 1)))
