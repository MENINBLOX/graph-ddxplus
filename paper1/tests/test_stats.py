from __future__ import annotations

import numpy as np
import pytest

from ddxkb.stats import (
    Interval,
    McNemar,
    bootstrap_interval,
    bootstrap_intervals,
    mcnemar_exact,
    wilson_interval,
)


def test_wilson_matches_hand_computation() -> None:
    ci = wilson_interval(0.5, 100)
    assert ci.estimate == 0.5
    assert (ci.low, ci.high) == pytest.approx((0.403832, 0.596168), abs=1e-6)
    zero = wilson_interval(0.0, 10)
    assert zero.low == 0.0
    assert zero.high == pytest.approx(0.277533, abs=1e-6)
    one = wilson_interval(1.0, 10)
    assert (one.low, one.high) == pytest.approx((1 - 0.277533, 1.0), abs=1e-6)
    with pytest.raises(ValueError):
        wilson_interval(0.5, 0)


def test_mcnemar_exact_hand_cases() -> None:
    a = np.array([1, 1, 0, 0, 0, 0, 0, 1])
    b = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    # 0 discordant for a, 5 for b: p = 2 * (1/2)^5.
    assert mcnemar_exact(a, b) == McNemar(only_a=0, only_b=5, p_value=0.0625)
    a = np.array([1] + [0] * 9)
    b = np.array([0] + [1] * 9)
    # 1 vs 9 discordant: p = 2 * (C(10,0) + C(10,1)) / 2^10 = 22/1024.
    assert mcnemar_exact(a, b).p_value == 22 / 1024
    assert mcnemar_exact([True, False], [False, True]).p_value == 1.0
    assert mcnemar_exact([1, 1], [1, 1]) == McNemar(0, 0, 1.0)
    with pytest.raises(ValueError, match="0/1"):
        mcnemar_exact([0.5, 1.0], [1, 1])


def test_mcnemar_large_counts_stay_finite() -> None:
    a = np.r_[np.ones(3000), np.zeros(3300)]
    b = np.r_[np.zeros(3000), np.ones(3300)]
    p = mcnemar_exact(a, b).p_value
    assert 0.0 < p < 0.001


def test_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    x = np.arange(200, dtype=float) % 7
    first = bootstrap_interval(x)
    assert first == bootstrap_interval(x)
    assert first.estimate == pytest.approx(x.mean())
    assert first.low < first.estimate < first.high
    assert bootstrap_interval(x, seed=1) != first
    assert bootstrap_interval(np.full(50, 3.0)) == Interval(3.0, 3.0, 3.0)


def test_bootstrap_resamples_patients_from_one_generator() -> None:
    x = np.array([0.0, 1.0, 5.0, 2.0])
    rng = np.random.default_rng(0)
    means = [x[rng.integers(0, 4, size=4)].mean() for _ in range(10)]
    ci = bootstrap_interval(x, b=10, seed=0)
    assert (ci.low, ci.high) == pytest.approx(tuple(np.quantile(means, [0.025, 0.975])))


def test_joint_bootstrap_equals_each_column_alone() -> None:
    rng = np.random.default_rng(3)
    columns = [rng.random(37), rng.integers(1, 31, 37).astype(float), rng.random(37) > 0.5]
    joint = bootstrap_intervals(columns, b=300)
    assert joint == [bootstrap_interval(np.asarray(c, dtype=float), b=300) for c in columns]
    assert bootstrap_intervals(columns[:1], b=300) == joint[:1]
    with pytest.raises(ValueError):
        bootstrap_intervals([])
    with pytest.raises(ValueError):
        bootstrap_intervals([np.ones(3), np.ones(4)])
