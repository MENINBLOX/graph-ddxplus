from __future__ import annotations

import pytest

from ddxkb.kb import KB
from ddxkb.protocol import (
    BUDGETS,
    CURVE,
    MIN_GAIN,
    P2_VARIANTS,
    P1Config,
    P2Config,
    P2Variant,
    p1_grid,
    p2_grid,
    selection_key,
)
from ddxkb.simulator import AnswerMode
from helpers import toy_counts


def test_grids_are_the_design_candidates_in_listed_order() -> None:
    p1 = p1_grid()
    assert len(p1) == 27
    assert p1[:4] == (
        P1Config(0.3, 1e-2, 0.9),
        P1Config(0.3, 1e-2, 0.95),
        P1Config(0.3, 1e-2, 0.99),
        P1Config(0.3, 1e-3, 0.9),
    )
    assert p1[-1] == P1Config(0.7, 1e-4, 0.99)
    assert [v.name for v in P2_VARIANTS] == ["P2-binary", "P2-binary-agesex", "P2-value", "P2-value-agesex"]
    grid = p2_grid(P2_VARIANTS[2])
    assert len(grid) == 8
    assert [(c.lam, c.tau) for c in grid[:5]] == [(0.5, 0.9), (0.5, 0.95), (0.5, 0.99), (0.5, 0.999), (1.0, 0.9)]
    assert all(c.variant == P2Variant(AnswerMode.VALUE, False) for c in grid)
    assert BUDGETS == (5, 10, 15, 20, 29)
    assert len(CURVE) == 30 and CURVE[-1] == 29


def test_selection_key_rounds_to_two_decimals_then_prefers_fewer_questions() -> None:
    a = selection_key(0.98391, 11.9, 5)
    b = selection_key(0.98389, 12.4, 2)
    c = selection_key(0.98401, 15.0, 0)
    assert a < b  # 98.39 = 98.39, fewer questions wins
    assert c < a  # 98.40 beats 98.39 whatever the IL
    assert selection_key(0.5, 10.0, 1) < selection_key(0.5, 10.0, 2)


def test_configs_round_trip_and_build_policies(toy_kb: KB) -> None:
    p1 = P1Config(0.5, 1e-3, 0.99)
    assert P1Config.from_dict(p1.as_dict()) == p1 and p1.name == "P1"
    policy = p1.policy(toy_kb, (0, 3))
    assert (policy.tau, policy.budgets, policy.table.probs[0, 0]) == (0.99, (0, 3), 0.5)
    assert policy.min_gain == MIN_GAIN == 1e-6
    p2 = P2Config(P2Variant(AnswerMode.VALUE, True), 1.0, 0.999)
    assert p2.as_dict() == {"mode": "value", "age_sex": True, "lam": 1.0, "tau": 0.999}
    assert P2Config.from_dict(p2.as_dict()) == p2 and p2.name == "P2-value-agesex"
    built = p2.policy(toy_kb, toy_counts(), (0,))
    assert built.table.mode is AnswerMode.VALUE and built.prior.log_p.ndim == 3
    assert (built.tau, built.min_gain) == (0.999, MIN_GAIN)
    with pytest.raises(ValueError):
        P2Config.from_dict({"mode": "neither", "age_sex": False, "lam": 1.0, "tau": 0.9})
