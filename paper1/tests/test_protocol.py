from __future__ import annotations

import pytest

from ddxkb.kb import KB
from ddxkb.protocol import (
    A4_HIERARCHIES,
    BED_REPORTED_GTPA1,
    BED_REPORTED_IL,
    BUDGETS,
    CURVE,
    LADDER,
    MAIN_HIERARCHY,
    MIN_GAIN,
    P1_WIDER,
    P2_LAMBDA,
    P2_SENSITIVITY_LAMBDAS,
    P2_VARIANTS,
    SENSITIVITY_HIERARCHY,
    SENSITIVITY_TAUS,
    TAU,
    P1Config,
    P2Config,
    P2Variant,
    check_fixed,
    p1_boundary,
    p1_grid,
    p1_wider_grid,
    p2_fixed,
    selection_key,
)
from ddxkb.simulator import AnswerMode, Hierarchy, StartInput
from helpers import toy_counts


def test_one_stop_threshold_and_one_smoothing_for_every_policy() -> None:
    assert TAU == 0.99
    assert SENSITIVITY_TAUS == (0.9, 0.95, 0.99, 0.999) and TAU in SENSITIVITY_TAUS
    assert (P2_LAMBDA, P2_SENSITIVITY_LAMBDAS) == (1.0, (0.5,))
    assert [v.name for v in P2_VARIANTS] == ["P2-binary", "P2-binary-agesex", "P2-value", "P2-value-agesex"]
    for variant in P2_VARIANTS:
        assert p2_fixed(variant) == P2Config(variant, 1.0, 0.99)
    assert BUDGETS == (5, 10, 15, 20, 30)
    assert len(CURVE) == 31 and CURVE[-1] == 30


def test_the_p1_candidates_are_alpha_and_eps_at_the_one_threshold() -> None:
    p1 = p1_grid()
    assert len(p1) == 9 and {c.tau for c in p1} == {TAU}
    assert p1[:4] == (
        P1Config(0.3, 1e-2, 0.99),
        P1Config(0.3, 1e-3, 0.99),
        P1Config(0.3, 1e-4, 0.99),
        P1Config(0.5, 1e-2, 0.99),
    )
    assert p1[-1] == P1Config(0.7, 1e-4, 0.99)


def test_the_fixed_settings_are_checked() -> None:
    p1 = P1Config(0.5, 1e-4, TAU)
    p2 = [p2_fixed(v) for v in P2_VARIANTS]
    check_fixed(p1, p2)
    with pytest.raises(ValueError, match=r"P1 stops at tau 0\.95"):
        check_fixed(P1Config(0.5, 1e-4, 0.95), p2)
    with pytest.raises(ValueError, match=r"P2-value stops at tau 0\.999"):
        check_fixed(p1, [*p2[:2], P2Config(P2_VARIANTS[2], 1.0, 0.999), p2[3]])
    with pytest.raises(ValueError, match=r"P2-binary smooths with lambda 0\.5"):
        check_fixed(p1, [P2Config(P2_VARIANTS[0], 0.5, TAU), *p2[1:]])
    for candidate in p1_grid():
        check_fixed(candidate, p2)
    # A value of the wider grid is recorded in development, never selected; nor is any other value.
    for outside in (P1Config(0.9, 1e-5, TAU), P1Config(0.5, 1e-5, TAU), P1Config(0.42, 1e-3, TAU)):
        with pytest.raises(ValueError, match=f"P1 with alpha {outside.alpha} and eps {outside.eps} is not one of"):
            check_fixed(outside, p2)


def test_the_ladder_changes_one_factor_at_a_time() -> None:
    assert list(LADDER) == [
        ("P2-binary", "P1"),
        ("P2-binary-agesex", "P2-binary"),
        ("P2-value", "P2-binary"),
        ("P2-value-agesex", "P2-value"),
        ("B-BED", "P2-value"),
    ]
    assert (BED_REPORTED_GTPA1, BED_REPORTED_IL) == (0.9947, 5.47)


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
    assert policy.hierarchy is MAIN_HIERARCHY is Hierarchy.NONE
    assert policy.start_inputs == {StartInput.INITIAL_EVIDENCE}
    p2 = P2Config(P2Variant(AnswerMode.VALUE, True), 1.0, 0.999)
    assert p2.as_dict() == {"mode": "value", "age_sex": True, "lam": 1.0, "tau": 0.999}
    assert P2Config.from_dict(p2.as_dict()) == p2 and p2.name == "P2-value-agesex"
    built = p2.policy(toy_kb, toy_counts(), (0,))
    assert built.table.mode is AnswerMode.VALUE and built.prior.log_p.ndim == 3
    assert (built.tau, built.min_gain, built.hierarchy) == (0.999, MIN_GAIN, MAIN_HIERARCHY)
    assert built.start_inputs == set(StartInput)
    plain = P2Config(P2Variant(AnswerMode.BINARY, False), 1.0, 0.9).policy(toy_kb, toy_counts(), (0,))
    assert plain.start_inputs == {StartInput.INITIAL_EVIDENCE}
    with pytest.raises(ValueError):
        P2Config.from_dict({"mode": "neither", "age_sex": False, "lam": 1.0, "tau": 0.9})


def test_the_wider_grid_adds_one_value_at_each_end_the_selection_reaches() -> None:
    assert p1_boundary(P1Config(0.5, 1e-3, TAU)) == {}
    assert p1_wider_grid(P1Config(0.5, 1e-3, TAU)) == p1_grid()
    first = P1Config(0.5, 1e-4, TAU)
    assert p1_boundary(first) == {"eps": "last"}
    wider = p1_wider_grid(first)
    assert len(wider) == 3 * 4 and set(p1_grid()) < set(wider) and {c.tau for c in wider} == {TAU}
    assert wider[:5] == (
        P1Config(0.3, 1e-2, TAU),
        P1Config(0.3, 1e-3, TAU),
        P1Config(0.3, 1e-4, TAU),
        P1Config(0.3, 1e-5, TAU),
        P1Config(0.5, 1e-2, TAU),
    )
    assert wider[-1] == P1Config(0.7, 1e-5, TAU)
    corner = P1Config(0.3, 1e-2, TAU)
    assert p1_boundary(corner) == {"alpha": "first", "eps": "first"}
    low = p1_wider_grid(corner)
    assert {c.alpha for c in low} == {0.1, 0.3, 0.5, 0.7} and {c.eps for c in low} == {1e-1, 1e-2, 1e-3, 1e-4}
    assert len(low) == 16 and low[0] == P1Config(0.1, 1e-1, TAU)
    assert {c.alpha for c in p1_wider_grid(P1Config(0.7, 1e-3, TAU))} == {0.3, 0.5, 0.7, 0.9}
    # The threshold is fixed, so no end of a tau list exists to widen.
    assert P1_WIDER == {"alpha": (0.1, 0.9), "eps": (1e-1, 1e-5)}
    with pytest.raises(ValueError, match="not a P1 candidate"):
        p1_boundary(P1Config(0.4, 1e-3, TAU))


def test_hierarchy_rules_of_the_revision() -> None:
    assert (MAIN_HIERARCHY, SENSITIVITY_HIERARCHY) == (Hierarchy.NONE, Hierarchy.IMPLICIT)
    assert A4_HIERARCHIES == (Hierarchy.IMPLICIT, Hierarchy.PARENT_FIRST, Hierarchy.NO_CHILDREN)
