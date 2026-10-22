from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ddxkb.kb import KB
from ddxkb.policies.ablations import CoverageOrder, Heard, heard_answers
from ddxkb.policies.bayes import BayesPolicy, membership_table, normalize, uniform_prior
from ddxkb.simulator import Answer
from helpers import binary_kb, toy_patient

LISTS = {"A": ["e0", "e1", "e2"], "B": ["e0", "e3"], "C": ["e4", "e5"]}
ORDER = ["e0", "e1", "e2", "e3", "e4", "e5"]


@pytest.fixture
def trio() -> KB:
    return binary_kb(LISTS, ORDER)


def test_heard_answers_drop_one_polarity(trio: KB) -> None:
    table = membership_table(trio, 0.5, 0.01)
    yes, no = Answer("e1", positive=True), Answer("e1", positive=False)
    positive = heard_answers(table, Heard.POSITIVE)
    negative = heard_answers(table, Heard.NEGATIVE)
    assert np.array_equal(positive.log_likelihood(yes), table.log_likelihood(yes))
    assert np.array_equal(positive.log_likelihood(no), np.zeros(3))
    assert np.array_equal(negative.log_likelihood(no), table.log_likelihood(no))
    assert np.array_equal(negative.log_likelihood(yes), np.zeros(3))
    assert np.array_equal(positive.probs, table.probs)


def test_positive_only_episode_by_hand(trio: KB) -> None:
    p1 = BayesPolicy(membership_table(trio, 0.5, 0.01), uniform_prior(trio), None, (0, 2, 3))
    ablated = dataclasses.replace(p1, table=heard_answers(p1.table, Heard.POSITIVE))
    ep = ablated.run(toy_patient(["e0", "e3"], initial="e0", pathology="B"))
    # After e0 (yes) A and B tie, and e1, e2, e3 each split them equally: KB order asks e1, then e2. Their
    # "no" answers are ignored, so the tie stands until e3 (yes) moves B ahead.
    assert ep.questions[:3] == ("e1", "e2", "e3")
    after_initial = np.log([0.5, 0.5, 0.01])
    assert ep.at_budget(0) == pytest.approx(normalize(after_initial))
    assert ep.at_budget(2) == pytest.approx(normalize(after_initial))
    assert ep.at_budget(3) == pytest.approx(normalize(after_initial + np.log([0.01, 0.5, 0.01])))
    full = p1.run(toy_patient(["e0", "e3"], initial="e0", pathology="B"))
    assert full.questions[:2] == ("e1", "e2")
    assert full.at_budget(2) == pytest.approx(normalize(after_initial + np.log([0.25, 0.99**2, 0.99**2])))


def test_negative_only_ignores_the_initial_evidence(trio: KB) -> None:
    p1 = BayesPolicy(membership_table(trio, 0.5, 0.01), uniform_prior(trio), None, (0,))
    ablated = dataclasses.replace(p1, table=heard_answers(p1.table, Heard.NEGATIVE))
    ep = ablated.run(toy_patient(["e0"], initial="e0", pathology="A"))
    assert ep.at_budget(0) == pytest.approx([1 / 3] * 3)


def test_coverage_order_asks_what_most_candidate_lists_hold(trio: KB) -> None:
    p1 = BayesPolicy(membership_table(trio, 0.5, 0.01), uniform_prior(trio), None, (0, 1, 2, 3, 4, 5))
    policy = dataclasses.replace(p1, selector=CoverageOrder())
    ep = policy.run(toy_patient(["e0", "e3"], initial="e0", pathology="B"))
    # Candidates after e0 (yes): A and B; their unasked codes e1, e2, e3 lie on one list each, so KB order
    # asks e1 (no), e2 (no), e3 (yes); then only B is left, its list is exhausted, and coverage 0 falls back
    # to KB order over the unasked e4, e5.
    assert ep.questions == ("e1", "e2", "e3", "e4", "e5")


def test_coverage_order_prefers_shared_codes() -> None:
    kb = binary_kb({"A": ["e0", "e1", "e2"], "B": ["e0", "e2"], "C": ["e0", "e3"]}, ["e0", "e1", "e2", "e3"])
    policy = BayesPolicy(membership_table(kb, 0.5, 0.01), uniform_prior(kb), None, (0, 1), selector=CoverageOrder())
    ep = policy.run(toy_patient(["e0", "e3"], initial="e0", pathology="C"))
    assert ep.questions[0] == "e2"
