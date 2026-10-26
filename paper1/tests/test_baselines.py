from __future__ import annotations

import os
import random

import numpy as np
import pytest

from ddxkb.data import Patient
from ddxkb.evaluation import Scored, ScoreTask
from ddxkb.experiment import Referenced
from ddxkb.kb import KB
from ddxkb.metrics import gtpa_at_k
from ddxkb.policies.baselines import (
    FixedOrder,
    FullProfile,
    MostFrequentDisease,
    ProfileMajority,
    RandomInKBCandidates,
    SetInclusion,
    initial_only,
    random_questions,
    train_frequency_questions,
)
from ddxkb.policies.bayes import BayesPolicy, frequency_prior, frequency_table, membership_table, uniform_prior
from ddxkb.policies.episode import Policy
from ddxkb.runner import chunks, map_items, worker_pool
from ddxkb.simulator import Answer, AnswerMode, Oracle, Start, StartInput
from helpers import binary_kb, toy_counts, toy_patient

# A and B share e0; C lists neither e0 nor anything of A or B.
TRIO = {"A": ["e0", "e1", "e2"], "B": ["e0", "e3"], "C": ["e4", "e5"]}
TRIO_ORDER = ["e0", "e1", "e2", "e3", "e4", "e5"]


@pytest.fixture
def trio() -> KB:
    return binary_kb(TRIO, TRIO_ORDER)


def _p1(
    kb: KB, tau: float | None = 0.99, budgets: tuple[int, ...] = (), max_il: int = 30, min_gain: float | None = None
) -> BayesPolicy:
    return BayesPolicy(membership_table(kb, 0.5, 1e-3), uniform_prior(kb), tau, budgets, max_il, min_gain=min_gain)


def _beta(row: int = 0) -> Patient:
    return toy_patient(
        ["douleur", "douleur_intens_@_2", "douleur_lieu_@_dos", "atcd", "trav1_@_N"], initial="douleur", row=row
    )


def test_most_frequent_disease_ranks_by_train_frequency(toy_kb: KB) -> None:
    pol = MostFrequentDisease.from_counts(toy_kb, toy_counts())
    alpha = pol.run(toy_patient(["fievre"], initial="fievre", pathology="Alpha", row=7))
    assert alpha.posterior == pytest.approx([0.5, 0.4, 0.1])
    assert (alpha.row, alpha.gold, alpha.il, alpha.questions, alpha.budgets) == (7, 0, 1, (), ())
    assert gtpa_at_k(alpha.posterior, alpha.gold, 1) == 1.0
    beta = pol.run(_beta())
    assert gtpa_at_k(beta.posterior, beta.gold, 1) == 0.0
    assert gtpa_at_k(beta.posterior, beta.gold, 3) == 1.0
    with pytest.raises(ValueError):
        MostFrequentDisease(toy_kb, np.array([0.5, 0.4]))


def test_initial_only_is_the_posterior_before_any_question(toy_kb: KB) -> None:
    p1 = _p1(toy_kb, budgets=(0, 2), min_gain=1e-6)
    b0 = initial_only(p1)
    assert (b0.tau, b0.min_gain, b0.budgets, b0.max_il) == (None, None, (), 1)
    ep = b0.run(_beta())
    assert (ep.il, ep.questions, ep.budgets) == (1, (), ())
    # Beta and Gamma both list the initial evidence douleur: a two-way tie at rank 1.
    assert ep.posterior[1] == ep.posterior[2] > ep.posterior[0]
    assert gtpa_at_k(ep.posterior, ep.gold, 1) == 0.5
    assert np.array_equal(ep.posterior, p1.run(_beta()).at_budget(0))
    assert (p1.tau, p1.max_il) == (0.99, 30)


def test_set_inclusion_is_uniform_over_the_consistent_diseases(toy_kb: KB) -> None:
    pol = SetInclusion(toy_kb)
    # atcd is on Beta's list only.
    assert pol.run(_beta()).posterior.tolist() == [0.0, 1.0, 0.0]
    # trav1_@_N is the default value, so absent; fievre and trav1 are on Alpha's and Gamma's lists.
    assert pol.run(toy_patient(["fievre", "trav1_@_N"], initial="fievre")).posterior.tolist() == [0.5, 0.0, 0.5]
    assert pol.run(toy_patient(["fievre", "trav1_@_Asie"], initial="fievre")).posterior.tolist() == [0.5, 0.0, 0.5]
    # No list holds both toux and douleur.
    none = pol.run(toy_patient(["toux", "douleur"], initial="toux"))
    assert none.posterior.tolist() == [0.0, 0.0, 0.0]
    assert gtpa_at_k(none.posterior, none.gold, 1) == 0.0


def test_full_profile_kb_bayes_by_hand() -> None:
    kb = binary_kb({"A": ["e0", "e1", "e2", "e4"], "B": ["e0", "e1", "e3", "e4"], "C": ["e0", "e4"]}, TRIO_ORDER[:5])
    pol = FullProfile(membership_table(kb, 0.8, 0.2), uniform_prior(kb))
    ep = pol.run(toy_patient(["e0", "e1", "e2", "e4"], initial="e0", pathology="A"))
    # Each answer scores 0.8 where it agrees with the disease's list and 0.2 where it does not:
    # A agrees on all five, B misses e2 and e3, C misses e1 and e2.
    assert ep.posterior == pytest.approx(np.array([0.8**5, 0.8**3 * 0.04, 0.8**3 * 0.04]) / 0.8**3 / 0.72)
    assert (ep.il, ep.questions) == (1, ())


def test_full_profile_naive_bayes_binary_is_the_product_over_every_evidence(toy_kb: KB) -> None:
    counts = toy_counts()
    table = frequency_table(toy_kb, counts, 1.0, AnswerMode.BINARY)
    prior = frequency_prior(counts, 1.0)
    ep = FullProfile(table, prior).run(_beta())
    present = [False, False, True, True, True, False, True]
    p_yes = table.probs[:, 0::2]
    joint = np.exp(prior.log_p) * np.prod(np.where(present, p_yes, 1 - p_yes), axis=1)
    assert ep.posterior == pytest.approx(joint / joint.sum(), rel=1e-12)


def test_full_profile_naive_bayes_reads_values_in_value_mode(toy_kb: KB) -> None:
    counts = toy_counts()
    table = frequency_table(toy_kb, counts, 1.0, AnswerMode.VALUE)
    prior = frequency_prior(counts, 1.0, by_age_sex=True)
    patient = _beta()
    oracle = Oracle(toy_kb, patient)
    start = Start(oracle.answer(patient.initial), patient.age, patient.sex)
    log_post = prior(start) + sum(table.log_likelihood(oracle.answer(e.code)) for e in toy_kb.evidences)
    expected = np.exp(log_post - log_post.max())
    ep = FullProfile(table, prior).run(patient)
    assert ep.posterior == pytest.approx(expected / expected.sum(), rel=1e-12)
    other = FullProfile(table, prior).run(
        toy_patient(["douleur", "douleur_intens_@_3", "douleur_lieu_@_tete", "atcd", "trav1_@_N"], "douleur")
    )
    assert not np.allclose(other.posterior, ep.posterior)


def test_profile_majority_shares_the_diseases_of_identical_positive_sets(toy_kb: KB) -> None:
    # The default values of douleur_intens and trav1 are negative, so the first three records share one profile.
    patients = [
        toy_patient(["douleur", "atcd"], "douleur", "Beta", row=0),
        toy_patient(["douleur", "atcd", "douleur_intens_@_0", "trav1_@_N"], "douleur", "Beta", row=1),
        toy_patient(["douleur", "atcd"], "douleur", "Gamma", row=2),
        toy_patient(["fievre"], "fievre", "Alpha", row=3),
    ]
    majority = ProfileMajority.from_patients(toy_kb, patients)
    shared = majority.run(patients[2])
    assert shared.posterior.tolist() == pytest.approx([0.0, 2 / 3, 1 / 3])
    assert (shared.il, shared.questions) == (1, ())
    assert gtpa_at_k(shared.posterior, shared.gold, 1) == 0.0
    assert gtpa_at_k(majority.run(patients[0]).posterior, 1, 1) == 1.0
    assert majority.run(patients[3]).posterior.tolist() == [1.0, 0.0, 0.0]
    with pytest.raises(KeyError):
        majority.run(toy_patient(["toux"], "toux", "Alpha"))


def test_random_questions_come_from_the_candidate_lists(trio: KB) -> None:
    patient = toy_patient(["e0", "e1", "e2"], initial="e0", pathology="A", row=11)
    firsts = set()
    for seed in range(30):
        ep = random_questions(_p1(trio, tau=None, max_il=6), seed).run(patient)
        assert sorted(ep.questions) == TRIO_ORDER[1:]
        firsts.add(ep.questions[0])
        # A lists every positive answer, so its unasked evidences stay in the pool until asked; C's
        # evidences enter only after that.
        pos = {q: i for i, q in enumerate(ep.questions)}
        assert max(pos["e1"], pos["e2"]) < min(pos["e4"], pos["e5"])
    assert firsts == {"e1", "e2", "e3"}


def test_random_choice_falls_back_to_the_least_violating_diseases(trio: KB) -> None:
    table = membership_table(trio, 0.5, 1e-3)
    asked = np.array([True, True, False, False, True, False])
    positive = np.array([True, True, False, False, True, False])
    # Positives e0, e1, e4: A misses one (e4), B and C two each, so A is the only candidate and e2 the pool.
    for seed in range(10):
        choose = RandomInKBCandidates(seed).start(table, toy_patient(["e0"], initial="e0", row=seed))
        assert choose(np.full(3, 1 / 3), asked, positive, np.zeros(6)) == 2
    # Once A's list is exhausted, any unasked evidence.
    asked[2] = True
    drawn = {
        RandomInKBCandidates(seed).start(table, toy_patient(["e0"], "e0"))(
            np.full(3, 1 / 3), asked, positive, np.zeros(6)
        )
        for seed in range(20)
    }
    assert drawn == {3, 5}


def test_random_questions_are_seeded_by_seed_and_row(toy_kb: KB) -> None:
    p1 = _p1(toy_kb, tau=None, max_il=7, min_gain=1e-6)
    b2 = random_questions(p1, 3)
    assert (b2.table, b2.prior, b2.tau, b2.min_gain, b2.budgets, b2.max_il) == (
        p1.table,
        p1.prior,
        p1.tau,
        p1.min_gain,
        p1.budgets,
        p1.max_il,
    )
    patients = [_beta(row) for row in range(12)]
    first = ScoreTask(b2, toy_kb, ())(patients).questions
    again = ScoreTask(b2, toy_kb, ())(patients[::-1]).questions[::-1]
    assert first == again
    assert len(set(first)) > 1
    assert ScoreTask(random_questions(p1, 4), toy_kb, ())(patients).questions != first


def test_fixed_order_by_train_frequency(toy_kb: KB) -> None:
    counts = toy_counts()
    # Records with each evidence: fievre 11, toux 7, douleur 10, intens 10, lieu 8, trav1 1, atcd 3.
    assert FixedOrder.by_train_frequency(counts).order == (0, 2, 3, 4, 1, 6, 5)
    b6 = train_frequency_questions(_p1(toy_kb, tau=None, max_il=7), counts)
    ep = b6.run(_beta())
    assert ep.questions == ("fievre", "douleur_intens", "douleur_lieu", "toux", "atcd", "trav1")
    with pytest.raises(ValueError, match="permutation"):
        FixedOrder((0, 0, 1))
    with pytest.raises(ValueError, match="evidences"):
        FixedOrder((1, 0)).start(b6.table, _beta())


def test_frequency_order_ranks_and_stops_like_the_given_policy(toy_kb: KB) -> None:
    counts = toy_counts()
    p2 = BayesPolicy(frequency_table(toy_kb, counts, 1.0, AnswerMode.BINARY), frequency_prior(counts, 1.0), 0.9)
    b6 = train_frequency_questions(p2, counts)
    assert (b6.table, b6.prior, b6.tau) == (p2.table, p2.prior, p2.tau)
    ep = b6.run(_beta())
    assert ep.posterior.max() >= 0.9 or ep.il == toy_kb.n_evidences
    log_post = p2.prior(Start(Answer("douleur", True))) + p2.table.log_likelihood(Answer("douleur", True))
    oracle = Oracle(toy_kb, _beta())
    for code in ep.asked:
        log_post = log_post + p2.table.log_likelihood(oracle.answer(code))
    assert ep.posterior == pytest.approx(np.exp(log_post) / np.exp(log_post).sum())


def _sample(validation: list[Patient]) -> list[Patient]:
    return [validation[i] for i in random.Random(0).sample(range(len(validation)), 20_000)]


def _mean_gtpa1(policy: Policy, kb: KB, patients: list[Patient]) -> float:
    processes = min(16, os.cpu_count() or 1)
    with worker_pool(processes) as pool:
        s = Scored.concat(map_items(ScoreTask(policy, kb, ()), chunks(patients, 4 * processes), pool))
    assert not s.failed.any()
    return 100 * float(s.column("gtpa1").mean())


def test_no_question_and_full_profile_references_on_the_review_sample(real_kb: KB, validation: list[Patient]) -> None:
    sample = _sample(validation)
    p1 = _p1(real_kb)
    # Budget 0 of P1 on this sample (test_bayes), and the review's full-profile KB-Bayes figure (§6.2), which
    # it printed as 99.08%.
    assert round(_mean_gtpa1(initial_only(p1), real_kb, sample), 2) == 25.47
    assert _mean_gtpa1(FullProfile(p1.table, p1.prior), real_kb, sample) == pytest.approx(99.085, abs=1e-9)


def test_set_inclusion_on_validation(real_kb: KB, validation: list[Patient]) -> None:
    assert round(_mean_gtpa1(SetInclusion(real_kb), real_kb, validation), 2) == 95.90


def test_reference_points_declare_what_they_read(toy_kb: KB) -> None:
    counts = toy_counts()
    initial = frozenset({StartInput.INITIAL_EVIDENCE})
    value = frequency_table(toy_kb, counts, 1.0, AnswerMode.VALUE)
    p1 = BayesPolicy(membership_table(toy_kb, 0.5, 1e-2), uniform_prior(toy_kb), 0.9)
    majority = ProfileMajority.from_patients(toy_kb, [toy_patient(["douleur"], "douleur")])
    declared: list[tuple[Referenced, frozenset[StartInput], AnswerMode | None]] = [
        (MostFrequentDisease.from_counts(toy_kb, counts), frozenset(), None),
        (initial_only(p1), initial, AnswerMode.BINARY),
        (SetInclusion(toy_kb), initial, AnswerMode.BINARY),
        (majority, initial, AnswerMode.BINARY),
        (FullProfile(p1.table, p1.prior), initial, AnswerMode.BINARY),
        (FullProfile(value, frequency_prior(counts, 1.0, by_age_sex=True)), frozenset(StartInput), AnswerMode.VALUE),
    ]
    for policy, inputs, mode in declared:
        assert (policy.start_inputs, policy.answer_mode) == (inputs, mode), policy
