from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import zipfile
from typing import Any

import numpy as np
import pytest

from ddxkb.data import KB_ZIP, REPO_ROOT, Patient
from ddxkb.evaluation import Scored, ScoreTask
from ddxkb.frequencies import N_AGE_BINS
from ddxkb.kb import KB, BoolArray, FloatArray
from ddxkb.policies.bayes import (
    BayesPolicy,
    LikelihoodTable,
    QuestionChoice,
    best_open_gain,
    expected_information_gain,
    frequency_prior,
    frequency_table,
    membership_table,
    select_question,
    uniform_prior,
)
from ddxkb.protocol import MIN_GAIN
from ddxkb.runner import chunks, map_items, worker_pool
from ddxkb.simulator import Answer, AnswerMode
from helpers import binary_kb, toy_counts, toy_patient

BUDGETS = (0, 5, 10, 15, 20, 29)


def _h(*p: float) -> float:
    return -sum(x * math.log(x) for x in p)


# e1 splits {A, B} from {C, D}; e2 and e3 single out A and B; e0 and e4 are on every list.
QUAD = {"A": ["e0", "e1", "e2", "e4"], "B": ["e0", "e1", "e3", "e4"], "C": ["e0", "e4"], "D": ["e0", "e4"]}
QUAD_ORDER = ["e0", "e1", "e2", "e3", "e4"]


@pytest.fixture
def quad() -> KB:
    return binary_kb(QUAD, QUAD_ORDER)


def _patient_a() -> Patient:
    return toy_patient(["e0", "e1", "e2", "e4"], initial="e0", pathology="A")


def _policy(
    kb: KB, tau: float | None, budgets: tuple[int, ...] = (), max_il: int = 5, min_gain: float | None = None
) -> BayesPolicy:
    return BayesPolicy(membership_table(kb, 0.8, 0.2), uniform_prior(kb), tau, budgets, max_il, min_gain=min_gain)


def _naive_eig(q: FloatArray, outcome_probs: FloatArray) -> float:
    """H(D) - sum_y P(y) H(D | y) for one component given (diseases x outcomes) probabilities."""
    h_d = _h(*q)
    expected = 0.0
    for y in range(outcome_probs.shape[1]):
        joint = q * outcome_probs[:, y]
        p_y = joint.sum()
        expected += p_y * _h(*(joint / p_y))
    return h_d - expected


def test_membership_table_is_alpha_inside_eps_outside(quad: KB) -> None:
    t = membership_table(quad, 0.8, 0.2)
    assert t.mode is AnswerMode.BINARY
    assert t.probs.shape == (4, 10)
    assert t.probs[:, 2].tolist() == [0.8, 0.8, 0.2, 0.2]
    assert t.probs[:, 3] == pytest.approx([0.2, 0.2, 0.8, 0.8])
    assert t.outcome_columns(Answer("e1", positive=True)) == [2]
    assert t.outcome_columns(Answer("e1", positive=False)) == [3]
    with pytest.raises(ValueError):
        membership_table(quad, 1.0, 0.2)


def test_eig_matches_the_closed_form(quad: KB) -> None:
    gain = expected_information_gain(membership_table(quad, 0.8, 0.2), np.full(4, 0.25))
    h80 = _h(0.8, 0.2)
    expected = [0.0, _h(0.5, 0.5) - h80, _h(0.35, 0.65) - h80, _h(0.35, 0.65) - h80, 0.0]
    assert gain == pytest.approx(expected, abs=1e-12)


def test_eig_matches_the_definition_for_any_posterior(quad: KB) -> None:
    t = membership_table(quad, 0.7, 0.05)
    q = np.array([0.5, 0.3, 0.15, 0.05])
    gain = expected_information_gain(t, q)
    for j in range(quad.n_evidences):
        assert gain[j] == pytest.approx(_naive_eig(q, t.probs[:, 2 * j : 2 * j + 2]), abs=1e-12)


def test_selection_skips_asked_and_breaks_ties_by_kb_order() -> None:
    gain = np.array([0.1, 0.3, 0.3 + 1e-14, 0.2])
    assert select_question(gain, np.array([False, False, False, False])) == 1
    assert select_question(gain, np.array([False, True, False, False])) == 2
    with pytest.raises(RuntimeError):
        select_question(gain, np.ones(4, dtype=bool))


def test_hand_computed_episode(quad: KB) -> None:
    ep = _policy(quad, tau=0.9).run(_patient_a())
    # e1 splits the four diseases in half; then e2 and e3 tie and e2 comes first in the file; e4 is
    # uninformative and left for last. tau = 0.9 is never reached, so |T| hits max_il = 5.
    assert ep.questions == ("e1", "e2", "e3", "e4")
    assert ep.il == 5
    assert ep.asked == ep.questions
    assert ep.posterior == pytest.approx([16 / 19, 1 / 19, 1 / 19, 1 / 19])
    assert ep.gold == 0


def test_kb_order_decides_a_tied_question() -> None:
    kb = binary_kb(QUAD, ["e0", "e1", "e3", "e2", "e4"])
    ep = _policy(kb, tau=0.9).run(_patient_a())
    assert ep.questions[:2] == ("e1", "e3")


def test_stops_once_the_posterior_reaches_tau(quad: KB) -> None:
    ep = _policy(quad, tau=0.8).run(_patient_a())
    assert (ep.il, ep.questions) == (4, ("e1", "e2", "e3"))
    assert ep.posterior.max() == pytest.approx(16 / 19)


def test_budget_curve_bookkeeping(quad: KB) -> None:
    ep = _policy(quad, tau=0.8, budgets=(0, 1, 2, 3, 4)).run(_patient_a())
    # Budgets run past the stop on the same trajectory; the stopped result is unchanged.
    assert (ep.il, ep.questions) == (4, ("e1", "e2", "e3", "e4"))
    assert ep.asked == ("e1", "e2", "e3")
    expected = [
        [0.25, 0.25, 0.25, 0.25],
        [0.4, 0.4, 0.1, 0.1],
        [8 / 11, 2 / 11, 1 / 22, 1 / 22],
        [16 / 19, 1 / 19, 1 / 19, 1 / 19],
        [16 / 19, 1 / 19, 1 / 19, 1 / 19],
    ]
    assert ep.budget_posteriors == pytest.approx(np.array(expected))
    assert np.array_equal(ep.at_budget(3), ep.posterior)
    no_stop = _policy(quad, tau=None, budgets=(2,)).run(_patient_a())
    assert np.array_equal(no_stop.at_budget(2), ep.at_budget(2))
    assert no_stop.il == 5


def test_budget_tau_and_min_gain_validation(quad: KB) -> None:
    with pytest.raises(ValueError):
        _policy(quad, tau=0.9, budgets=(5,), max_il=5)
    with pytest.raises(ValueError):
        _policy(quad, tau=0.9, budgets=(2, 1))
    with pytest.raises(ValueError):
        _policy(quad, tau=0.0)
    with pytest.raises(ValueError, match="min_gain"):
        _policy(quad, tau=0.9, min_gain=0.0)


def test_stops_once_no_unasked_question_has_gain(quad: KB) -> None:
    # After e1, e2 and e3 only e4 is left, on every list and so worth nothing: the episode stops at |T| = 4
    # below tau, where without the gain rule it asks e4 and stops at max_il.
    ep = _policy(quad, tau=0.9, min_gain=1e-6).run(_patient_a())
    assert (ep.il, ep.questions) == (4, ("e1", "e2", "e3"))
    assert ep.posterior == pytest.approx([16 / 19, 1 / 19, 1 / 19, 1 / 19])
    assert _policy(quad, tau=0.9).run(_patient_a()).il == 5
    # A large min_gain stops before the first question: e1 is worth ln 2 - H(0.8) = 0.193 nats.
    assert _policy(quad, tau=0.9, min_gain=0.2).run(_patient_a()).il == 1
    assert _policy(quad, tau=0.9, min_gain=0.19).run(_patient_a()).il > 1


def test_budget_gains_are_the_best_unasked_eig(quad: KB) -> None:
    table = membership_table(quad, 0.8, 0.2)
    ep = _policy(quad, tau=None, budgets=(0, 1, 2, 3, 4)).run(_patient_a())
    asked = [0]
    for k, q in enumerate(ep.budget_posteriors):
        open_gains = [_naive_eig(q, table.probs[:, 2 * j : 2 * j + 2]) for j in range(5) if j not in asked]
        expected = max(max(open_gains), 0.0) if open_gains else 0.0
        assert ep.budget_gains[k] == pytest.approx(expected, abs=1e-12)
        if k < len(ep.questions):
            asked.append(QUAD_ORDER.index(ep.questions[k]))
    assert ep.budget_gains[0] == pytest.approx(_h(0.5, 0.5) - _h(0.8, 0.2))
    assert ep.budget_gains[3] < 1e-12 and ep.budget_gains[4] == 0.0


def test_best_open_gain() -> None:
    gain = np.array([0.5, 0.2, -1e-17])
    assert best_open_gain(gain, np.array([True, False, False])) == 0.2
    assert best_open_gain(gain, np.array([True, True, False])) == 0.0
    assert best_open_gain(gain, np.ones(3, dtype=bool)) == 0.0


@dataclasses.dataclass(frozen=True)
class _Scripted:
    """Asks evidence indices in the given order and records the state it is shown."""

    order: tuple[int, ...]
    seen: list[tuple[list[bool], list[bool]]] = dataclasses.field(default_factory=list, compare=False)

    def start(self, table: LikelihoodTable, patient: Patient) -> QuestionChoice:
        it = iter(self.order)

        def choose(posterior: FloatArray, asked: BoolArray, positive: BoolArray, gain: FloatArray) -> int:
            assert not asked.flags.writeable and not positive.flags.writeable
            assert np.array_equal(gain, expected_information_gain(table, posterior))
            self.seen.append((asked.tolist(), positive.tolist()))
            return next(it)

        return choose


def test_a_selector_sees_the_asked_and_positive_masks(quad: KB) -> None:
    script = _Scripted((3, 1))
    pol = BayesPolicy(membership_table(quad, 0.8, 0.2), uniform_prior(quad), None, (), 3, script)
    ep = pol.run(_patient_a())
    assert ep.questions == ("e3", "e1")
    # Patient A has e0, e1, e2 and e4; e3 is answered no.
    assert script.seen == [
        ([True, False, False, False, False], [True, False, False, False, False]),
        ([True, False, False, True, False], [True, False, False, False, False]),
    ]


def test_a_selector_cannot_repeat_a_question(quad: KB) -> None:
    pol = BayesPolicy(membership_table(quad, 0.8, 0.2), uniform_prior(quad), None, (), 3, _Scripted((2, 2)))
    with pytest.raises(ValueError, match="already asked"):
        pol.run(_patient_a())


def test_same_output_twice(toy_kb: KB) -> None:
    pol = BayesPolicy(membership_table(toy_kb, 0.5, 1e-3), uniform_prior(toy_kb), 0.99, (0, 1, 3))
    p = toy_patient(["douleur", "douleur_intens_@_2", "douleur_lieu_@_dos", "atcd", "trav1_@_N"], initial="douleur")
    a, b = pol.run(p), pol.run(p)
    assert a.questions == b.questions and a.il == b.il
    assert np.array_equal(a.posterior, b.posterior)
    assert np.array_equal(a.budget_posteriors, b.budget_posteriors)


def test_frequency_table_binary_mode(toy_kb: KB) -> None:
    t = frequency_table(toy_kb, toy_counts(), 1.0, AnswerMode.BINARY)
    assert t.probs.shape == (3, 14)
    assert t.probs[:, 0] == pytest.approx([10 / 12, 1 / 10, 3 / 4])
    # douleur_lieu in binary mode: any non-default location.
    assert t.probs[:, 8] == pytest.approx([1 / 12, 7 / 10, 3 / 4])


def test_frequency_table_value_mode(toy_kb: KB) -> None:
    t = frequency_table(toy_kb, toy_counts(), 1.0, AnswerMode.VALUE)
    # Components: 3 binaries, intens (4 values), lieu (3 options), trav1 (3 values), atcd.
    assert t.evidence_start.tolist() == [0, 1, 2, 3, 4, 7, 8]
    intens = t.probs[:, t.component_start[3] : t.component_start[3] + 4]
    # Beta: 8 rows, values 1/2/3 held by 1/3/4, so the default 0 by none; lam = 1 over 4 values.
    assert intens[1] == pytest.approx([1 / 12, 2 / 12, 4 / 12, 5 / 12])
    assert intens[0] == pytest.approx([11 / 14, 1 / 14, 1 / 14, 1 / 14])
    tete = t.probs[:, t.component_start[4] : t.component_start[4] + 2]
    assert tete[1] == pytest.approx([5 / 10, 5 / 10])
    assert tete[2] == pytest.approx([3 / 4, 1 / 4])


def test_value_answers_pick_one_column_per_component(toy_kb: KB) -> None:
    t = frequency_table(toy_kb, toy_counts(), 1.0, AnswerMode.VALUE)
    lieu = int(t.evidence_start[4])
    s = [int(t.component_start[lieu + i]) for i in range(3)]
    assert t.outcome_columns(Answer("douleur_lieu", True, ("tete", "ventre"))) == [s[0], s[1] + 1, s[2]]
    assert t.outcome_columns(Answer("douleur_lieu", False)) == [s[0] + 1, s[1] + 1, s[2] + 1]
    intens = int(t.component_start[3])
    assert t.outcome_columns(Answer("douleur_intens", True, ("2",))) == [intens + 2]
    assert t.outcome_columns(Answer("douleur_intens", False)) == [intens]
    ll = t.log_likelihood(Answer("douleur_lieu", True, ("tete", "ventre")))
    assert ll == pytest.approx(np.log(t.probs[:, s[0]] * t.probs[:, s[1] + 1] * t.probs[:, s[2]]))
    with pytest.raises(ValueError, match="categorical"):
        t.outcome_columns(Answer("douleur_intens", True, ("1", "2")))


def test_value_mode_eig(toy_kb: KB) -> None:
    t = frequency_table(toy_kb, toy_counts(), 1.0, AnswerMode.VALUE)
    q = np.array([0.2, 0.5, 0.3])
    gain = expected_information_gain(t, q)
    c = int(t.component_start[3])
    assert gain[3] == pytest.approx(_naive_eig(q, t.probs[:, c : c + 4]), abs=1e-12)
    first = int(t.evidence_start[4])
    options = [_naive_eig(q, t.probs[:, t.component_start[first + i] :][:, :2]) for i in range(3)]
    assert gain[4] == pytest.approx(max(options), abs=1e-12)


def test_both_answer_modes_run_a_frequency_episode(toy_kb: KB) -> None:
    counts = toy_counts()
    p = toy_patient(["douleur", "douleur_intens_@_3", "douleur_lieu_@_dos", "atcd", "trav1_@_N"], initial="douleur")
    for mode in AnswerMode:
        pol = BayesPolicy(frequency_table(toy_kb, counts, 0.5, mode), frequency_prior(counts, 0.5), None, (), 4)
        ep = pol.run(p)
        assert ep.il == 4 and len(set(ep.questions)) == 3


def test_frequency_prior(toy_kb: KB) -> None:
    counts = toy_counts()
    flat = frequency_prior(counts, 1.0)
    assert np.exp(flat.log_p) == pytest.approx([11 / 23, 9 / 23, 3 / 23])
    by = frequency_prior(counts, 1.0, by_age_sex=True)
    assert by.log_p.shape == (N_AGE_BINS, 2, 3)
    boy = dataclasses.replace(toy_patient(["fievre"], initial="fievre"), age=1, sex="M")
    # Age bin 0, male: only Beta's 2 rows.
    assert np.exp(by(boy)) == pytest.approx([1 / 5, 3 / 5, 1 / 5])
    assert np.exp(uniform_prior(toy_kb).log_p) == pytest.approx([1 / 3] * 3)


def test_table_rejects_zero_likelihoods(quad: KB) -> None:
    t = membership_table(quad, 0.8, 0.2)
    probs = np.array(t.probs)
    probs[0, 0], probs[0, 1] = 1.0, 0.0
    with pytest.raises(ValueError, match="strictly"):
        LikelihoodTable(quad, AnswerMode.BINARY, probs, np.array(t.component_start), np.array(t.evidence_start))


def _sample(validation: list[Patient]) -> list[Patient]:
    """The 20,000-patient validation sample of reference/adv_bayes.py: random.Random(0).sample of the rows."""
    return [validation[i] for i in random.Random(0).sample(range(len(validation)), 20_000)]


def _p1(kb: KB, validation: list[Patient], min_gain: float | None = MIN_GAIN) -> tuple[float, float, list[float]]:
    """Tie-aware GTPA@1 (%) and mean IL at the stop, and the GTPA@1 curve (%) over BUDGETS."""
    pol = BayesPolicy(membership_table(kb, 0.5, 1e-3), uniform_prior(kb), 0.99, BUDGETS, min_gain=min_gain)
    processes = min(16, os.cpu_count() or 1)
    with worker_pool(processes) as pool:
        s = Scored.concat(map_items(ScoreTask(pol, kb, BUDGETS), chunks(_sample(validation), 4 * processes), pool))
    assert s.n == 20_000 and not s.failed.any()
    curve = [100 * float(s.column("gtpa1", b).mean()) for b in BUDGETS]
    return 100 * float(s.column("gtpa1").mean()), float(s.il.mean()), curve


def test_p1_on_the_review_sample(real_kb: KB, validation: list[Patient]) -> None:
    # Without the gain stop, 8% of these patients reach |T| = 30 below tau, asking questions worth nothing.
    g1, il, curve = _p1(real_kb, validation)
    assert (round(g1, 2), round(il, 2)) == (98.40, 11.65)
    assert [round(x, 2) for x in curve] == [25.47, 64.95, 85.30, 93.46, 96.86, 98.44]
    g1, il, _ = _p1(real_kb, validation, min_gain=None)
    assert (round(g1, 2), round(il, 2)) == (98.40, 12.15)


def test_p1_reproduces_the_prototype_under_its_tie_order(real_kb: KB, validation: list[Patient]) -> None:
    # reference/adv_bayes.py (98.39% / IL 11.46 on this sample) breaks EIG ties by code name, not KB order.
    with zipfile.ZipFile(REPO_ROOT / KB_ZIP) as z:
        conditions: dict[str, Any] = json.loads(z.read("release_conditions.json"))
        evidences: dict[str, Any] = json.loads(z.read("release_evidences.json"))
    by_name = KB.from_json(
        dict(sorted(conditions.items(), key=lambda kv: str(kv[1]["condition_name"]))), dict(sorted(evidences.items()))
    )
    g1, il, curve = _p1(by_name, validation)
    assert (round(g1, 2), round(il, 2)) == (98.39, 11.46)
    assert [round(x, 2) for x in curve] == [25.47, 64.85, 85.39, 93.85, 96.96, 98.42]
