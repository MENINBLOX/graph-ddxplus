"""Bayes inquiry with greedy EIG question selection (design §3).

P1 and P2 are this one engine with different likelihood tables and priors: P1 knows only whether an
evidence is on a disease's list (`membership_table`, `uniform_prior`), P2 uses train minus H2 frequencies
(`frequency_table`, `frequency_prior`). Answers are conditionally independent given the disease. The
baselines (`ddxkb.policies.baselines`) keep the engine and swap the question `Selector`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ddxkb.data import Patient
from ddxkb.frequencies import FrequencyCounts, age_bin, sex_index, value_columns
from ddxkb.kb import KB, BoolArray, DataType, FloatArray
from ddxkb.policies.episode import Episode
from ddxkb.simulator import MAX_IL, Answer, AnswerMode, Interaction, Oracle

IndexArray = npt.NDArray[np.intp]

# EIGs closer than this (nats) are one tie, broken by KB evidence order. Each EIG is a difference of
# entropies of order 1, so a smaller gap is rounding, and the BLAS build would otherwise pick the question.
EIG_TIE_ATOL = 1e-12


@dataclass(frozen=True, eq=False)
class LikelihoodTable:
    """P(outcome | disease) for every question; diseases and evidences in KB order.

    An evidence is asked as one or more components, each with two or more outcome columns whose
    probabilities sum to 1 for every disease. A binary question is one [yes, no] component. In value mode
    a categorical evidence is one component over its catalogue values (the default value is the negative
    answer), and a multi-choice evidence has one [chosen, not chosen] component per non-default option.
    """

    kb: KB
    mode: AnswerMode
    probs: FloatArray
    component_start: IndexArray
    evidence_start: IndexArray
    log_probs: FloatArray = field(init=False)
    cond_entropy: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        d, n_outcomes = self.probs.shape
        if d != self.kb.n_diseases or len(self.evidence_start) != self.kb.n_evidences:
            raise ValueError("table shape does not match the KB")
        for name, starts, end in (
            ("component_start", self.component_start, n_outcomes),
            ("evidence_start", self.evidence_start, len(self.component_start)),
        ):
            if starts[0] != 0 or (np.diff(starts) <= 0).any() or starts[-1] >= end:
                raise ValueError(f"{name} must rise strictly from 0 inside its axis")
        # Open interval: a zero would make a single answer rule a disease out for good.
        if not ((self.probs > 0) & (self.probs < 1)).all():
            raise ValueError("likelihoods must lie strictly between 0 and 1")
        if not np.allclose(np.add.reduceat(self.probs, self.component_start, axis=1), 1.0, rtol=0, atol=1e-9):
            raise ValueError("outcome probabilities of a component must sum to 1")
        log_probs = np.log(self.probs)
        for a in (self.probs, self.component_start, self.evidence_start, log_probs):
            a.flags.writeable = False
        object.__setattr__(self, "log_probs", log_probs)
        cond = -np.add.reduceat(self.probs * log_probs, self.component_start, axis=1)
        cond.flags.writeable = False
        object.__setattr__(self, "cond_entropy", cond)

    def outcome_columns(self, answer: Answer) -> list[int]:
        """The observed outcome column of each component of the answered evidence."""
        j = self.kb.evidence_index[answer.code]
        e = self.kb.evidences[j]
        first = int(self.evidence_start[j])
        start = int(self.component_start[first])
        if self.mode is AnswerMode.BINARY or e.is_binary:
            return [start + (0 if answer.positive else 1)]
        if e.data_type is DataType.CATEGORICAL:
            if answer.positive and len(answer.values) != 1:
                raise ValueError(f"categorical {e.code} answered with {len(answer.values)} values")
            value = answer.values[0] if answer.positive else e.default
            return [start + e.values.index(value)]
        chosen = set(answer.values)
        options = [v for v in e.values if v != e.default]
        return [int(self.component_start[first + i]) + (0 if v in chosen else 1) for i, v in enumerate(options)]

    def log_likelihood(self, answer: Answer) -> FloatArray:
        cols = self.outcome_columns(answer)
        return np.asarray(self.log_probs[:, cols].sum(axis=1))


def _assemble(kb: KB, mode: AnswerMode, blocks: list[list[FloatArray]]) -> LikelihoodTable:
    """`blocks[j]` lists evidence j's components as (diseases x outcomes) arrays."""
    components = [c for block in blocks for c in block]
    component_sizes = [c.shape[1] for c in components]
    return LikelihoodTable(
        kb=kb,
        mode=mode,
        probs=np.concatenate(components, axis=1),
        component_start=np.cumsum([0, *component_sizes[:-1]], dtype=np.intp),
        evidence_start=np.cumsum([0, *[len(b) for b in blocks[:-1]]], dtype=np.intp),
    )


def _yes_no(p_yes: FloatArray) -> FloatArray:
    return np.stack([p_yes, 1.0 - p_yes], axis=1)


def membership_table(kb: KB, alpha: float, eps: float) -> LikelihoodTable:
    """P1: P(yes | d) = alpha when the evidence is on d's list, eps otherwise; binary answers only."""
    if not (0 < alpha < 1 and 0 < eps < 1):
        raise ValueError("alpha and eps must lie strictly between 0 and 1")
    yes = np.where(kb.membership, alpha, eps)
    return _assemble(kb, AnswerMode.BINARY, [[_yes_no(yes[:, j])] for j in range(kb.n_evidences)])


def frequency_table(kb: KB, counts: FrequencyCounts, lam: float, mode: AnswerMode) -> LikelihoodTable:
    """P2: train frequencies with Laplace smoothing lam, each outcome of a component getting lam pseudo-counts."""
    if lam <= 0:
        raise ValueError("lam must be positive")
    n = counts.rows.astype(np.float64)
    if mode is AnswerMode.BINARY:
        yes = (counts.positive + lam) / (n[:, None] + 2 * lam)
        return _assemble(kb, mode, [[_yes_no(yes[:, j])] for j in range(kb.n_evidences)])
    column = {pair: c for c, pair in enumerate(value_columns(kb))}
    blocks: list[list[FloatArray]] = []
    for j, e in enumerate(kb.evidences):
        if e.is_binary:
            blocks.append([_yes_no((counts.positive[:, j] + lam) / (n + 2 * lam))])
        elif e.data_type is DataType.CATEGORICAL:
            held = np.stack(
                [np.zeros_like(n) if v == e.default else counts.value[:, column[(j, v)]] for v in e.values], axis=1
            )
            held[:, e.values.index(e.default)] = n - held.sum(axis=1)
            if (held < 0).any():
                raise ValueError(f"{e.code}: records hold more than one value of a categorical evidence")
            blocks.append([np.asarray((held + lam) / (n[:, None] + len(e.values) * lam), dtype=np.float64)])
        else:
            blocks.append(
                [_yes_no((counts.value[:, column[(j, v)]] + lam) / (n + 2 * lam)) for v in e.values if v != e.default]
            )
    return _assemble(kb, mode, blocks)


@dataclass(frozen=True, eq=False)
class Prior:
    """log P(d) in KB order, shape (diseases,), or log P(d | age bin, sex), shape (age bins, sexes, diseases).

    Age and sex enter only here; the likelihood tables are per disease.
    """

    log_p: FloatArray

    def __post_init__(self) -> None:
        self.log_p.flags.writeable = False

    def __call__(self, patient: Patient) -> FloatArray:
        if self.log_p.ndim == 1:
            return self.log_p
        return np.asarray(self.log_p[age_bin(patient.age), sex_index(patient.sex)])


def uniform_prior(kb: KB) -> Prior:
    return Prior(np.full(kb.n_diseases, -math.log(kb.n_diseases)))


def frequency_prior(counts: FrequencyCounts, lam: float, by_age_sex: bool = False) -> Prior:
    if lam <= 0:
        raise ValueError("lam must be positive")
    n_diseases = counts.rows.shape[0]
    if not by_age_sex:
        return Prior(np.log((counts.rows + lam) / (counts.rows.sum() + n_diseases * lam)))
    cells = np.moveaxis(counts.by_age_sex, 0, -1).astype(np.float64)
    return Prior(np.log((cells + lam) / (cells.sum(axis=-1, keepdims=True) + n_diseases * lam)))


def expected_information_gain(table: LikelihoodTable, posterior: FloatArray) -> FloatArray:
    """EIG of every evidence in nats, I(D; Y) = H(Y) - H(Y | D) per component.

    A multi-choice evidence in value mode scores its most informative option (CASANDE App. B.3): the exact
    gain over all option sets is intractable for 164 body locations.
    """
    p = posterior @ table.probs
    h_answer = -np.add.reduceat(p * np.log(p), table.component_start)
    gain = h_answer - posterior @ table.cond_entropy
    return np.asarray(np.maximum.reduceat(gain, table.evidence_start))


def select_question(gain: FloatArray, asked: BoolArray) -> int:
    """The unasked evidence with the largest EIG, ties going to the first in KB order."""
    open_gain = np.where(asked, -np.inf, gain)
    best = open_gain.max()
    if not np.isfinite(best):
        raise RuntimeError("every evidence has been asked")
    return int(np.argmax(open_gain >= best - EIG_TIE_ATOL))


QuestionChoice = Callable[[FloatArray, BoolArray, BoolArray, FloatArray], int]
"""(posterior, asked mask, positive-answer mask, EIG of every evidence) -> index of the next evidence to ask.
The masks are read-only views of the episode state and include the initial evidence; the EIG is
`expected_information_gain` of the posterior, asked evidences included."""


class Selector(Protocol):
    def start(self, table: LikelihoodTable, patient: Patient) -> QuestionChoice:
        """The question choice of one episode, created when the episode starts."""
        ...


@dataclass(frozen=True)
class GreedyEIG:
    """P1 and P2: the unasked evidence with the largest EIG (`select_question`)."""

    def start(self, table: LikelihoodTable, patient: Patient) -> QuestionChoice:
        def choose(posterior: FloatArray, asked: BoolArray, positive: BoolArray, gain: FloatArray) -> int:
            return select_question(gain, asked)

        return choose


def normalize(log_posterior: FloatArray) -> FloatArray:
    w = np.exp(log_posterior - log_posterior.max())
    return np.asarray(w / w.sum())


def best_open_gain(gain: FloatArray, asked: BoolArray) -> float:
    """The largest EIG among the unasked evidences, 0 when every evidence has been asked."""
    open_gain = gain[~asked]
    return max(float(open_gain.max()), 0.0) if open_gain.size else 0.0


@dataclass(frozen=True, eq=False)
class BayesPolicy:
    """Inquiry over the unasked evidences with a Bayes posterior; questions come from `selector`.

    The episode stops at the first |T| where the max posterior is >= tau, or no unasked evidence has an
    EIG >= min_gain, or |T| = max_il, or a KB with fewer evidences than max_il has none left to ask; None
    turns the tau or min_gain condition off. For each budget b the posterior after exactly b questions and
    the largest unasked EIG there are recorded. Question choice never depends on tau or min_gain, so one
    trajectory, continued past the stop, gives both the stopped result and the budget curve.
    """

    table: LikelihoodTable
    prior: Prior
    tau: float | None
    budgets: tuple[int, ...] = ()
    max_il: int = MAX_IL
    selector: Selector = GreedyEIG()
    min_gain: float | None = None

    def __post_init__(self) -> None:
        if self.max_il < 1:
            raise ValueError("max_il must be at least 1")
        if self.tau is not None and not 0 < self.tau <= 1:
            raise ValueError("tau must lie in (0, 1]")
        if self.min_gain is not None and not self.min_gain > 0:
            raise ValueError("min_gain must be positive")
        if list(self.budgets) != sorted(set(self.budgets)):
            raise ValueError("budgets must be strictly increasing")
        if self.budgets and not 0 <= self.budgets[0] <= self.budgets[-1] < self.length_cap:
            raise ValueError(f"budgets count questions after the initial evidence: 0 to {self.length_cap - 1}")

    @property
    def length_cap(self) -> int:
        return min(self.max_il, self.table.kb.n_evidences)

    def run(self, patient: Patient) -> Episode:
        kb = self.table.kb
        it = Interaction(Oracle(kb, patient, self.table.mode), self.length_cap)
        choose = self.selector.start(self.table, patient)
        asked = np.zeros(kb.n_evidences, dtype=np.bool_)
        positive = np.zeros(kb.n_evidences, dtype=np.bool_)
        first = kb.evidence_index[patient.initial]
        asked[first] = positive[first] = True
        asked_view, positive_view = asked.view(), positive.view()
        asked_view.flags.writeable = positive_view.flags.writeable = False
        log_post = self.prior(patient) + self.table.log_likelihood(it.initial)
        last_budget = self.budgets[-1] if self.budgets else 0
        snapshots: list[FloatArray] = []
        gains: list[float] = []
        stop: tuple[int, FloatArray] | None = None
        while True:
            q = normalize(log_post)
            gain = expected_information_gain(self.table, q)
            open_gain = best_open_gain(gain, asked)
            n_questions = it.il - 1
            if n_questions in self.budgets:
                snapshots.append(q)
                gains.append(open_gain)
            if stop is None and self._stops(it.remaining, float(q.max()), open_gain):
                stop = (it.il, q)
            if stop is not None and n_questions >= last_budget:
                break
            j = choose(q, asked_view, positive_view, gain)
            answer = it.ask(kb.evidences[j].code)
            asked[j] = True
            positive[j] = answer.positive
            log_post = log_post + self.table.log_likelihood(answer)
        return Episode(
            row=patient.row,
            gold=kb.disease_index[patient.pathology],
            questions=tuple(a.code for a in it.answers[1:]),
            il=stop[0],
            posterior=stop[1],
            budgets=self.budgets,
            budget_posteriors=np.stack(snapshots) if snapshots else np.empty((0, kb.n_diseases)),
            budget_gains=np.asarray(gains, dtype=np.float64),
        )

    def _stops(self, remaining: int, peak: float, open_gain: float) -> bool:
        return (
            remaining == 0
            or (self.tau is not None and peak >= self.tau)
            or (self.min_gain is not None and open_gain < self.min_gain)
        )
