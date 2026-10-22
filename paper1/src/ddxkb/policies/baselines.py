"""Baselines and reference points of design §3.

- B0, no questions: `MostFrequentDisease` (train minus H2 prior) and `initial_only` (posterior from the
  initial evidence alone; with P1's table every disease listing it ties).
- B1, full profile at no question cost: `SetInclusion`, and `FullProfile` with P1's table (KB-Bayes) or
  P2's (naive Bayes). B1 is a reference point, not an upper bound.
- B2, `random_questions`: random questions from the lists of the KB candidate diseases, P1 ranking and stop.
- B6, `train_frequency_questions`: one KB-free question order, ranking and stop of P1 or P2.

Policies without questions return an episode with |T| = 1 and no questions; their IL is not a result.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from ddxkb.data import Patient
from ddxkb.frequencies import FrequencyCounts
from ddxkb.kb import KB, BoolArray, FloatArray
from ddxkb.policies.bayes import BayesPolicy, IndexArray, LikelihoodTable, Prior, QuestionChoice, normalize
from ddxkb.policies.episode import Episode
from ddxkb.simulator import Oracle, positive_codes

B2_SEEDS = (0, 1, 2, 3, 4)


def _no_question_episode(kb: KB, patient: Patient, posterior: FloatArray) -> Episode:
    return Episode(
        row=patient.row,
        gold=kb.disease_index[patient.pathology],
        questions=(),
        il=1,
        posterior=posterior,
        budgets=(),
        budget_posteriors=np.empty((0, kb.n_diseases)),
        budget_gains=np.empty(0),
    )


@dataclass(frozen=True, eq=False)
class MostFrequentDisease:
    """B0: the train disease frequencies as the differential for every patient."""

    kb: KB
    frequencies: FloatArray

    def __post_init__(self) -> None:
        if self.frequencies.shape != (self.kb.n_diseases,) or not np.isclose(self.frequencies.sum(), 1.0):
            raise ValueError("frequencies must be one distribution over the KB diseases")
        self.frequencies.flags.writeable = False

    @classmethod
    def from_counts(cls, kb: KB, counts: FrequencyCounts) -> MostFrequentDisease:
        rows = counts.rows.astype(np.float64)
        return cls(kb, rows / rows.sum())

    def run(self, patient: Patient) -> Episode:
        return _no_question_episode(self.kb, patient, self.frequencies)


def initial_only(policy: BayesPolicy) -> BayesPolicy:
    """B0: `policy`'s posterior after the initial evidence, with no question asked."""
    return dataclasses.replace(policy, tau=None, min_gain=None, budgets=(), max_il=1)


@dataclass(frozen=True, eq=False)
class SetInclusion:
    """B1: uniform over the diseases whose list holds every positive evidence of the record.

    The vector is all zero when no disease does, and the patient then scores 0 on every metric.
    """

    kb: KB

    def run(self, patient: Patient) -> Episode:
        consistent = self.kb.diseases_containing(positive_codes(self.kb, patient))
        n = int(np.count_nonzero(consistent))
        posterior = consistent / n if n else np.zeros(self.kb.n_diseases)
        return _no_question_episode(self.kb, patient, posterior)


@dataclass(frozen=True, eq=False)
class FullProfile:
    """B1: the posterior given the answer to every evidence of the catalogue, in the table's answer mode."""

    table: LikelihoodTable
    prior: Prior

    def run(self, patient: Patient) -> Episode:
        kb = self.table.kb
        oracle = Oracle(kb, patient, self.table.mode)
        columns = [c for e in kb.evidences for c in self.table.outcome_columns(oracle.answer(e.code))]
        log_post = self.prior(patient) + self.table.log_probs[:, columns].sum(axis=1)
        return _no_question_episode(kb, patient, normalize(log_post))


@dataclass(frozen=True)
class RandomInKBCandidates:
    """B2 question choice: a uniformly random unasked evidence on the list of a KB candidate disease.

    Candidates are the diseases with the fewest positive answers off their list so far; on DDXPlus, where no
    record has a positive evidence off its disease's list, these are the diseases listing every positive
    answer. Once every evidence on their lists has been asked, the draw is over all unasked evidences.
    Each episode draws from its own generator seeded with (seed, patient row), so a patient's questions do
    not depend on the other patients or on how a run splits them over processes.
    """

    seed: int

    def start(self, table: LikelihoodTable, patient: Patient) -> QuestionChoice:
        membership = table.kb.membership
        off_list = ~membership
        rng = np.random.default_rng([self.seed, patient.row])

        def choose(posterior: FloatArray, asked: BoolArray, positive: BoolArray, gain: FloatArray) -> int:
            violations = (off_list & positive).sum(axis=1)
            candidates = violations == violations.min()
            pool = np.flatnonzero(membership[candidates].any(axis=0) & ~asked)
            if pool.size == 0:
                pool = np.flatnonzero(~asked)
            return int(pool[rng.integers(pool.size)])

        return choose


@dataclass(frozen=True)
class FixedOrder:
    """Asks the evidences in one order fixed before any answer, skipping those already asked."""

    order: tuple[int, ...]
    _order: IndexArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if sorted(self.order) != list(range(len(self.order))):
            raise ValueError("order must be a permutation of the evidence indices")
        order = np.asarray(self.order, dtype=np.intp)
        order.flags.writeable = False
        object.__setattr__(self, "_order", order)

    @classmethod
    def by_train_frequency(cls, counts: FrequencyCounts) -> FixedOrder:
        """B6: the evidence present in the most train-minus-H2 records first, whatever the disease; ties in
        KB order."""
        totals = counts.positive.sum(axis=0)
        return cls(tuple(int(j) for j in np.argsort(-totals, kind="stable")))

    def start(self, table: LikelihoodTable, patient: Patient) -> QuestionChoice:
        if len(self.order) != table.kb.n_evidences:
            raise ValueError(f"order has {len(self.order)} evidences, the KB {table.kb.n_evidences}")
        order = self._order

        def choose(posterior: FloatArray, asked: BoolArray, positive: BoolArray, gain: FloatArray) -> int:
            unasked = order[~asked[order]]
            if unasked.size == 0:
                raise RuntimeError("every evidence has been asked")
            return int(unasked[0])

        return choose


def random_questions(p1: BayesPolicy, seed: int) -> BayesPolicy:
    """B2: `p1`'s table, prior, stop rules and budgets with random questions from the KB candidates."""
    return dataclasses.replace(p1, selector=RandomInKBCandidates(seed))


def train_frequency_questions(policy: BayesPolicy, counts: FrequencyCounts) -> BayesPolicy:
    """B6: `policy` (P1 or P2) asking in the order of `FixedOrder.by_train_frequency`."""
    return dataclasses.replace(policy, selector=FixedOrder.by_train_frequency(counts))
