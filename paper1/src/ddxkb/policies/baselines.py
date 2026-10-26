"""Baselines and reference points of design §3.

- B0, no questions: `MostFrequentDisease` (train prior) and `initial_only` (posterior from the initial
  evidence alone; with P1's table every disease listing it ties).
- B1, full profile at no question cost: `SetInclusion`, and `FullProfile` with P1's table (KB-Bayes) or
  P2's (naive Bayes). B1 is a reference point, not an upper bound. `ProfileMajority` adds the reference line
  of the data review (§4.8, §6.4): the resubstitution majority over the binary representation.
- B2, `random_questions`: random questions from the lists of the KB candidate diseases, P1 ranking and stop.
- B6, `train_frequency_questions`: one KB-free question order, ranking and stop of P1 or P2.

Policies without questions return an episode with |T| = 1 and no questions; their IL is not a result. Each
declares the start inputs it reads and whether it reads the revealed values (`answer_mode`, None when it reads no
answer), as the inquiry policies do.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from ddxkb.data import Patient
from ddxkb.frequencies import FrequencyCounts
from ddxkb.kb import KB, BoolArray, FloatArray, IndexArray
from ddxkb.policies.bayes import BayesPolicy, LikelihoodTable, Prior, QuestionChoice, normalize
from ddxkb.policies.episode import Episode
from ddxkb.simulator import AnswerMode, Oracle, Start, StartInput, positive_codes

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

    @property
    def start_inputs(self) -> frozenset[StartInput]:
        return frozenset()

    @property
    def answer_mode(self) -> AnswerMode | None:
        return None

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

    @property
    def start_inputs(self) -> frozenset[StartInput]:
        return frozenset({StartInput.INITIAL_EVIDENCE})

    @property
    def answer_mode(self) -> AnswerMode | None:
        return AnswerMode.BINARY

    def run(self, patient: Patient) -> Episode:
        consistent = self.kb.diseases_containing(positive_codes(self.kb, patient))
        n = int(np.count_nonzero(consistent))
        posterior = consistent / n if n else np.zeros(self.kb.n_diseases)
        return _no_question_episode(self.kb, patient, posterior)


@dataclass(frozen=True, eq=False)
class FullProfile:
    """B1: the posterior given the answer to every evidence of the catalogue, in the table's answer mode, and
    the age and sex when the prior reads them."""

    table: LikelihoodTable
    prior: Prior

    @property
    def start_inputs(self) -> frozenset[StartInput]:
        return frozenset({StartInput.INITIAL_EVIDENCE, *self.prior.inputs})

    @property
    def answer_mode(self) -> AnswerMode | None:
        return self.table.mode

    def run(self, patient: Patient) -> Episode:
        kb = self.table.kb
        oracle = Oracle(kb, patient)
        columns = [c for e in kb.evidences for c in self.table.outcome_columns(oracle.answer(e.code))]
        start = Start(oracle.answer(patient.initial), patient.age, patient.sex)
        log_post = self.prior(start) + self.table.log_probs[:, columns].sum(axis=1)
        return _no_question_episode(kb, patient, normalize(log_post))


def _positive_profile(kb: KB, patient: Patient) -> bytes:
    return np.packbits(kb.evidence_mask(positive_codes(kb, patient))).tobytes()


@dataclass(frozen=True, eq=False)
class ProfileMajority:
    """B1 reference line: the shares of the gold diseases among the patients of one set whose positive evidences
    are exactly the patient's, the patient included.

    Over the set it is built from, its GTPA@1 is the resubstitution accuracy of a majority vote on the binary
    representation (which evidences are positive), the line the data review reports (validate 99.52%) for policies
    that read no values. It reads the gold diseases of that set, so it is not a policy; a patient outside the set
    fails. Over a subset of that set it keeps the shares of the whole set, which is not the subset's own majority.
    """

    kb: KB
    shares: Mapping[bytes, tuple[tuple[int, int], ...]]

    @property
    def start_inputs(self) -> frozenset[StartInput]:
        return frozenset({StartInput.INITIAL_EVIDENCE})

    @property
    def answer_mode(self) -> AnswerMode | None:
        return AnswerMode.BINARY

    @classmethod
    def from_patients(cls, kb: KB, patients: Iterable[Patient]) -> ProfileMajority:
        counts: dict[bytes, dict[int, int]] = {}
        for p in patients:
            cell = counts.setdefault(_positive_profile(kb, p), {})
            d = kb.disease_index[p.pathology]
            cell[d] = cell.get(d, 0) + 1
        return cls(kb, {key: tuple(sorted(cell.items())) for key, cell in counts.items()})

    def run(self, patient: Patient) -> Episode:
        posterior = np.zeros(self.kb.n_diseases)
        for d, count in self.shares[_positive_profile(self.kb, patient)]:
            posterior[d] = count
        return _no_question_episode(self.kb, patient, posterior / posterior.sum())


@dataclass(frozen=True)
class RandomInKBCandidates:
    """B2 question choice: a uniformly random open evidence on the list of a KB candidate disease.

    Candidates are the diseases with the fewest positive answers off their list so far; on DDXPlus, where no
    record has a positive evidence off its disease's list, these are the diseases listing every positive
    answer. Once no evidence on their lists is open, the draw is over all open evidences.
    Each episode draws from its own generator seeded with (seed, patient row), so a patient's questions do
    not depend on the other patients or on how a run splits them over processes.
    """

    seed: int

    def start(self, table: LikelihoodTable, patient: Patient) -> QuestionChoice:
        membership = table.kb.membership
        off_list = ~membership
        rng = np.random.default_rng([self.seed, patient.row])

        def choose(posterior: FloatArray, closed: BoolArray, positive: BoolArray, gain: FloatArray) -> int:
            violations = (off_list & positive).sum(axis=1)
            candidates = violations == violations.min()
            pool = np.flatnonzero(membership[candidates].any(axis=0) & ~closed)
            if pool.size == 0:
                pool = np.flatnonzero(~closed)
            return int(pool[rng.integers(pool.size)])

        return choose


@dataclass(frozen=True)
class FixedOrder:
    """Asks the evidences in one order fixed before any answer, skipping closed ones."""

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
        """B6: the evidence present in the most train records first, whatever the disease; ties in KB order."""
        totals = counts.positive.sum(axis=0)
        return cls(tuple(int(j) for j in np.argsort(-totals, kind="stable")))

    def start(self, table: LikelihoodTable, patient: Patient) -> QuestionChoice:
        if len(self.order) != table.kb.n_evidences:
            raise ValueError(f"order has {len(self.order)} evidences, the KB {table.kb.n_evidences}")
        order = self._order

        def choose(posterior: FloatArray, closed: BoolArray, positive: BoolArray, gain: FloatArray) -> int:
            open_order = order[~closed[order]]
            if open_order.size == 0:
                raise RuntimeError("no evidence is left to ask")
            return int(open_order[0])

        return choose


def random_questions(p1: BayesPolicy, seed: int) -> BayesPolicy:
    """B2: `p1`'s table, prior, stop rules and budgets with random questions from the KB candidates."""
    return dataclasses.replace(p1, selector=RandomInKBCandidates(seed))


def train_frequency_questions(policy: BayesPolicy, counts: FrequencyCounts) -> BayesPolicy:
    """B6: `policy` (P1 or P2) asking in the order of `FixedOrder.by_train_frequency`."""
    return dataclasses.replace(policy, selector=FixedOrder.by_train_frequency(counts))
