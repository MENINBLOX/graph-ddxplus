"""Train frequencies for the frequency-Bayes policy P2 (design 보완 2, §5), counted from the whole train split
as BED counts them (Casande-RL `Baselines/BED/data_utils.py`).

A record value equal to the evidence's default counts as absent, as the oracle answers.
"""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ddxkb.data import Patient
from ddxkb.kb import KB
from ddxkb.simulator import present_values

IntArray = npt.NDArray[np.int64]

# The eight age groups of the official baselines: [0,1), 1-4, 5-14, 15-29, 30-44, 45-59, 60-74, 75+
# (`encode_age` in DDXPlus `basd/sim_utils.py` and CASANDE `chloe/utils/sim_utils.py`).
AGE_BIN_EDGES = (1, 5, 15, 30, 45, 60, 75)
N_AGE_BINS = len(AGE_BIN_EDGES) + 1
SEXES = ("F", "M")


def check_age_edges(edges: Sequence[int]) -> None:
    if not edges or edges[0] < 1 or any(b <= a for a, b in itertools.pairwise(edges)):
        raise ValueError(f"age bin edges must rise strictly from 1 or more: {list(edges)}")


def age_bin(age: int, edges: Sequence[int] = AGE_BIN_EDGES) -> int:
    """The age group of `age`: group k holds the ages from edges[k - 1] up to edges[k] - 1."""
    if age < 0:
        raise ValueError(f"negative age {age}")
    return bisect.bisect_right(edges, age)


def sex_index(sex: str) -> int:
    if sex not in SEXES:
        raise ValueError(f"unknown sex {sex!r}")
    return SEXES.index(sex)


def value_columns(kb: KB) -> tuple[tuple[int, str], ...]:
    """(evidence index, value) for every non-default value of a categorical or multi-choice evidence, in
    KB evidence order and catalogue value order."""
    return tuple((j, v) for j, e in enumerate(kb.evidences) if not e.is_binary for v in e.values if v != e.default)


@dataclass(frozen=True, eq=False)
class FrequencyCounts:
    """Record counts per disease in KB order.

    `by_age_sex[d, a, s]` counts records of d aged a years with sex SEXES[s], up to the oldest record, so that
    any age grouping is a sum over it (`age_groups`). `positive[d, j]` counts records of d with evidence j present
    (a non-default value for a categorical or multi-choice one); `value[d, c]` counts records of d holding
    `value_columns(kb)[c]`.
    """

    rows: IntArray
    by_age_sex: IntArray
    positive: IntArray
    value: IntArray

    def __post_init__(self) -> None:
        d = self.rows.shape[0]
        if self.by_age_sex.ndim != 3 or self.by_age_sex.shape[::2] != (d, len(SEXES)):
            raise ValueError(f"by_age_sex has shape {self.by_age_sex.shape}")
        if self.positive.shape[0] != d or self.value.shape[0] != d:
            raise ValueError("count tables disagree on the number of diseases")
        if not np.array_equal(self.by_age_sex.sum(axis=(1, 2)), self.rows):
            raise ValueError("age/sex cells do not add up to the disease rows")

    def age_groups(self, edges: Sequence[int] = AGE_BIN_EDGES) -> IntArray:
        """Records per disease, age group (`age_bin` with `edges`) and sex."""
        check_age_edges(edges)
        out = np.zeros((self.rows.shape[0], len(edges) + 1, len(SEXES)), dtype=np.int64)
        for age in range(self.by_age_sex.shape[1]):
            out[:, age_bin(age, edges)] += self.by_age_sex[:, age]
        return out


def count_frequencies(kb: KB, patients: Iterable[Patient]) -> FrequencyCounts:
    columns = {pair: c for c, pair in enumerate(value_columns(kb))}
    rows = np.zeros(kb.n_diseases, dtype=np.int64)
    by_age_sex = np.zeros((kb.n_diseases, 0, len(SEXES)), dtype=np.int64)
    positive = np.zeros((kb.n_diseases, kb.n_evidences), dtype=np.int64)
    value = np.zeros((kb.n_diseases, len(columns)), dtype=np.int64)
    for p in patients:
        d = kb.disease_index[p.pathology]
        if p.age < 0:
            raise ValueError(f"negative age {p.age}")
        if p.age >= by_age_sex.shape[1]:
            by_age_sex = np.pad(by_age_sex, ((0, 0), (0, p.age + 1 - by_age_sex.shape[1]), (0, 0)))
        rows[d] += 1
        by_age_sex[d, p.age, sex_index(p.sex)] += 1
        for code, values in present_values(kb, p.evidences).items():
            j = kb.evidence_index[code]
            positive[d, j] += 1
            for v in values:
                value[d, columns[(j, v)]] += 1
    return FrequencyCounts(rows=rows, by_age_sex=by_age_sex, positive=positive, value=value)
