"""Train frequencies for the frequency-Bayes policy P2 (design §3), counted from train minus H2.

A record value equal to the evidence's default counts as absent, as the oracle answers.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ddxkb.data import Patient
from ddxkb.kb import KB
from ddxkb.simulator import present_values

IntArray = npt.NDArray[np.int64]

# Edges sit at every age where some disease starts or stops occurring in train minus H2 (bronchiolitis only
# below 2, croup below 10; first ages 2, 5, 8, 10, 18, 30 and 33), so no bin mixes ages where a disease
# cannot occur with ages where it does. 45, 60 and 75 split the adult range, where incidence only drifts;
# the smallest (bin, sex) cell still holds over 8,000 train rows.
AGE_BIN_EDGES = (2, 5, 8, 10, 18, 30, 33, 45, 60, 75)
N_AGE_BINS = len(AGE_BIN_EDGES) + 1
SEXES = ("F", "M")


def age_bin(age: int) -> int:
    if age < 0:
        raise ValueError(f"negative age {age}")
    return bisect.bisect_right(AGE_BIN_EDGES, age)


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

    `positive[d, j]` counts records of d with evidence j present (a non-default value for a categorical or
    multi-choice one); `value[d, c]` counts records of d holding `value_columns(kb)[c]`.
    """

    rows: IntArray
    by_age_sex: IntArray
    positive: IntArray
    value: IntArray

    def __post_init__(self) -> None:
        d = self.rows.shape[0]
        if self.by_age_sex.shape != (d, N_AGE_BINS, len(SEXES)):
            raise ValueError(f"by_age_sex has shape {self.by_age_sex.shape}")
        if self.positive.shape[0] != d or self.value.shape[0] != d:
            raise ValueError("count tables disagree on the number of diseases")
        if not np.array_equal(self.by_age_sex.sum(axis=(1, 2)), self.rows):
            raise ValueError("age/sex cells do not add up to the disease rows")


def count_frequencies(kb: KB, patients: Iterable[Patient]) -> FrequencyCounts:
    columns = {pair: c for c, pair in enumerate(value_columns(kb))}
    rows = np.zeros(kb.n_diseases, dtype=np.int64)
    by_age_sex = np.zeros((kb.n_diseases, N_AGE_BINS, len(SEXES)), dtype=np.int64)
    positive = np.zeros((kb.n_diseases, kb.n_evidences), dtype=np.int64)
    value = np.zeros((kb.n_diseases, len(columns)), dtype=np.int64)
    for p in patients:
        d = kb.disease_index[p.pathology]
        rows[d] += 1
        by_age_sex[d, age_bin(p.age), sex_index(p.sex)] += 1
        for code, values in present_values(kb, p.evidences).items():
            j = kb.evidence_index[code]
            positive[d, j] += 1
            for v in values:
                value[d, columns[(j, v)]] += 1
    return FrequencyCounts(rows=rows, by_age_sex=by_age_sex, positive=positive, value=value)
