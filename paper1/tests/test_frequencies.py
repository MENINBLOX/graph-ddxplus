from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ddxkb.frequencies import N_AGE_BINS, FrequencyCounts, age_bin, count_frequencies, sex_index, value_columns
from ddxkb.kb import KB
from helpers import toy_patient


@pytest.mark.parametrize(
    ("age", "expected"),
    [(0, 0), (1, 0), (2, 1), (4, 1), (9, 3), (10, 4), (17, 4), (18, 5), (32, 6), (33, 7), (74, 9), (75, 10), (109, 10)],
)
def test_age_bins(age: int, expected: int) -> None:
    assert age_bin(age) == expected
    assert N_AGE_BINS == 11


def test_bad_age_or_sex_is_rejected() -> None:
    with pytest.raises(ValueError):
        age_bin(-1)
    with pytest.raises(ValueError):
        sex_index("X")
    assert (sex_index("F"), sex_index("M")) == (0, 1)


def test_value_columns_skip_defaults_and_binaries(toy_kb: KB) -> None:
    assert value_columns(toy_kb) == (
        (3, "1"),
        (3, "2"),
        (3, "3"),
        (4, "tete"),
        (4, "dos"),
        (4, "ventre"),
        (5, "AfriqN"),
        (5, "Asie"),
    )


def test_counts_follow_the_oracle_rule(toy_kb: KB) -> None:
    beta = toy_patient(
        ["douleur", "douleur_intens_@_2", "douleur_lieu_@_tete", "douleur_lieu_@_dos", "trav1_@_N", "atcd"],
        initial="douleur",
    )
    beta_child = dataclasses.replace(
        toy_patient(["douleur", "douleur_intens_@_0", "douleur_lieu_@_nulle_part", "trav1_@_Asie"], initial="douleur"),
        age=1,
        sex="M",
    )
    alpha = toy_patient(["fievre", "toux", "trav1_@_N"], initial="fievre", pathology="Alpha")
    c = count_frequencies(toy_kb, [beta, beta_child, alpha])

    assert c.rows.tolist() == [1, 2, 0]
    assert c.by_age_sex[1, age_bin(40), sex_index("F")] == 1
    assert c.by_age_sex[1, 0, sex_index("M")] == 1
    assert c.by_age_sex.sum() == 3
    # Columns: fievre, toux, douleur, douleur_intens, douleur_lieu, trav1, atcd.
    assert c.positive.tolist() == [[1, 1, 0, 0, 0, 0, 0], [0, 0, 2, 1, 1, 1, 1], [0] * 7]
    # Columns: intens 1/2/3, lieu tete/dos/ventre, trav1 AfriqN/Asie.
    assert c.value.tolist() == [[0] * 8, [0, 1, 0, 1, 1, 0, 0, 1], [0] * 8]


def test_counts_reject_inconsistent_tables() -> None:
    ok = np.zeros((2, N_AGE_BINS, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="add up"):
        FrequencyCounts(
            rows=np.array([1, 0]), by_age_sex=ok, positive=np.zeros((2, 3), np.int64), value=np.zeros((2, 1), np.int64)
        )
    with pytest.raises(ValueError, match="shape"):
        FrequencyCounts(
            rows=np.array([0, 0]),
            by_age_sex=np.zeros((2, 3, 2), np.int64),
            positive=np.zeros((2, 3), np.int64),
            value=np.zeros((2, 1), np.int64),
        )
