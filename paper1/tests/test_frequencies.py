from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ddxkb.frequencies import N_AGE_BINS, FrequencyCounts, age_bin, count_frequencies, sex_index, value_columns
from ddxkb.kb import KB
from ddxkb.protocol import FIRST_AGE_BIN_EDGES
from helpers import toy_patient


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        *[(0, 0), (1, 1), (4, 1), (5, 2), (14, 2), (15, 3), (29, 3), (30, 4), (44, 4)],
        *[(45, 5), (59, 5), (60, 6), (74, 6), (75, 7), (109, 7)],
    ],
)
def test_age_bins_are_the_official_eight(age: int, expected: int) -> None:
    # [0,1), 1-4, 5-14, 15-29, 30-44, 45-59, 60-74, 75+ (encode_age of the DDXPlus and CASANDE code).
    assert age_bin(age) == expected
    assert N_AGE_BINS == 8


def test_the_first_protocols_age_groups() -> None:
    # Edges (2, 5, 8, 10, 18, 30, 33, 45, 60, 75): eleven groups, the first [0, 2).
    assert [age_bin(a, FIRST_AGE_BIN_EDGES) for a in (0, 1, 2, 9, 10, 17, 18, 32, 33, 74, 75, 109)] == [
        0,
        0,
        1,
        3,
        4,
        4,
        5,
        6,
        7,
        9,
        10,
        10,
    ]


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
    # By exact age, up to the oldest record.
    assert c.by_age_sex.shape == (3, 41, 2)
    assert c.by_age_sex[1, 40, sex_index("F")] == 1
    assert c.by_age_sex[1, 1, sex_index("M")] == 1
    assert c.by_age_sex.sum() == 3
    groups = c.age_groups()
    assert groups.shape == (3, N_AGE_BINS, 2)
    assert (groups[1, age_bin(40), sex_index("F")], groups[1, age_bin(1), sex_index("M")]) == (1, 1)
    first = c.age_groups(FIRST_AGE_BIN_EDGES)
    assert first.shape == (3, len(FIRST_AGE_BIN_EDGES) + 1, 2)
    assert (first[1, 7, sex_index("F")], first[1, 0, sex_index("M")], int(first.sum())) == (1, 1, 3)
    with pytest.raises(ValueError, match="edges"):
        c.age_groups((5, 5))
    # Columns: fievre, toux, douleur, douleur_intens, douleur_lieu, trav1, atcd.
    assert c.positive.tolist() == [[1, 1, 0, 0, 0, 0, 0], [0, 0, 2, 1, 1, 1, 1], [0] * 7]
    # Columns: intens 1/2/3, lieu tete/dos/ventre, trav1 AfriqN/Asie.
    assert c.value.tolist() == [[0] * 8, [0, 1, 0, 1, 1, 0, 0, 1], [0] * 8]


def test_counts_reject_inconsistent_tables() -> None:
    ok = np.zeros((2, 90, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="add up"):
        FrequencyCounts(
            rows=np.array([1, 0]), by_age_sex=ok, positive=np.zeros((2, 3), np.int64), value=np.zeros((2, 1), np.int64)
        )
    with pytest.raises(ValueError, match="shape"):
        FrequencyCounts(
            rows=np.array([0, 0]),
            by_age_sex=np.zeros((2, 3, 3), np.int64),
            positive=np.zeros((2, 3), np.int64),
            value=np.zeros((2, 1), np.int64),
        )
