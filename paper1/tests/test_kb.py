from __future__ import annotations

import collections

import numpy as np
import pytest

from ddxkb.kb import KB, DataType
from helpers import TOY_CONDITIONS, TOY_EVIDENCES


def test_catalogue_keeps_file_order_types_and_defaults(toy_kb: KB) -> None:
    assert toy_kb.diseases == ("Alpha", "Beta", "Gamma")
    assert [e.code for e in toy_kb.evidences] == list(TOY_EVIDENCES)
    assert toy_kb.evidence("fievre").data_type is DataType.BINARY
    assert toy_kb.evidence("douleur_intens").data_type is DataType.CATEGORICAL
    assert toy_kb.evidence("douleur_lieu").data_type is DataType.MULTI_CHOICE
    assert toy_kb.evidence("douleur_intens").default == "0"
    assert toy_kb.evidence("douleur_intens").values == ("0", "1", "2", "3")
    assert toy_kb.evidence("douleur_lieu").default == "nulle_part"
    assert toy_kb.evidence("trav1").default == "N"
    assert toy_kb.evidence("trav1").is_antecedent
    assert not toy_kb.evidence("toux").is_antecedent
    assert toy_kb.evidence("douleur_lieu").parent == "douleur"


def test_membership_is_symptoms_plus_antecedents(toy_kb: KB) -> None:
    expected = np.array(
        [
            [1, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 0],
        ],
        dtype=bool,
    )
    assert np.array_equal(toy_kb.membership, expected)
    assert toy_kb.support("Beta") == {"douleur", "douleur_intens", "douleur_lieu", "atcd", "trav1"}
    assert toy_kb.support("Alpha") == {"fievre", "toux", "trav1"}
    with pytest.raises(ValueError):
        toy_kb.membership[0, 0] = False


def test_diseases_containing(toy_kb: KB) -> None:
    assert toy_kb.diseases_containing(["fievre"]).tolist() == [True, False, True]
    assert toy_kb.diseases_containing(["fievre", "douleur"]).tolist() == [False, False, True]
    assert toy_kb.diseases_containing([]).tolist() == [True, True, True]
    assert toy_kb.diseases_containing(["toux", "atcd"]).tolist() == [False, False, False]


def test_disease_vector(toy_kb: KB) -> None:
    assert toy_kb.disease_vector([("Gamma", 0.7), ("Alpha", 0.3)]).tolist() == [0.3, 0.0, 0.7]


def test_unknown_evidence_in_a_condition_is_rejected() -> None:
    bad = dict(TOY_CONDITIONS)
    bad["Alpha"] = {**TOY_CONDITIONS["Alpha"], "symptoms": {"nope": {}}}
    with pytest.raises(ValueError, match="nope"):
        KB.from_json(bad, TOY_EVIDENCES)


def test_real_kb_shape(real_kb: KB) -> None:
    assert (real_kb.n_diseases, real_kb.n_evidences) == (49, 223)
    assert collections.Counter(e.data_type.value for e in real_kb.evidences) == {"B": 208, "C": 10, "M": 5}
    assert real_kb.evidence("trav1").default == "N"
    assert real_kb.evidence("trav1").is_antecedent
    assert real_kb.membership.any(axis=1).all()
    assert all(real_kb.membership[:, real_kb.evidence_index["trav1"]])
