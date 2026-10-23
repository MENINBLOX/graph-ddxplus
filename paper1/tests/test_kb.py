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


def test_subquestions_link_to_their_parent(toy_kb: KB) -> None:
    assert toy_kb.parent_index.tolist() == [-1, -1, -1, 2, 2, -1, -1]
    assert (toy_kb.parent_of("douleur_lieu"), toy_kb.parent_of("douleur")) == ("douleur", None)
    assert toy_kb.children_of("douleur") == ("douleur_intens", "douleur_lieu")
    assert toy_kb.children_of("fievre") == ()
    assert toy_kb.linked_mask.tolist() == [False, False, True, True, True, False, False]
    assert toy_kb.child_mask.tolist() == [False, False, False, True, True, False, False]
    with pytest.raises(ValueError):
        toy_kb.parent_index[0] = 1


def test_a_code_question_naming_no_evidence_links_nothing() -> None:
    evidences = {**TOY_EVIDENCES, "atcd": {**TOY_EVIDENCES["atcd"], "code_question": "atcd, autre"}}
    assert KB.from_json(TOY_CONDITIONS, evidences).parent_of("atcd") is None


def test_nested_subquestions_are_rejected() -> None:
    evidences = {**TOY_EVIDENCES, "douleur": {**TOY_EVIDENCES["douleur"], "code_question": "toux"}}
    with pytest.raises(ValueError, match="subquestions of subquestions"):
        KB.from_json(TOY_CONDITIONS, evidences)


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


def test_real_subquestions(real_kb: KB) -> None:
    # Review §2.2: three binary parents with 14 non-binary children; the five code_question typos link nothing.
    children = {real_kb.evidences[j].code for j in np.flatnonzero(real_kb.child_mask)}
    assert len(children) == 14 and all(not real_kb.evidence(c).is_binary for c in children)
    parents = {real_kb.parent_of(c) for c in children}
    assert parents == {"douleurxx", "lesions_peau", "oedeme"}
    assert [len(real_kb.children_of(p)) for p in ("douleurxx", "lesions_peau", "oedeme")] == [6, 7, 1]
    assert all(real_kb.evidence(p).is_binary for p in parents if p is not None)
    assert real_kb.parent_of("e10_e11") is None and real_kb.evidence("e10_e11").parent == "e10, e11"


def test_official_order_sorts_the_condition_keys() -> None:
    conditions = {name: dict(TOY_CONDITIONS[name]) for name in ("Gamma", "Alpha", "Beta")}
    kb = KB.from_json(conditions, TOY_EVIDENCES)
    assert kb.diseases == ("Gamma", "Alpha", "Beta")
    assert kb.official_order.tolist() == [1, 2, 0]
    assert [kb.diseases[i] for i in kb.official_order] == sorted(conditions)
    with pytest.raises(ValueError):
        kb.official_order[0] = 0


def test_condition_keys_must_be_their_names() -> None:
    conditions = {"alpha": TOY_CONDITIONS["Alpha"], "Beta": TOY_CONDITIONS["Beta"]}
    with pytest.raises(ValueError, match="condition_name"):
        KB.from_json(conditions, TOY_EVIDENCES)


def test_real_official_order_is_not_the_file_order(real_kb: KB) -> None:
    order = real_kb.official_order.tolist()
    assert [real_kb.diseases[i] for i in order] == sorted(real_kb.diseases)
    assert order != list(range(real_kb.n_diseases))
