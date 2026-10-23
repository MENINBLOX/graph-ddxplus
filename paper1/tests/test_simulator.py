from __future__ import annotations

import numpy as np
import pytest

from ddxkb.data import Patient
from ddxkb.kb import KB
from ddxkb.simulator import (
    MAX_IL,
    Answer,
    Hierarchy,
    Interaction,
    Oracle,
    Start,
    StartInput,
    positive_codes,
    present_values,
)
from helpers import toy_patient

RECORD = [
    "douleur",
    "douleur_intens_@_2",
    "douleur_lieu_@_ventre",
    "douleur_lieu_@_tete",
    "atcd",
    "trav1_@_N",
]


def test_default_values_are_negative(toy_kb: KB) -> None:
    p = toy_patient(RECORD, initial="douleur")
    assert positive_codes(toy_kb, p) == {"douleur", "douleur_intens", "douleur_lieu", "atcd"}
    assert Oracle(toy_kb, p).answer("trav1") == Answer("trav1", positive=False)


def test_travel_with_a_destination_is_positive(toy_kb: KB) -> None:
    p = toy_patient(["fievre", "trav1_@_Asie"], initial="fievre", pathology="Alpha")
    assert Oracle(toy_kb, p).answer("trav1") == Answer("trav1", True, ("Asie",))


def test_answers_reveal_one_categorical_value_and_every_multi_choice_value(toy_kb: KB) -> None:
    o = Oracle(toy_kb, toy_patient(RECORD, initial="douleur"))
    assert o.answer("douleur_lieu") == Answer("douleur_lieu", True, ("tete", "ventre"))
    assert o.answer("douleur_intens") == Answer("douleur_intens", True, ("2",))
    assert o.answer("douleur") == Answer("douleur", True, ())
    assert o.answer("toux") == Answer("toux", False, ())


def test_multi_choice_with_only_the_default_is_negative(toy_kb: KB) -> None:
    p = toy_patient(["douleur", "douleur_lieu_@_nulle_part", "douleur_intens_@_0"], initial="douleur")
    o = Oracle(toy_kb, p)
    assert o.answer("douleur_lieu") == Answer("douleur_lieu", positive=False)
    assert o.answer("douleur_intens") == Answer("douleur_intens", positive=False)
    assert positive_codes(toy_kb, p) == {"douleur"}


def test_multi_choice_drops_a_default_listed_with_other_values(toy_kb: KB) -> None:
    pairs = [("douleur_lieu", "nulle_part"), ("douleur_lieu", "dos")]
    assert present_values(toy_kb, pairs) == {"douleur_lieu": ("dos",)}


@pytest.mark.parametrize(
    "record",
    [["douleur_lieu_@_genou"], ["douleur_lieu"], ["fievre_@_1"], ["douleur_intens_@_1", "douleur_intens_@_2"]],
)
def test_malformed_records_are_rejected(toy_kb: KB, record: list[str]) -> None:
    with pytest.raises(ValueError):
        positive_codes(toy_kb, toy_patient(record, initial="douleur"))


def test_interaction_counts_the_initial_evidence(toy_kb: KB) -> None:
    it = Interaction(Oracle(toy_kb, toy_patient(RECORD, initial="douleur")), max_il=4)
    assert it.il == 1
    assert it.initial == Answer("douleur", positive=True)
    assert it.ask("fievre") == (Answer("fievre", positive=False),)
    assert it.ask("atcd") == (Answer("atcd", positive=True),)
    assert (it.il, it.remaining) == (3, 1)
    assert [a.code for a in it.answers] == ["douleur", "fievre", "atcd"]
    assert it.implied == []


def test_the_official_cap_is_thirty_questions_after_the_initial_evidence() -> None:
    assert MAX_IL == 31


def test_start_gives_age_and_sex_only_when_declared(toy_kb: KB) -> None:
    oracle = Oracle(toy_kb, toy_patient(RECORD, initial="douleur"))
    assert Interaction(oracle).start == Start(Answer("douleur", True), None, None)
    full = Interaction(oracle, inputs=frozenset(StartInput))
    assert full.start == Start(Answer("douleur", True), 40, "F")
    with pytest.raises(ValueError, match="initial evidence"):
        Interaction(oracle, inputs=frozenset({StartInput.AGE, StartInput.SEX}))


def _pain(kb: KB, record: list[str], hierarchy: Hierarchy, initial: str = "fievre") -> Interaction:
    """An interaction on the toy KB, where douleur_intens and douleur_lieu are subquestions of douleur."""
    return Interaction(Oracle(kb, toy_patient(record, initial=initial)), max_il=6, hierarchy=hierarchy)


def test_without_propagation_a_child_may_come_first_and_implies_nothing(toy_kb: KB) -> None:
    it = _pain(toy_kb, ["fievre", "douleur", "douleur_lieu_@_dos"], Hierarchy.NONE)
    assert it.ask("douleur_lieu") == (Answer("douleur_lieu", True, ("dos",)),)
    assert not it.closed()[toy_kb.evidence_index["douleur"]]
    assert it.ask("douleur") == (Answer("douleur", True),)


def test_implicit_answers_fill_children_of_a_negative_parent(toy_kb: KB) -> None:
    it = _pain(toy_kb, ["fievre"], Hierarchy.IMPLICIT)
    answers = it.ask("douleur")
    assert answers == (Answer("douleur", False), Answer("douleur_intens", False), Answer("douleur_lieu", False))
    # One turn for three answers; the children are closed and refused.
    assert (it.il, [a.code for a in it.implied]) == (2, ["douleur_intens", "douleur_lieu"])
    assert it.closed()[[toy_kb.evidence_index[c] for c in ("douleur_intens", "douleur_lieu")]].all()
    with pytest.raises(ValueError, match="already asked or implied"):
        it.ask("douleur_lieu")


def test_implicit_answers_fill_the_parent_of_a_positive_child(toy_kb: KB) -> None:
    it = _pain(toy_kb, ["fievre", "douleur", "douleur_intens_@_2"], Hierarchy.IMPLICIT)
    assert it.ask("douleur_intens") == (Answer("douleur_intens", True, ("2",)), Answer("douleur", True))
    assert it.positive[toy_kb.evidence_index["douleur"]] and it.il == 2
    # A negative child implies nothing, and an already known answer is not implied again.
    assert it.ask("douleur_lieu") == (Answer("douleur_lieu", False),)


def test_implicit_answers_reject_a_record_that_breaks_the_hierarchy(toy_kb: KB) -> None:
    it = _pain(toy_kb, ["fievre", "douleur_lieu_@_dos"], Hierarchy.IMPLICIT)
    with pytest.raises(ValueError, match="contradicts"):
        it.ask("douleur")


def test_parent_first_closes_children_until_the_parent_is_positive(toy_kb: KB) -> None:
    lieu = toy_kb.evidence_index["douleur_lieu"]
    it = _pain(toy_kb, ["fievre", "douleur", "douleur_lieu_@_dos"], Hierarchy.PARENT_FIRST)
    assert it.closed()[lieu]
    with pytest.raises(ValueError, match="before its parent douleur"):
        it.ask("douleur_lieu")
    assert it.ask("douleur") == (Answer("douleur", True),)
    assert not it.closed()[lieu]
    assert it.ask("douleur_lieu") == (Answer("douleur_lieu", True, ("dos",)),)
    negative = _pain(toy_kb, ["fievre"], Hierarchy.PARENT_FIRST)
    negative.ask("douleur")
    assert negative.closed()[lieu] and negative.implied == []
    started = _pain(toy_kb, ["douleur", "douleur_lieu_@_dos"], Hierarchy.PARENT_FIRST, initial="douleur")
    assert not started.closed()[lieu]


def test_no_children_never_opens_a_subquestion(toy_kb: KB) -> None:
    children = [toy_kb.evidence_index[c] for c in ("douleur_intens", "douleur_lieu")]
    it = _pain(toy_kb, ["fievre", "douleur", "douleur_lieu_@_dos"], Hierarchy.NO_CHILDREN)
    assert it.closed()[children].all()
    assert it.ask("douleur") == (Answer("douleur", True),)
    assert it.closed()[children].all() and it.implied == []
    with pytest.raises(ValueError, match="subquestions are never asked"):
        it.ask("douleur_lieu")


def test_closed_and_positive_masks_are_read_only(toy_kb: KB) -> None:
    it = _pain(toy_kb, ["fievre"], Hierarchy.NONE)
    with pytest.raises(ValueError):
        it.closed()[0] = False
    with pytest.raises(ValueError):
        it.positive[0] = False
    assert np.flatnonzero(it.closed()).tolist() == [toy_kb.evidence_index["fievre"]]


def test_interaction_refuses_repeats_and_the_budget(toy_kb: KB) -> None:
    it = Interaction(Oracle(toy_kb, toy_patient(RECORD, initial="douleur")), max_il=2)
    with pytest.raises(ValueError, match="already asked"):
        it.ask("douleur")
    it.ask("trav1")
    with pytest.raises(ValueError, match="already asked"):
        it.ask("trav1")
    with pytest.raises(RuntimeError, match="limit"):
        it.ask("toux")
    with pytest.raises(KeyError):
        Interaction(Oracle(toy_kb, toy_patient(RECORD, initial="douleur"))).ask("unknown")


def test_interaction_rejects_a_bad_initial_evidence(toy_kb: KB) -> None:
    with pytest.raises(ValueError, match="not positive"):
        Interaction(Oracle(toy_kb, toy_patient(RECORD, initial="fievre")))
    with pytest.raises(ValueError, match="not binary"):
        Interaction(Oracle(toy_kb, toy_patient(RECORD, initial="douleur_lieu")))


def test_real_hierarchy_holds_on_validation(real_kb: KB, validation: list[Patient]) -> None:
    # No record holds a positive child without its parent, so implied answers are the recorded ones.
    child = real_kb.child_mask
    for p in validation:
        pos = np.zeros(real_kb.n_evidences, dtype=np.bool_)
        pos[[real_kb.evidence_index[c] for c in positive_codes(real_kb, p)]] = True
        assert not (pos & child & ~pos[np.maximum(real_kb.parent_index, 0)]).any(), p.row


def test_real_trav1_default_is_negative(real_kb: KB, validation: list[Patient]) -> None:
    n_recorded = sum(any(c == "trav1" for c, _ in p.evidences) for p in validation)
    n_default = sum(("trav1", "N") in p.evidences for p in validation)
    n_positive = sum(Oracle(real_kb, p).answer("trav1").positive for p in validation)
    assert (n_recorded, n_default, n_positive) == (132_448, 122_066, 10_382)


def test_real_initial_evidence_is_a_positive_binary_evidence(real_kb: KB, validation: list[Patient]) -> None:
    for p in validation:
        assert Interaction(Oracle(real_kb, p)).initial.positive
