from __future__ import annotations

import numpy as np
import pytest

from ddxkb.kb import KB, DataType
from ddxkb.policies.bayes import BayesPolicy, membership_table, uniform_prior
from ddxkb.units import QuestionUnits, code_units, cui_units
from helpers import toy_patient

RECORD = ["douleur", "douleur_intens_@_0", "douleur_lieu_@_dos", "atcd", "trav1_@_N"]


def test_code_units_ask_the_same_questions_as_p1(toy_kb: KB) -> None:
    units = code_units(toy_kb)
    assert [e.code for e in units.kb.evidences] == [e.code for e in toy_kb.evidences]
    assert np.array_equal(units.kb.membership, toy_kb.membership)
    patient = toy_patient(RECORD, initial="douleur")
    direct = BayesPolicy(membership_table(toy_kb, 0.5, 1e-3), uniform_prior(toy_kb), None, (0, 3, 6)).run(patient)
    policy = BayesPolicy(membership_table(units.kb, 0.5, 1e-3), uniform_prior(units.kb), None, (0, 3, 6))
    via_units = policy.run(units.translate(patient))
    assert via_units.questions == direct.questions
    assert np.array_equal(via_units.budget_posteriors, direct.budget_posteriors)


def test_translation_keeps_only_positive_units(toy_kb: KB) -> None:
    units = code_units(toy_kb)
    p = units.translate(toy_patient(RECORD, initial="douleur", row=9))
    # douleur_intens_@_0 and trav1_@_N hold default values; douleur_lieu_@_dos does not.
    assert p.evidences == (("douleur", None), ("douleur_lieu", None), ("atcd", None))
    assert (p.row, p.initial, p.pathology) == (9, "douleur", "Beta")
    assert all(e.data_type is DataType.BINARY for e in units.kb.evidences)


def test_dropped_codes_cannot_be_asked(toy_kb: KB) -> None:
    units = code_units(toy_kb, drop=frozenset({"douleur_intens", "douleur_lieu"}))
    assert [e.code for e in units.kb.evidences] == ["fievre", "toux", "douleur", "trav1", "atcd"]
    p = units.translate(toy_patient(RECORD, initial="douleur"))
    assert p.evidences == (("douleur", None), ("atcd", None))
    with pytest.raises(ValueError, match="no question unit"):
        code_units(toy_kb, drop=frozenset({"douleur"})).translate(toy_patient(RECORD, initial="douleur"))


def test_cui_units_merge_codes_and_their_lists(toy_kb: KB) -> None:
    # fievre and toux share C1; atcd has no CUI.
    units = cui_units(toy_kb, {"fievre": "C1", "toux": "C1", "douleur": "C2", "douleur_lieu": "C3", "trav1": "C4"})
    assert [e.code for e in units.kb.evidences] == ["C1", "C2", "C3", "C4"]
    # Alpha lists fievre and toux, Beta neither, Gamma fievre.
    assert units.kb.membership[:, 0].tolist() == [True, False, True]
    assert units.kb.evidence("C1").question_en == "fievre? / toux?"
    p = units.translate(toy_patient(["toux", "trav1_@_Asie"], initial="toux", pathology="Alpha"))
    assert (p.initial, p.evidences) == ("C1", (("C1", None), ("C4", None)))
    assert units.translate(toy_patient(["atcd", "douleur"], initial="douleur")).evidences == (("C2", None),)


def test_cui_units_can_keep_codes_without_a_cui(toy_kb: KB) -> None:
    code_cui = {"fievre": "C1", "toux": "C1", "douleur": "C2", "douleur_lieu": "C3", "trav1": "C4"}
    units = cui_units(toy_kb, code_cui, keep_unmapped=True)
    # Only the merge of fievre and toux differs from the code units: douleur_intens and atcd stay askable.
    assert [e.code for e in units.kb.evidences] == ["C1", "C2", "douleur_intens", "C3", "C4", "atcd"]
    assert np.array_equal(units.kb.membership[:, [2, 5]], toy_kb.membership[:, [3, 6]])
    p = units.translate(toy_patient(["atcd", "douleur", "douleur_intens_@_2"], initial="atcd"))
    assert (p.initial, p.evidences) == ("atcd", (("C2", None), ("douleur_intens", None), ("atcd", None)))
    with pytest.raises(ValueError, match="share a name"):
        cui_units(toy_kb, {"fievre": "atcd"}, keep_unmapped=True)


def test_units_validate_their_codes(toy_kb: KB) -> None:
    with pytest.raises(ValueError, match="not in the KB"):
        QuestionUnits(toy_kb, {"nope": "u"})
    with pytest.raises(ValueError, match="no askable unit"):
        QuestionUnits(toy_kb, {})
