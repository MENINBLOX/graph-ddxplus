"""Check C1 of the review (§5.6): the code-level KB is the generator support the v1 graph folded to CUIs."""

from __future__ import annotations

import pytest

from ddxkb.audit import support_audit
from ddxkb.data import REPO_ROOT, Patient
from ddxkb.kb import KB
from ddxkb.policies.v1_replay import V1_DATA_DIR, folded_cui_edges, load_cui_mappings, load_v1_edges
from ddxkb.units import cui_units


def _require_v1_files() -> None:
    for name in ("kg_edges.json", "umls_mapping.json", "disease_umls_mapping.json"):
        if not (REPO_ROOT / V1_DATA_DIR / name).exists():
            pytest.skip(f"{name} not present")


def test_kb_folded_to_cuis_reproduces_the_v1_graph(real_kb: KB) -> None:
    _require_v1_files()
    folded = folded_cui_edges(real_kb, *load_cui_mappings())
    v1 = set(load_v1_edges())
    assert (len(folded), len(v1), len(folded & v1)) == (586, 581, 580)
    assert v1 - folded == {("C0003811", "C3264374")}


def test_real_cui_question_spaces(real_kb: KB) -> None:
    # Review §2.1: 204 codes map to 188 CUIs; the 19 others (14 subattributes, 5 comorbidities) have none.
    _require_v1_files()
    code_cui, _ = load_cui_mappings()
    assert cui_units(real_kb, code_cui).kb.n_evidences == 188
    assert cui_units(real_kb, code_cui, keep_unmapped=True).kb.n_evidences == 188 + 19


def test_validation_support_audit(real_kb: KB, validation: list[Patient]) -> None:
    a = support_audit(real_kb, validation)
    assert a.n == 132_448
    assert a.outside_support == 0
    # Recorded codes whatever their value: the figure of reference/audit_struct.py (92.42%).
    assert a.unique_recorded == 122_408
    assert round(100 * a.unique_recorded_share, 2) == 92.42
    # Positive set under the oracle rule (default value = absent).
    assert a.unique == 122_321
    assert round(100 * a.unique_share, 2) == 92.35
