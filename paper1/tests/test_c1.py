"""Check C1 of the review (§5.6): the code-level KB is the generator support."""

from __future__ import annotations

from ddxkb.audit import support_audit
from ddxkb.data import Patient
from ddxkb.kb import KB


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
