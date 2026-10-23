"""Dataset properties reported for validation and each evaluation set (design §Goals): KB violations and
set-inclusion uniqueness."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ddxkb.data import Patient
from ddxkb.kb import KB, BoolArray
from ddxkb.simulator import positive_codes


@dataclass(frozen=True, slots=True)
class SupportAudit:
    """`unique` uses the positive set (default values negative, as the oracle answers). `unique_recorded`
    uses every recorded code whatever its value, so a record such as douleurxx_irrad_@_nulle_part still
    rules out diseases without douleurxx_irrad; the review prototype (reference/audit_struct.py) counts this way."""

    n: int
    outside_support: int
    unique: int
    unique_recorded: int

    @property
    def unique_share(self) -> float:
        return self.unique / self.n

    @property
    def unique_recorded_share(self) -> float:
        return self.unique_recorded / self.n

    def summary(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "outside_support": self.outside_support,
            "unique": {"count": self.unique, "share": self.unique_share},
            "unique_recorded": {"count": self.unique_recorded, "share": self.unique_recorded_share},
            "definitions": {
                "outside_support": "patients with a positive evidence off their disease's list (default values absent)",
                "unique": "the gold disease is the only one whose list holds every positive evidence (main figure)",
                "unique_recorded": "the same over every recorded code whatever its value",
            },
        }


def support_audit(kb: KB, patients: Iterable[Patient]) -> SupportAudit:
    n = outside = unique = unique_recorded = 0
    for p in patients:
        gold = kb.disease_index[p.pathology]
        positives = positive_codes(kb, p)
        n += 1
        outside += not positives <= kb.support(p.pathology)
        unique += _only(kb.diseases_containing(positives), gold)
        unique_recorded += _only(kb.diseases_containing(code for code, _ in p.evidences), gold)
    return SupportAudit(n=n, outside_support=outside, unique=unique, unique_recorded=unique_recorded)


def _only(mask: BoolArray, gold: int) -> bool:
    return bool(mask[gold]) and int(np.count_nonzero(mask)) == 1
