"""Question units for ablation A4 (design §3): one yes/no question standing for one or more evidence codes."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ddxkb.data import Patient
from ddxkb.kb import KB, DataType, Evidence
from ddxkb.simulator import positive_codes


@dataclass(frozen=True, eq=False)
class QuestionUnits:
    """A binary question space over the evidences of `source`.

    `unit_of` maps evidence codes to unit names. A unit is answered yes when any of its codes is positive
    under the main protocol's rule (a categorical or multi-choice code only with a non-default value). Codes
    outside `unit_of` cannot be asked and their answers are never heard. In `kb` the units are binary
    evidences in the source order of their first code, and a disease lists a unit when it lists any of its codes.
    """

    source: KB
    unit_of: Mapping[str, str]
    kb: KB = field(init=False)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.unit_of) - set(self.source.evidence_index))
        if unknown:
            raise ValueError(f"codes not in the KB: {unknown}")
        codes_of: dict[str, list[str]] = {}
        for e in self.source.evidences:
            if e.code in self.unit_of:
                codes_of.setdefault(self.unit_of[e.code], []).append(e.code)
        if not codes_of:
            raise ValueError("no askable unit")
        columns = [[self.source.evidence_index[c] for c in codes] for codes in codes_of.values()]
        membership = np.stack([self.source.membership[:, cols].any(axis=1) for cols in columns], axis=1)
        evidences = tuple(
            Evidence(
                code=unit,
                data_type=DataType.BINARY,
                default="0",
                values=(),
                is_antecedent=all(self.source.evidence(c).is_antecedent for c in codes),
                parent=unit,
                question_en=" / ".join(self.source.evidence(c).question_en for c in codes),
            )
            for unit, codes in codes_of.items()
        )
        kb = KB(self.source.diseases, self.source.disease_names_en, self.source.severity, evidences, membership)
        object.__setattr__(self, "kb", kb)

    def translate(self, patient: Patient) -> Patient:
        """The patient as a record over the units: its positive units, binary, in unit order."""
        if patient.initial not in self.unit_of:
            raise ValueError(f"initial evidence {patient.initial} has no question unit")
        yes = {self.unit_of[c] for c in positive_codes(self.source, patient) if c in self.unit_of}
        return dataclasses.replace(
            patient,
            initial=self.unit_of[patient.initial],
            evidences=tuple((e.code, None) for e in self.kb.evidences if e.code in yes),
        )


def code_units(kb: KB, drop: frozenset[str] = frozenset()) -> QuestionUnits:
    """One unit per evidence code, except the dropped codes."""
    return QuestionUnits(kb, {e.code: e.code for e in kb.evidences if e.code not in drop})


def cui_units(kb: KB, code_cui: Mapping[str, str], keep_unmapped: bool = False) -> QuestionUnits:
    """Codes sharing a CUI in `code_cui` form one unit.

    A code without a CUI cannot be asked, as in v1's question space, or with `keep_unmapped` is a unit of its
    own, so that only the merging differs from the code units.
    """
    unit_of = {e.code: code_cui[e.code] for e in kb.evidences if e.code in code_cui}
    if keep_unmapped:
        unmapped = [e.code for e in kb.evidences if e.code not in code_cui]
        clashes = sorted(set(unmapped) & set(unit_of.values()))
        if clashes:
            raise ValueError(f"codes without a CUI share a name with a CUI: {clashes}")
        unit_of.update({code: code for code in unmapped})
    return QuestionUnits(kb, unit_of)
