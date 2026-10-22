"""The DDXPlus membership knowledge base: evidence catalogue and the disease x evidence support matrix."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


class DataType(StrEnum):
    BINARY = "B"
    CATEGORICAL = "C"
    MULTI_CHOICE = "M"


@dataclass(frozen=True, slots=True)
class Evidence:
    code: str
    data_type: DataType
    # str() of the JSON default_value, because patient records carry values as text ("0", "N", "nulle_part").
    default: str
    values: tuple[str, ...]
    is_antecedent: bool
    parent: str
    question_en: str

    @property
    def is_binary(self) -> bool:
        return self.data_type is DataType.BINARY


@dataclass(frozen=True)
class KB:
    """Diseases and evidences keep the key order of the release JSON files; that order is the tie-break key."""

    diseases: tuple[str, ...]
    disease_names_en: tuple[str, ...]
    severity: tuple[int, ...]
    evidences: tuple[Evidence, ...]
    membership: BoolArray
    disease_index: Mapping[str, int] = field(init=False)
    evidence_index: Mapping[str, int] = field(init=False)

    def __post_init__(self) -> None:
        if self.membership.shape != (len(self.diseases), len(self.evidences)):
            raise ValueError(f"membership shape {self.membership.shape} does not match the catalogue")
        self.membership.flags.writeable = False
        object.__setattr__(self, "disease_index", {d: i for i, d in enumerate(self.diseases)})
        object.__setattr__(self, "evidence_index", {e.code: j for j, e in enumerate(self.evidences)})

    @classmethod
    def from_json(cls, conditions: Mapping[str, Any], evidences: Mapping[str, Any]) -> KB:
        """Build from the parsed release_conditions.json and release_evidences.json (French-code release)."""
        catalogue = tuple(
            Evidence(
                code=code,
                data_type=DataType(e["data_type"]),
                default=str(e["default_value"]),
                values=tuple(str(v) for v in e["possible-values"]),
                is_antecedent=bool(e["is_antecedent"]),
                parent=str(e["code_question"]),
                question_en=str(e["question_en"]),
            )
            for code, e in evidences.items()
        )
        col = {e.code: j for j, e in enumerate(catalogue)}
        membership = np.zeros((len(conditions), len(catalogue)), dtype=np.bool_)
        for i, c in enumerate(conditions.values()):
            for code in (*c["symptoms"], *c["antecedents"]):
                if code not in col:
                    raise ValueError(f"{c['condition_name']}: evidence {code!r} is not in the evidence catalogue")
                membership[i, col[code]] = True
        return cls(
            diseases=tuple(str(c["condition_name"]) for c in conditions.values()),
            disease_names_en=tuple(str(c["cond-name-eng"]) for c in conditions.values()),
            severity=tuple(int(c["severity"]) for c in conditions.values()),
            evidences=catalogue,
            membership=membership,
        )

    @property
    def n_diseases(self) -> int:
        return len(self.diseases)

    @property
    def n_evidences(self) -> int:
        return len(self.evidences)

    def evidence(self, code: str) -> Evidence:
        return self.evidences[self.evidence_index[code]]

    def support(self, disease: str) -> frozenset[str]:
        row = self.membership[self.disease_index[disease]]
        return frozenset(self.evidences[j].code for j in np.flatnonzero(row))

    def evidence_mask(self, codes: Iterable[str]) -> BoolArray:
        mask = np.zeros(self.n_evidences, dtype=np.bool_)
        mask[[self.evidence_index[c] for c in codes]] = True
        return mask

    def diseases_containing(self, codes: Iterable[str]) -> BoolArray:
        """Diseases whose support contains every given code."""
        mask = self.evidence_mask(codes)
        return np.asarray(~(mask & ~self.membership).any(axis=1))

    def disease_vector(self, pairs: Iterable[tuple[str, float]]) -> FloatArray:
        """A per-disease probability vector in KB order from (disease, probability) pairs."""
        v = np.zeros(self.n_diseases, dtype=np.float64)
        for name, p in pairs:
            v[self.disease_index[name]] += p
        return v
