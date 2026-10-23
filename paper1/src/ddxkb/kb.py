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
IndexArray = npt.NDArray[np.intp]


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
    """Diseases and evidences keep the key order of the release JSON files; that order is the tie-break key.

    `official_order` lists the disease indices in the order the official code indexes diseases: the
    release_conditions.json keys sorted as Python strings (DDXPlus `basd/sim_utils.py` and CASANDE
    `chloe/utils/sim_utils.py`, `load_and_check_data`; BED reads them from its stats file, grouped by pandas in the
    same order). The keys are the condition names (`from_json` checks it), and the file does not list them sorted.

    `parent_index[j]` is the evidence that question j refines (its `code_question`), or -1. A `code_question`
    naming no evidence of the catalogue links nothing: the French release has five such typos ("e10, e11" for
    e10_e11 and the like), and CASANDE's simulator leaves them unlinked too. In DDXPlus the links are one
    level deep: 14 subquestions under three binary parents (pain, skin lesions, oedema).
    """

    diseases: tuple[str, ...]
    disease_names_en: tuple[str, ...]
    severity: tuple[int, ...]
    evidences: tuple[Evidence, ...]
    membership: BoolArray
    disease_index: Mapping[str, int] = field(init=False)
    evidence_index: Mapping[str, int] = field(init=False)
    parent_index: IndexArray = field(init=False)
    official_order: IndexArray = field(init=False)

    def __post_init__(self) -> None:
        if self.membership.shape != (len(self.diseases), len(self.evidences)):
            raise ValueError(f"membership shape {self.membership.shape} does not match the catalogue")
        self.membership.flags.writeable = False
        evidence_index = {e.code: j for j, e in enumerate(self.evidences)}
        parent = np.array(
            [
                evidence_index[e.parent] if e.parent != e.code and e.parent in evidence_index else -1
                for e in self.evidences
            ],
            dtype=np.intp,
        )
        nested = [self.evidences[j].code for j in np.flatnonzero(parent >= 0) if parent[parent[j]] >= 0]
        if nested:
            raise ValueError(f"subquestions of subquestions are not supported: {nested}")
        parent.flags.writeable = False
        official = np.array(sorted(range(len(self.diseases)), key=self.diseases.__getitem__), dtype=np.intp)
        official.flags.writeable = False
        object.__setattr__(self, "disease_index", {d: i for i, d in enumerate(self.diseases)})
        object.__setattr__(self, "evidence_index", evidence_index)
        object.__setattr__(self, "parent_index", parent)
        object.__setattr__(self, "official_order", official)

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
        renamed = [key for key, c in conditions.items() if key != c["condition_name"]]
        if renamed:
            raise ValueError(f"condition keys differ from their condition_name: {renamed}")
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

    def parent_of(self, code: str) -> str | None:
        j = int(self.parent_index[self.evidence_index[code]])
        return self.evidences[j].code if j >= 0 else None

    def children_of(self, code: str) -> tuple[str, ...]:
        j = self.evidence_index[code]
        return tuple(self.evidences[k].code for k in np.flatnonzero(self.parent_index == j))

    @property
    def child_mask(self) -> BoolArray:
        return np.asarray(self.parent_index >= 0)

    @property
    def linked_mask(self) -> BoolArray:
        """Evidences with a parent or a child."""
        linked = self.child_mask
        linked[self.parent_index[linked]] = True
        return linked

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
