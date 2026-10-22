"""The oracle patient of the interactive protocol (design §2).

A record value equal to the evidence's default value counts as absent, as the DDXPlus README defines
("as if the evidence was not synthesized"). So `trav1_@_N` (no travel) answers "no".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from ddxkb.data import EvidencePair, Patient
from ddxkb.kb import KB

MAX_IL = 30


class AnswerMode(StrEnum):
    BINARY = "binary"
    VALUE = "value"


@dataclass(frozen=True, slots=True)
class Answer:
    """`values` lists the non-default values in catalogue order; it is empty in binary mode, for binary
    evidences, and for negative answers (a categorical answer of () means the default value)."""

    code: str
    positive: bool
    values: tuple[str, ...] = ()


def present_values(kb: KB, evidences: Iterable[EvidencePair]) -> dict[str, tuple[str, ...]]:
    """Positive evidences of a record mapped to their non-default values (() for binary evidences)."""
    found: dict[str, set[str]] = {}
    for code, value in evidences:
        e = kb.evidence(code)
        if e.is_binary:
            if value is not None:
                raise ValueError(f"binary evidence {code} carries a value {value!r}")
            found.setdefault(code, set())
            continue
        if value is None:
            raise ValueError(f"{e.data_type.name.lower()} evidence {code} has no value")
        if value not in e.values:
            raise ValueError(f"{code}: value {value!r} is not a possible value")
        if value != e.default:
            found.setdefault(code, set()).add(value)
    return {
        code: tuple(v for v in kb.evidence(code).values if v in vals)
        for code, vals in found.items()
        if kb.evidence(code).is_binary or vals
    }


def positive_codes(kb: KB, patient: Patient) -> frozenset[str]:
    return frozenset(present_values(kb, patient.evidences))


class Oracle:
    """Answers any evidence question about one patient from the record."""

    def __init__(self, kb: KB, patient: Patient, mode: AnswerMode = AnswerMode.BINARY) -> None:
        self.kb = kb
        self.patient = patient
        self.mode = mode
        self._present = present_values(kb, patient.evidences)

    @property
    def positives(self) -> frozenset[str]:
        return frozenset(self._present)

    def answer(self, code: str) -> Answer:
        if code not in self.kb.evidence_index:
            raise KeyError(f"unknown evidence {code!r}")
        if code not in self._present:
            return Answer(code, positive=False)
        values = self._present[code] if self.mode is AnswerMode.VALUE else ()
        return Answer(code, positive=True, values=values)


class Interaction:
    """One episode: the initial evidence is given as T's first element, then one question per `ask`.

    IL is |T| including the initial evidence (DDXPlus eq. 4); `ask` refuses a repeated question and any
    question once |T| = max_il.
    """

    def __init__(self, oracle: Oracle, max_il: int = MAX_IL) -> None:
        if max_il < 1:
            raise ValueError("max_il must be at least 1")
        initial = oracle.patient.initial
        if not oracle.kb.evidence(initial).is_binary:
            raise ValueError(f"initial evidence {initial} is not binary")
        first = oracle.answer(initial)
        if not first.positive:
            raise ValueError(f"initial evidence {initial} is not positive in the patient record")
        self.oracle = oracle
        self.max_il = max_il
        self.answers: list[Answer] = [first]
        self._asked = {initial}

    @property
    def initial(self) -> Answer:
        return self.answers[0]

    @property
    def il(self) -> int:
        return len(self.answers)

    @property
    def remaining(self) -> int:
        return self.max_il - self.il

    def ask(self, code: str) -> Answer:
        if code in self._asked:
            raise ValueError(f"{code} was already asked")
        if self.remaining <= 0:
            raise RuntimeError(f"interaction length limit {self.max_il} reached")
        a = self.oracle.answer(code)
        self._asked.add(code)
        self.answers.append(a)
        return a
