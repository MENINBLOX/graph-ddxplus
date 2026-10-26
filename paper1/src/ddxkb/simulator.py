"""The oracle patient and the interactive protocol (design 보완 2, §3).

A record value equal to the evidence's default value counts as absent, as the DDXPlus README defines
("as if the evidence was not synthesized"). So `trav1_@_N` (no travel) answers "no". The oracle reveals values
the way the official simulators do: a categorical answer carries its one value, a multi-choice answer all of
its values. A policy that ignores them answers binary questions only.

The protocol gives the age, the sex and the initial evidence, then up to 30 questions: IL counts the initial
evidence, so it is at most 31 (DDXPlus `basd/metrics.py`: `sum(valid_timesteps) + 1`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ddxkb.data import EvidencePair, Patient
from ddxkb.kb import KB, BoolArray, DataType

MAX_IL = 31


class AnswerMode(StrEnum):
    """Whether a policy reads the revealed values (VALUE) or only whether an answer is positive (BINARY)."""

    BINARY = "binary"
    VALUE = "value"


class Hierarchy(StrEnum):
    """How a subquestion (`KB.parent_index`) relates to its parent question.

    NONE (main, as BASD, AARLC and BED): only the asked evidence is answered, and a child may be asked first.
    IMPLICIT (CASANDE `simulator.py`, `_set_symptom_to_observation`): a negative parent answers its unknown
    children negative and a positive child answers its unknown parent positive, without using a turn.
    PARENT_FIRST: a child can be asked only once its parent is known positive.
    NO_CHILDREN: a child is never asked.
    """

    NONE = "none"
    IMPLICIT = "implicit"
    PARENT_FIRST = "parent_first"
    NO_CHILDREN = "no_children"


class StartInput(StrEnum):
    AGE = "age"
    SEX = "sex"
    INITIAL_EVIDENCE = "initial_evidence"


@dataclass(frozen=True, slots=True)
class Answer:
    """`values` lists the non-default values in catalogue order; it is empty for binary evidences and for
    negative answers (a categorical answer of () means the default value)."""

    code: str
    positive: bool
    values: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Start:
    """What a policy is given before its first question; age and sex are None unless the policy declares them."""

    initial: Answer
    age: int | None = None
    sex: str | None = None


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
    out = {
        code: tuple(v for v in kb.evidence(code).values if v in vals)
        for code, vals in found.items()
        if kb.evidence(code).is_binary or vals
    }
    for code, vals in out.items():
        if kb.evidence(code).data_type is DataType.CATEGORICAL and len(vals) > 1:
            raise ValueError(f"categorical evidence {code} holds {len(vals)} values")
    return out


def positive_codes(kb: KB, patient: Patient) -> frozenset[str]:
    return frozenset(present_values(kb, patient.evidences))


class Oracle:
    """Answers any evidence question about one patient from the record."""

    def __init__(self, kb: KB, patient: Patient) -> None:
        self.kb = kb
        self.patient = patient
        self._present = present_values(kb, patient.evidences)

    @property
    def positives(self) -> frozenset[str]:
        return frozenset(self._present)

    def answer(self, code: str) -> Answer:
        if code not in self.kb.evidence_index:
            raise KeyError(f"unknown evidence {code!r}")
        if code not in self._present:
            return Answer(code, positive=False)
        return Answer(code, positive=True, values=self._present[code])


class Interaction:
    """One episode: the initial evidence is given as T's first element, then one question per `ask`.

    IL is |T| including the initial evidence (DDXPlus eq. 4); `ask` refuses a question whose answer is known,
    a question the hierarchy rule closes, and any question once |T| = max_il. `inputs` are what the policy
    declares it uses at the start; the initial evidence is always part of T.
    """

    def __init__(
        self,
        oracle: Oracle,
        max_il: int = MAX_IL,
        hierarchy: Hierarchy = Hierarchy.NONE,
        inputs: frozenset[StartInput] = frozenset({StartInput.INITIAL_EVIDENCE}),
    ) -> None:
        if max_il < 1:
            raise ValueError("max_il must be at least 1")
        if StartInput.INITIAL_EVIDENCE not in inputs:
            raise ValueError("the protocol starts from the initial evidence")
        kb = oracle.kb
        initial = oracle.patient.initial
        if not kb.evidence(initial).is_binary:
            raise ValueError(f"initial evidence {initial} is not binary")
        first = oracle.answer(initial)
        if not first.positive:
            raise ValueError(f"initial evidence {initial} is not positive in the patient record")
        self.oracle = oracle
        self.max_il = max_il
        self.hierarchy = hierarchy
        patient = oracle.patient
        self.start = Start(
            first,
            age=patient.age if StartInput.AGE in inputs else None,
            sex=patient.sex if StartInput.SEX in inputs else None,
        )
        self.answers: list[Answer] = [first]
        self.implied: list[Answer] = []
        self._known = np.zeros(kb.n_evidences, dtype=np.bool_)
        self._positive = np.zeros(kb.n_evidences, dtype=np.bool_)
        self._record(first)
        self._imply(first)

    @property
    def initial(self) -> Answer:
        return self.answers[0]

    @property
    def il(self) -> int:
        return len(self.answers)

    @property
    def remaining(self) -> int:
        return self.max_il - self.il

    @property
    def positive(self) -> BoolArray:
        """Known positive answers in KB order, implied ones included; a read-only view."""
        view = self._positive.view()
        view.flags.writeable = False
        return view

    def closed(self) -> BoolArray:
        """Evidences that cannot be asked now: answered, implied, or a child the hierarchy rule holds back."""
        closed = self._known.copy()
        parents = self.oracle.kb.parent_index
        children = parents >= 0
        if self.hierarchy is Hierarchy.PARENT_FIRST:
            closed[children] |= ~self._positive[parents[children]]
        elif self.hierarchy is Hierarchy.NO_CHILDREN:
            closed |= children
        closed.flags.writeable = False
        return closed

    def ask(self, code: str) -> tuple[Answer, ...]:
        """The answer to `code`, then the answers it implies without a turn."""
        kb = self.oracle.kb
        if code not in kb.evidence_index:
            raise KeyError(f"unknown evidence {code!r}")
        j = kb.evidence_index[code]
        if self._known[j]:
            raise ValueError(f"{code} was already asked or implied")
        if self.remaining <= 0:
            raise RuntimeError(f"interaction length limit {self.max_il} reached")
        if self.closed()[j]:
            if self.hierarchy is Hierarchy.NO_CHILDREN:
                raise ValueError(f"{code} is a subquestion of {kb.parent_of(code)}; subquestions are never asked")
            raise ValueError(f"{code} cannot be asked before its parent {kb.parent_of(code)} is positive")
        a = self.oracle.answer(code)
        self.answers.append(a)
        self._record(a)
        return (a, *self._imply(a))

    def _record(self, a: Answer) -> None:
        j = self.oracle.kb.evidence_index[a.code]
        self._known[j] = True
        self._positive[j] = a.positive

    def _imply(self, a: Answer) -> list[Answer]:
        if self.hierarchy is not Hierarchy.IMPLICIT:
            return []
        kb = self.oracle.kb
        if a.positive:
            parent = kb.parent_of(a.code)
            targets, expected = ([] if parent is None else [parent]), True
        else:
            targets, expected = list(kb.children_of(a.code)), False
        implied = []
        for code in targets:
            if self._known[kb.evidence_index[code]]:
                continue
            b = self.oracle.answer(code)
            # Records never hold a positive child without its parent, so the implied answer is the recorded one.
            if b.positive is not expected:
                raise ValueError(f"record contradicts the hierarchy: {a.code} {a.positive}, {code} {b.positive}")
            self._record(b)
            implied.append(b)
        self.implied += implied
        return implied
