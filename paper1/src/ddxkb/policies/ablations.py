"""Policy parts that only the ablations of design §3 use: A1 (`HeardAnswers`) and A5 (`CoverageOrder`).

The other ablations swap a table, prior or hierarchy rule of P1 and need no code of their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ddxkb.data import Patient
from ddxkb.kb import BoolArray, FloatArray
from ddxkb.policies.bayes import LikelihoodTable, QuestionChoice
from ddxkb.simulator import Answer


class Heard(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True, eq=False)
class HeardAnswers(LikelihoodTable):
    """A1: answers of the other polarity are asked and recorded but leave the posterior unchanged.

    The initial evidence is a positive answer, so NEGATIVE starts from the prior. Questions and the EIG stop
    still use the EIG over the full table: the ablation removes what an answer does, not which questions are asked.
    """

    heard: Heard = Heard.POSITIVE

    def log_likelihood(self, answer: Answer) -> FloatArray:
        if answer.positive == (self.heard is Heard.POSITIVE):
            return super().log_likelihood(answer)
        return np.zeros(self.kb.n_diseases)


def heard_answers(table: LikelihoodTable, heard: Heard) -> HeardAnswers:
    return HeardAnswers(
        kb=table.kb,
        mode=table.mode,
        probs=table.probs,
        component_start=table.component_start,
        evidence_start=table.evidence_start,
        heard=heard,
    )


@dataclass(frozen=True)
class CoverageOrder:
    """A5: the open evidence on the most lists of the KB candidate diseases; ties in KB order.

    Candidates are those of `RandomInKBCandidates` (B2): the diseases with the fewest positive answers off
    their lists. The choice ignores the posterior, the way the earlier study's coverage rule did.
    """

    def start(self, table: LikelihoodTable, patient: Patient) -> QuestionChoice:
        membership = table.kb.membership
        off_list = ~membership

        def choose(posterior: FloatArray, closed: BoolArray, positive: BoolArray, gain: FloatArray) -> int:
            violations = (off_list & positive).sum(axis=1)
            coverage = membership[violations == violations.min()].sum(axis=0)
            open_coverage = np.where(closed, -1, coverage)
            if open_coverage.max() < 0:
                raise RuntimeError("no evidence is left to ask")
            return int(np.argmax(open_coverage))

        return choose
