"""What a policy returns for one patient."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ddxkb.data import Patient
from ddxkb.kb import FloatArray


@dataclass(frozen=True, eq=False, slots=True)
class Episode:
    """One patient's interaction; disease vectors are in KB order.

    `questions` are the evidence codes asked after the initial evidence, in order, up to the later of the
    stop and the largest budget; the stopped interaction is `questions[: il - 1]`. `posterior` is the
    normalized posterior at the stop. Row i of `budget_posteriors` is the posterior after exactly
    `budgets[i]` questions, reached without stopping early, and `budget_gains[i]` the largest EIG among the
    evidences still unasked there (0 when none is left).
    """

    row: int
    gold: int
    questions: tuple[str, ...]
    il: int
    posterior: FloatArray
    budgets: tuple[int, ...]
    budget_posteriors: FloatArray
    budget_gains: FloatArray

    @property
    def asked(self) -> tuple[str, ...]:
        return self.questions[: self.il - 1]

    def at_budget(self, budget: int) -> FloatArray:
        return np.asarray(self.budget_posteriors[self.budgets.index(budget)])


class Policy(Protocol):
    def run(self, patient: Patient) -> Episode: ...
