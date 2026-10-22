"""Evaluation metrics of DDXPlus (Fansi Tchango et al. 2022, App. H) as fixed by design §4.

Predicted and ground-truth differentials are both post-processed to drop diseases with mass <= 0.01
before GTPA, GTPA@k and DDR/DDP/DDF1. Probability vectors are indexed in KB disease order, and that
order is the fixed tie-break.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from ddxkb.kb import BoolArray, FloatArray

POSTPROCESS_MASS = 0.01
# Probabilities this close (relative) are one tie; equal log-likelihood sums can differ in the last bits.
TIE_RTOL = 1e-9


class Ties(StrEnum):
    EXPECTED = "expected"
    FIXED = "fixed"


def postprocess(probs: FloatArray, mass: float = POSTPROCESS_MASS) -> BoolArray:
    return np.asarray(probs > mass)


def ranking(probs: FloatArray) -> npt.NDArray[np.intp]:
    """Disease indices by descending probability; ties within TIE_RTOL keep KB order."""
    order = np.argsort(-probs, kind="stable")
    out: list[int] = []
    i = 0
    while i < len(order):
        top = probs[order[i]]
        j = i + 1
        while j < len(order) and _close(probs[order[j]], top):
            j += 1
        out.extend(sorted(int(k) for k in order[i:j]))
        i = j
    return np.asarray(out, dtype=np.intp)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TIE_RTOL * max(abs(a), abs(b))


def gtpa_at_k(
    probs: FloatArray, gold: int, k: int, ties: Ties = Ties.EXPECTED, mass: float = POSTPROCESS_MASS
) -> float:
    """Whether the gold disease is in the top k of the post-processed differential.

    EXPECTED: the expectation over a uniformly random order within ties, so a gold disease tied with
    t - 1 others at rank 1 scores 1/t at k = 1. FIXED: ties broken by KB order, a 0/1 score.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    p = float(probs[gold])
    if p <= mass:
        return 0.0
    tied = np.abs(probs - p) <= TIE_RTOL * np.maximum(np.abs(probs), abs(p))
    above = int(np.count_nonzero((probs > p) & ~tied))
    if ties is Ties.FIXED:
        return float(above + int(np.count_nonzero(tied[:gold])) < k)
    return float(np.clip((k - above) / np.count_nonzero(tied), 0.0, 1.0))


def gtpa(probs: FloatArray, gold: int, mass: float = POSTPROCESS_MASS) -> float:
    return float(probs[gold] > mass)


@dataclass(frozen=True, slots=True)
class Differential:
    ddr: float
    ddp: float
    ddf1: float


def differential_scores(probs: FloatArray, truth: FloatArray, mass: float = POSTPROCESS_MASS) -> Differential:
    """Per-patient DDR, DDP and their F1 (DDF1 is the mean of per-patient F1s); an empty set scores 0."""
    pred = postprocess(probs, mass)
    gt = postprocess(truth, mass)
    hit = int(np.count_nonzero(pred & gt))
    n_pred = int(np.count_nonzero(pred))
    n_gt = int(np.count_nonzero(gt))
    r = hit / n_gt if n_gt else 0.0
    p = hit / n_pred if n_pred else 0.0
    return Differential(ddr=r, ddp=p, ddf1=2 * r * p / (r + p) if r + p > 0 else 0.0)


def positive_evidence_recall(asked: Iterable[str], positives: frozenset[str], initial: str) -> float:
    """DDXPlus eq. 9: the initial evidence is given, not asked, so it never counts as found but stays in
    the denominator with every other positive evidence of the patient."""
    if not positives:
        raise ValueError("a patient has at least one positive evidence (the initial one)")
    found = (set(asked) - {initial}) & positives
    return len(found) / len(positives)


@dataclass(frozen=True, slots=True)
class PatientScores:
    gtpa1: float
    gtpa1_fixed: float
    gtpa3: float
    gtpa5: float
    gtpa: float
    ddr: float
    ddp: float
    ddf1: float


def score_patient(probs: FloatArray, gold: int, truth: FloatArray, mass: float = POSTPROCESS_MASS) -> PatientScores:
    d = differential_scores(probs, truth, mass)
    return PatientScores(
        gtpa1=gtpa_at_k(probs, gold, 1, Ties.EXPECTED, mass),
        gtpa1_fixed=gtpa_at_k(probs, gold, 1, Ties.FIXED, mass),
        gtpa3=gtpa_at_k(probs, gold, 3, Ties.EXPECTED, mass),
        gtpa5=gtpa_at_k(probs, gold, 5, Ties.EXPECTED, mass),
        gtpa=gtpa(probs, gold, mass),
        ddr=d.ddr,
        ddp=d.ddp,
        ddf1=d.ddf1,
    )


@dataclass(frozen=True, slots=True)
class ILSummary:
    mean: float
    median: float
    p95: float
    share_at_cap: float


def il_summary(lengths: Sequence[int] | npt.NDArray[np.int_], cap: int = 30) -> ILSummary:
    """IL counts the initial evidence (DDXPlus eq. 4); `share_at_cap` is the share with |T| = cap."""
    x = np.asarray(lengths, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no interaction lengths")
    if (x < 1).any() or (x > cap).any():
        raise ValueError(f"interaction lengths must lie in [1, {cap}]")
    return ILSummary(
        mean=float(x.mean()),
        median=float(np.median(x)),
        p95=float(np.percentile(x, 95, method="linear")),
        share_at_cap=float(np.mean(x == cap)),
    )
