"""Evaluation metrics of DDXPlus (Fansi Tchango et al. 2022, App. H) and CASANDE, fixed by design §4 and 보완 2.

Predicted and ground-truth differentials are both post-processed to drop diseases with mass <= 0.01
before GTPA, GTPA@k, DDR/DDP/DDF1 and DSHM. Probability vectors are indexed in KB disease order, and that
order is the fixed tie-break. Where this study's definition and the official code's differ, both are kept:
GTPA@k tie-aware (judgments) and as the code computes it (literature, `gtpa_at_k_official`), PER without and with
the initial evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from ddxkb.kb import BoolArray, FloatArray, IndexArray

IntArray = npt.NDArray[np.int64]

POSTPROCESS_MASS = 0.01
# Probabilities this close (relative) are one tie; equal log-likelihood sums can differ in the last bits.
TIE_RTOL = 1e-9
# CASANDE counts a disease as severe below this release_conditions.json severity (`severity_threshold = 3`).
SEVERE_BELOW = 3


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


def official_top(probs: FloatArray, k: int, order: IndexArray) -> IndexArray:
    """The first k diseases of the official code's ranking `argsort(pred)[::-1]`, as KB indices, where `pred` is
    `probs` indexed the official way: `order` is `KB.official_order`.

    Within an exact tie the winner is whatever numpy's sort kernel leaves last, which depends on the numpy version
    and the SIMD kernel it dispatches to (`provenance.environment` records both), and on the index order.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    return np.asarray(order[np.argsort(probs[order])[::-1][:k]])


def gtpa_at_k_official(
    probs: FloatArray, gold: int, k: int, order: IndexArray, mass: float = POSTPROCESS_MASS
) -> float:
    """GTPA@k as DDXPlus `basd/metrics.py`, CASANDE `chloe/evaluator/metrics.py` and BED `compute_metrics` compute
    it: the gold disease is among the first k of `official_top` and holds more than `mass`."""
    return float(gold in official_top(probs, k, order) and probs[gold] > mass)


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


@dataclass(frozen=True, slots=True)
class Severe:
    """CASANDE's severe-pathology scores of one patient (`compute_severity_stats`). `rule_out`: share of the
    severe diseases outside the true differential that the prediction also leaves out (DSP); `rule_in`: share of
    the severe diseases in the true differential that the prediction holds (DSR, 0 when there are none); `dshm`:
    their harmonic mean (DSF1, reported as DSHM)."""

    rule_out: float
    rule_in: float
    dshm: float


def severe_scores(probs: FloatArray, truth: FloatArray, severe: BoolArray, mass: float = POSTPROCESS_MASS) -> Severe:
    gt = postprocess(truth, mass) & severe
    pred = postprocess(probs, mass) & severe
    n_severe, n_gt = int(np.count_nonzero(severe)), int(np.count_nonzero(gt))
    out = int(np.count_nonzero(pred & ~gt))
    missed = int(np.count_nonzero(gt & ~pred))
    rule_out = (n_severe - n_gt - out) / max(1, n_severe - n_gt)
    rule_in = (n_gt - missed) / max(1, n_gt)
    # The official harmonic mean adds 1e-10 to the denominator (`compute_f1`).
    return Severe(rule_out, rule_in, 2 * rule_out * rule_in / (rule_out + rule_in + 1e-10))


def positive_evidence_recall(asked: Iterable[str], positives: frozenset[str], initial: str) -> float:
    """DDXPlus eq. 9: the initial evidence is given, not asked, so it never counts as found but stays in
    the denominator with every other positive evidence of the patient."""
    if not positives:
        raise ValueError("a patient has at least one positive evidence (the initial one)")
    found = (set(asked) - {initial}) & positives
    return len(found) / len(positives)


def positive_evidence_recall_official(asked: Iterable[str], positives: frozenset[str], initial: str) -> float:
    """PER as the official code counts it (`basd/env.py` marks the initial evidence inquired): the initial
    evidence counts as found. Answers implied by a hierarchy rule are not inquired."""
    if not positives:
        raise ValueError("a patient has at least one positive evidence (the initial one)")
    return len(({*asked, initial}) & positives) / len(positives)


def class_means(values: FloatArray, labels: npt.NDArray[np.int64], n_classes: int) -> tuple[FloatArray, IntArray]:
    """Per-class mean of `values` and per-class count; a class without members has mean 0 and count 0.
    Negative labels are left out."""
    keep = labels >= 0
    counts = np.bincount(labels[keep], minlength=n_classes)
    sums = np.bincount(labels[keep], weights=values[keep], minlength=n_classes)
    means = np.divide(sums, counts, out=np.zeros(n_classes), where=counts > 0)
    return np.asarray(means, dtype=np.float64), np.asarray(counts, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class PatientScores:
    gtpa1: float
    gtpa1_fixed: float
    gtpa1_official: float
    gtpa3: float
    gtpa3_official: float
    gtpa5: float
    gtpa5_official: float
    gtpa: float
    ddr: float
    ddp: float
    ddf1: float
    dshm: float


def score_patient(
    probs: FloatArray,
    gold: int,
    truth: FloatArray,
    severe: BoolArray,
    order: IndexArray,
    mass: float = POSTPROCESS_MASS,
) -> PatientScores:
    """Every metric of one patient; `order` is `KB.official_order`, which the official GTPA@k read."""
    d = differential_scores(probs, truth, mass)
    return PatientScores(
        gtpa1=gtpa_at_k(probs, gold, 1, Ties.EXPECTED, mass),
        gtpa1_fixed=gtpa_at_k(probs, gold, 1, Ties.FIXED, mass),
        gtpa1_official=gtpa_at_k_official(probs, gold, 1, order, mass),
        gtpa3=gtpa_at_k(probs, gold, 3, Ties.EXPECTED, mass),
        gtpa3_official=gtpa_at_k_official(probs, gold, 3, order, mass),
        gtpa5=gtpa_at_k(probs, gold, 5, Ties.EXPECTED, mass),
        gtpa5_official=gtpa_at_k_official(probs, gold, 5, order, mass),
        gtpa=gtpa(probs, gold, mass),
        ddr=d.ddr,
        ddp=d.ddp,
        ddf1=d.ddf1,
        dshm=severe_scores(probs, truth, severe, mass).dshm,
    )


@dataclass(frozen=True, slots=True)
class ILSummary:
    mean: float
    median: float
    p95: float
    share_at_cap: float
    questions_mean: float


def il_summary(lengths: Sequence[int] | npt.NDArray[np.int_], cap: int) -> ILSummary:
    """IL counts the initial evidence (DDXPlus eq. 4); `share_at_cap` is the share with |T| = cap and
    `questions_mean` the mean number of questions after the initial evidence (IL - 1), as the LLM-agent
    literature counts."""
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
        questions_mean=float(x.mean()) - 1.0,
    )
