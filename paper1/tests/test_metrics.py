from __future__ import annotations

from dataclasses import astuple

import numpy as np
import pytest

from ddxkb.metrics import (
    Differential,
    PatientScores,
    Ties,
    differential_scores,
    gtpa,
    gtpa_at_k,
    il_summary,
    positive_evidence_recall,
    ranking,
    score_patient,
)


def v(*xs: float) -> np.ndarray:
    return np.asarray(xs, dtype=np.float64)


def test_gtpa1_two_way_tie_at_the_top() -> None:
    p = v(0.4, 0.4, 0.2)
    assert gtpa_at_k(p, 1, 1) == 0.5
    assert gtpa_at_k(p, 0, 1) == 0.5
    assert gtpa_at_k(p, 1, 1, Ties.FIXED) == 0.0
    assert gtpa_at_k(p, 0, 1, Ties.FIXED) == 1.0
    assert gtpa_at_k(p, 2, 1) == 0.0


def test_gtpa_at_k_tie_straddling_the_cutoff() -> None:
    p = v(0.3, 0.2, 0.2, 0.2, 0.1)
    # One disease above, three tied for ranks 2-4: two of the three slots fall inside the top 3.
    assert gtpa_at_k(p, 3, 3) == pytest.approx(2 / 3)
    assert gtpa_at_k(p, 3, 3, Ties.FIXED) == 0.0
    assert gtpa_at_k(p, 2, 3, Ties.FIXED) == 1.0
    assert gtpa_at_k(p, 3, 5) == 1.0
    assert gtpa_at_k(p, 4, 4) == 0.0
    assert gtpa_at_k(p, 0, 1) == 1.0


def test_near_equal_probabilities_are_one_tie() -> None:
    p = v(0.45, 0.45 * (1 + 1e-12), 0.1)
    assert gtpa_at_k(p, 0, 1) == 0.5
    assert gtpa_at_k(p, 0, 1, Ties.FIXED) == 1.0
    assert ranking(p).tolist() == [0, 1, 2]


def test_ranking_breaks_ties_by_kb_order() -> None:
    assert ranking(v(0.1, 0.3, 0.2, 0.3, 0.1)).tolist() == [1, 3, 2, 0, 4]


def test_mass_at_or_below_001_is_removed() -> None:
    p = v(0.985, 0.01, 0.005)
    assert gtpa(p, 1) == 0.0
    assert gtpa_at_k(p, 1, 2) == 0.0
    q = v(0.98, 0.0100001, 0.0099999)
    assert gtpa(q, 1) == 1.0
    assert gtpa_at_k(q, 1, 2) == 1.0
    assert gtpa_at_k(q, 2, 3) == 0.0


def test_differential_recall_precision_f1() -> None:
    pred = v(0.5, 0.3, 0.195, 0.005)
    truth = v(0.9, 0.0, 0.01, 0.09)
    # pred keeps {0, 1, 2}; truth keeps {0, 3} (0.01 is removed): one shared disease.
    d = differential_scores(pred, truth)
    assert d.ddr == pytest.approx(1 / 2)
    assert d.ddp == pytest.approx(1 / 3)
    assert d.ddf1 == pytest.approx(2 * (1 / 2) * (1 / 3) / (1 / 2 + 1 / 3))
    assert differential_scores(v(0.0, 1.0), v(1.0, 0.0)) == Differential(0.0, 0.0, 0.0)


def test_ddf1_is_the_mean_of_patient_f1() -> None:
    a = differential_scores(v(0.5, 0.5, 0.0), v(0.5, 0.5, 0.0))
    b = differential_scores(v(1.0, 0.0, 0.0), v(0.4, 0.3, 0.3))
    assert (a.ddr, a.ddp, a.ddf1) == (1.0, 1.0, 1.0)
    assert (b.ddr, b.ddp) == pytest.approx((1 / 3, 1.0))
    assert b.ddf1 == pytest.approx(0.5)
    # Mean of per-patient F1 (0.75) differs from the F1 of mean DDR and DDP (0.8).
    assert (a.ddf1 + b.ddf1) / 2 == pytest.approx(0.75)


def test_positive_evidence_recall_excludes_the_initial_evidence() -> None:
    positives = frozenset({"a", "b", "c", "d"})
    assert positive_evidence_recall(["b", "x", "c"], positives, initial="a") == 0.5
    assert positive_evidence_recall(["a", "b"], positives, initial="a") == 0.25
    assert positive_evidence_recall([], frozenset({"a"}), initial="a") == 0.0
    with pytest.raises(ValueError):
        positive_evidence_recall([], frozenset(), initial="a")


def test_il_summary() -> None:
    s = il_summary([2, 4, 6, 8, 30])
    assert s.mean == 10.0
    assert s.median == 6.0
    assert s.p95 == pytest.approx(8 + 0.8 * 22)
    assert s.share_at_cap == 0.2
    assert il_summary([1]).mean == 1.0
    with pytest.raises(ValueError):
        il_summary([0, 3])
    with pytest.raises(ValueError):
        il_summary([31])


def test_score_patient_collects_every_metric() -> None:
    probs = v(0.4, 0.4, 0.15, 0.05)
    truth = v(0.6, 0.0, 0.3, 0.1)
    s = score_patient(probs, 1, truth)
    assert isinstance(s, PatientScores)
    # gtpa1, gtpa1_fixed, gtpa3, gtpa5, gtpa, ddr, ddp, ddf1
    assert astuple(s) == pytest.approx((0.5, 0.0, 1.0, 1.0, 1.0, 1.0, 0.75, 2 * 0.75 / 1.75))
