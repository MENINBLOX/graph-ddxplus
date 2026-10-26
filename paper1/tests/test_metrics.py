from __future__ import annotations

from dataclasses import astuple

import numpy as np
import pytest

from ddxkb.metrics import (
    Differential,
    PatientScores,
    Severe,
    Ties,
    class_means,
    differential_scores,
    gtpa,
    gtpa_at_k,
    gtpa_at_k_official,
    il_summary,
    official_top,
    positive_evidence_recall,
    positive_evidence_recall_official,
    ranking,
    score_patient,
    severe_scores,
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


# KB file order Gamma, Alpha, Beta, Delta; the official code indexes diseases by sorted name.
NAMES = ("Gamma", "Alpha", "Beta", "Delta")
ORDER = np.array(sorted(range(len(NAMES)), key=NAMES.__getitem__), dtype=np.intp)


def _verbatim(batch: np.ndarray, gold: np.ndarray, k: int) -> np.ndarray:
    """The official expression on a batch indexed the official way:
    any(disease == argsort(pred)[:, ::-1][:, :k]) and pred[disease] > 0.01."""
    top = np.argsort(batch, axis=-1)[:, ::-1][:, :k]
    held = batch[np.arange(len(batch)), gold] > 0.01
    return np.asarray(np.any(gold.reshape(-1, 1) == top, axis=-1) & held, dtype=np.float64)


def test_official_gtpa_at_k_is_the_official_expression_over_the_sorted_disease_order() -> None:
    assert [NAMES[i] for i in ORDER] == ["Alpha", "Beta", "Delta", "Gamma"]
    batch = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.4, 0.2, 0.0],
            [0.1, 0.45, 0.0, 0.45],
            [0.0, 0.0, 1.0, 0.0],
            [0.3, 0.2, 0.2, 0.3],
            [0.005, 0.004, 0.003, 0.988],
        ]
    )
    for k in (1, 2, 3):
        for gold in range(len(NAMES)):
            # The vectors re-indexed the official way, and the gold disease with them.
            expected = _verbatim(batch[:, ORDER], np.full(len(batch), int(np.flatnonzero(gold == ORDER)[0])), k)
            mine = [gtpa_at_k_official(probs, gold, k, ORDER) for probs in batch]
            assert mine == expected.tolist(), (k, gold)
    # A tie at the top scores exactly one of the tied diseases. Whichever position m numpy's kernel leaves on top,
    # the official winner is the disease at position m of the official order, never the one at position m of the
    # KB order, since no disease keeps its position here.
    assert sum(gtpa_at_k_official(batch[0], g, 1, ORDER) for g in range(4)) == 1.0
    assert not (np.arange(len(NAMES)) == ORDER).any()
    m = int(np.argsort(batch[0])[::-1][0])
    assert official_top(batch[0], 1, ORDER)[0] == ORDER[m] != m
    golds = np.full(len(batch), m)
    assert _verbatim(batch, golds, 1)[0] == 1.0 and gtpa_at_k_official(batch[0], m, 1, ORDER) == 0.0
    assert gtpa_at_k_official(batch[5], 0, 3, ORDER) == 0.0
    assert official_top(batch[3], 2, ORDER)[0] == 2
    with pytest.raises(ValueError):
        official_top(batch[0], 0, ORDER)


def test_severe_scores_by_hand() -> None:
    severe = np.array([True, True, True, False, True])
    truth = v(0.5, 0.3, 0.0, 0.2, 0.0)
    # Severe in the truth: 0, 1. The prediction keeps 0 and 2 among the severe ones (4 is at 0.01, removed).
    pred = v(0.6, 0.0, 0.2, 0.19, 0.01)
    s = severe_scores(pred, truth, severe)
    # Rule-out: of the severe diseases outside the truth (2, 4), the prediction also leaves out 4 only.
    # Rule-in: of the severe diseases in the truth (0, 1), the prediction holds 0.
    assert (s.rule_out, s.rule_in) == (0.5, 0.5)
    assert s.dshm == pytest.approx(0.5)
    # No severe disease in the truth: rule-in is 0, and so is the harmonic mean, as in the official code.
    assert severe_scores(v(0.0, 0.0, 0.0, 1.0, 0.0), v(0.0, 0.0, 0.0, 1.0, 0.0), severe) == Severe(1.0, 0.0, 0.0)
    assert severe_scores(truth, truth, severe).dshm == pytest.approx(1.0)


def test_positive_evidence_recall_excludes_the_initial_evidence() -> None:
    positives = frozenset({"a", "b", "c", "d"})
    assert positive_evidence_recall(["b", "x", "c"], positives, initial="a") == 0.5
    assert positive_evidence_recall(["a", "b"], positives, initial="a") == 0.25
    assert positive_evidence_recall([], frozenset({"a"}), initial="a") == 0.0
    with pytest.raises(ValueError):
        positive_evidence_recall([], frozenset(), initial="a")


def test_official_positive_evidence_recall_counts_the_initial_evidence() -> None:
    positives = frozenset({"a", "b", "c", "d"})
    assert positive_evidence_recall_official(["b", "x", "c"], positives, initial="a") == 0.75
    assert positive_evidence_recall_official(["a", "b"], positives, initial="a") == 0.5
    assert positive_evidence_recall_official([], frozenset({"a"}), initial="a") == 1.0
    with pytest.raises(ValueError):
        positive_evidence_recall_official([], frozenset(), initial="a")


def test_class_means() -> None:
    means, counts = class_means(v(1.0, 0.0, 0.5, 1.0, 0.25), np.array([0, 0, 2, 2, -1]), 4)
    assert means.tolist() == [0.5, 0.0, 0.75, 0.0]
    assert counts.tolist() == [2, 0, 2, 0]


def test_il_summary() -> None:
    s = il_summary([2, 4, 6, 8, 31], cap=31)
    assert s.mean == 10.2
    assert s.median == 6.0
    assert s.p95 == pytest.approx(8 + 0.8 * 23)
    assert s.share_at_cap == 0.2
    assert s.questions_mean == pytest.approx(9.2)
    assert il_summary([1], cap=31).mean == 1.0
    with pytest.raises(ValueError):
        il_summary([0, 3], cap=31)
    with pytest.raises(ValueError):
        il_summary([32], cap=31)


def test_score_patient_collects_every_metric() -> None:
    probs = v(0.4, 0.4, 0.15, 0.05)
    truth = v(0.6, 0.0, 0.3, 0.1)
    severe = np.array([True, False, True, False])
    s = score_patient(probs, 1, truth, severe, ORDER)
    assert isinstance(s, PatientScores)
    official = [gtpa_at_k_official(probs, 1, k, ORDER) for k in (1, 3, 5)]
    assert official[1:] == [1.0, 1.0]
    dshm = severe_scores(probs, truth, severe).dshm
    # gtpa1, gtpa1_fixed, gtpa1_official, gtpa3, gtpa3_official, gtpa5, gtpa5_official, gtpa, ddr, ddp, ddf1, dshm
    expected = (0.5, 0.0, official[0], 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 2 * 0.75 / 1.75, dshm)
    assert astuple(s) == pytest.approx(expected)
    # Every severe disease is in the truth, so no severe disease is left to rule out: rule-out is 0 / max(1, 0).
    assert severe_scores(probs, truth, severe) == Severe(0.0, 1.0, 0.0)
