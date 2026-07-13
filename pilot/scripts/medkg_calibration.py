#!/usr/bin/env python3
"""Calibration of medkg diagnosis scores using Platt scaling / temperature scaling.

Inputs:
  - Held-out validation set of (raw_score, true_label) pairs from DDXPlus
  - Trains logistic regression / temperature parameter to map raw_score → calibrated probability

Output:
  - calibration_params.json with sigmoid coefficients
  - For inference: raw_score → calibrated_confidence

Two methods:
  1. Platt scaling: P(correct | raw_score) = 1 / (1 + exp(-(A * raw_score + B)))
     where A, B fit via logistic regression on (raw_score, is_top1_correct)
  2. Temperature scaling: P_cal = softmax(raw_logits / T)
     where T is a single scalar fit on validation NLL

For our case (point estimate per disease, not full softmax distribution),
Platt scaling on the top-1 raw score is most natural.

Output usage at inference:
  calibrated = 1 / (1 + exp(-(A * raw + B)))

Metrics to report:
  - Brier score: mean (calibrated - true_outcome)^2
  - Expected Calibration Error (ECE): bin scores, |bin_avg_score - bin_accuracy|
  - Reliability diagram (calibrated vs empirical accuracy by bin)
"""
from __future__ import annotations
import json, math
from pathlib import Path

OUT = Path("/home/max/Graph-DDXPlus/data/medkg/kg/calibration_params.json")


def fit_platt(scores, labels):
    """Fit Platt scaling parameters via simple gradient descent (no sklearn dep)."""
    A, B = 0.0, 0.0
    lr = 0.1
    for _ in range(500):
        dA = dB = 0.0
        for s, y in zip(scores, labels):
            p = 1 / (1 + math.exp(-(A * s + B)))
            dA += (p - y) * s
            dB += (p - y)
        A -= lr * dA / len(scores)
        B -= lr * dB / len(scores)
    return A, B


def calibrate(raw_score, A, B):
    return 1 / (1 + math.exp(-(A * raw_score + B)))


def expected_calibration_error(scores, labels, n_bins=10):
    """ECE: weighted avg of |bin_score - bin_accuracy|."""
    bins = [[] for _ in range(n_bins)]
    bin_labels = [[] for _ in range(n_bins)]
    for s, y in zip(scores, labels):
        b = min(int(s * n_bins), n_bins - 1)
        bins[b].append(s)
        bin_labels[b].append(y)
    ece = 0.0
    n = len(scores)
    for b in range(n_bins):
        if not bins[b]: continue
        bin_avg = sum(bins[b]) / len(bins[b])
        bin_acc = sum(bin_labels[b]) / len(bin_labels[b])
        ece += len(bins[b]) / n * abs(bin_avg - bin_acc)
    return ece


def brier_score(scores, labels):
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(scores)


def main_train(train_scores_labels_path: Path):
    """Fit Platt on a JSONL of {raw_score, true_label_bin}."""
    scores = []
    labels = []
    with train_scores_labels_path.open() as f:
        for line in f:
            d = json.loads(line)
            scores.append(d["raw_score"])
            labels.append(int(d["true_label"]))
    A, B = fit_platt(scores, labels)
    cal_scores = [calibrate(s, A, B) for s in scores]
    ece_raw = expected_calibration_error(scores, labels)
    ece_cal = expected_calibration_error(cal_scores, labels)
    brier_raw = brier_score(scores, labels)
    brier_cal = brier_score(cal_scores, labels)
    print(f"Fit Platt: A={A:.4f}, B={B:.4f}")
    print(f"ECE: {ece_raw:.4f} (raw) → {ece_cal:.4f} (calibrated)")
    print(f"Brier: {brier_raw:.4f} (raw) → {brier_cal:.4f} (calibrated)")
    OUT.write_text(json.dumps({
        "method": "platt_scaling",
        "A": A, "B": B,
        "ece_raw": ece_raw, "ece_calibrated": ece_cal,
        "brier_raw": brier_raw, "brier_calibrated": brier_cal,
        "n_train": len(scores),
    }, indent=2))
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main_train(Path(sys.argv[1]))
    else:
        print(__doc__)
        print("Usage: medkg_calibration.py <train_scores_labels.jsonl>")
