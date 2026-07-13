#!/usr/bin/env python3
"""v324: Naive Bayes baseline from DDXPlus train evidence statistics.

Per-disease P(evidence | disease) learned from 1M train patients.
Classify test patient: argmax_D [log P(D) + Σ_e log P(e | D)]

Standard supervised baseline. KG/LLM not used. Tests how distinguishable
the 49 diseases are by simple statistics — establishes upper bound for
the LLM+KG approach.
"""
from __future__ import annotations
import csv, ast, json, math
from collections import Counter, defaultdict

# Build P(e | D) from train
print("Building P(e|D) from train...")
ev_count_by_disease = defaultdict(Counter)
total_by_disease = Counter()
all_evidences = set()
with open("data/ddxplus/release_train_patients.csv") as f:
    for row in csv.DictReader(f):
        d = row["PATHOLOGY"]
        evs = ast.literal_eval(row["EVIDENCES"])
        total_by_disease[d] += 1
        for ev in evs:
            ev_count_by_disease[d][ev] += 1
            all_evidences.add(ev)
print(f"  {len(total_by_disease)} diseases, {len(all_evidences)} unique evidences, {sum(total_by_disease.values()):,} train")

# Compute log P(e | D) with Laplace smoothing
diseases = list(total_by_disease.keys())
log_p_e_given_d = {}  # (e, d) → log P(e | D)
log_p_not_e_given_d = {}  # (e, d) → log P(¬e | D)
ALPHA = 1.0  # Laplace
for d in diseases:
    total_d = total_by_disease[d]
    for e in all_evidences:
        cnt = ev_count_by_disease[d][e]
        p = (cnt + ALPHA) / (total_d + 2 * ALPHA)
        log_p_e_given_d[(e, d)] = math.log(p)
        log_p_not_e_given_d[(e, d)] = math.log(1 - p)

log_prior = {d: math.log(total_by_disease[d] / sum(total_by_disease.values())) for d in diseases}

# Eval on test
print("Evaluating on test (30K)...")
N = 30000
correct = 0
total = 0
with open("data/ddxplus/release_test_patients.csv") as f:
    for row in csv.DictReader(f):
        if total >= N: break
        true_d = row["PATHOLOGY"]
        if true_d not in log_prior: continue
        evs_set = set(ast.literal_eval(row["EVIDENCES"]))
        # Score each disease
        scores = {}
        for d in diseases:
            score = log_prior[d]
            for e in all_evidences:
                if e in evs_set:
                    score += log_p_e_given_d[(e, d)]
                else:
                    score += log_p_not_e_given_d[(e, d)]
            scores[d] = score
        pred = max(scores, key=scores.get)
        if pred == true_d: correct += 1
        total += 1
        if total % 5000 == 0: print(f"  {total}/{N}: acc={100*correct/total:.2f}%")

print(f"\nNaive Bayes 30K: GTPA@1 = {100*correct/total:.2f}% (n={total})")

# Save predictions for later integration
import numpy as np
print("\nSaving NB scores for v325 integration...")
scores_matrix = np.zeros((N, 49), dtype=np.float32)
predicted = []
true_labels = []
with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"] for dn,info in cond.items() if dn in icd_map}
dcs_list = sorted(set(fr2cui.values()))
disease_to_cui = fr2cui

idx = 0
with open("data/ddxplus/release_test_patients.csv") as f:
    for row in csv.DictReader(f):
        if idx >= N: break
        true_d = row["PATHOLOGY"]
        true_cui = fr2cui.get(true_d)
        if true_cui not in dcs_list: continue
        evs_set = set(ast.literal_eval(row["EVIDENCES"]))
        for d_idx, dc in enumerate(dcs_list):
            # Find the French disease name for this CUI
            fr_d = next((fr for fr, c in fr2cui.items() if c == dc), None)
            if not fr_d:
                scores_matrix[idx, d_idx] = -1e9
                continue
            score = log_prior.get(fr_d, -1e9)
            for e in all_evidences:
                if e in evs_set:
                    score += log_p_e_given_d.get((e, fr_d), 0)
                else:
                    score += log_p_not_e_given_d.get((e, fr_d), 0)
            scores_matrix[idx, d_idx] = score
        idx += 1

np.save("pilot/results/v324_nb_scores_30k.npy", scores_matrix)
print(f"Saved NB scores → pilot/results/v324_nb_scores_30k.npy ({scores_matrix.shape})")
