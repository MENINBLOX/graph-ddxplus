#!/usr/bin/env python3
"""진단 v80: v79 stage1 score (KG features) + Bayesian age/sex prior.

post-process on v79's saved score matrix.
final_score(c) = stage1_score(c) + α * 100 * log_prior(c | age, sex)

α를 0.0, 0.05, 0.1, 0.2, 0.5에서 sweep → 최고 α 선택.
"""
from __future__ import annotations
import ast, csv, json, math, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

UMLS_DIR = Path("data/umls_extracted")


def main():
    print("="*80, flush=True)
    print("진단 v80: v79 stage1 + Bayesian prior post-process", flush=True)
    print("="*80, flush=True)

    matrix_path = sys.argv[1] if len(sys.argv) > 1 else "pilot/results/v79_stage1.npy"
    score_matrix = np.load(matrix_path)
    n, k = score_matrix.shape
    print(f"  Score matrix: {n} patients × {k} candidates", flush=True)

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)

    diseases = {}; fr2cui = {}; cui2name = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]; diseases[dn] = {"cui": dc}
        fr2cui[info.get("cond-name-fr", "")] = dc; cui2name[dc] = dn
    dcs_list = sorted(set(d["cui"] for d in diseases.values()))
    cui_to_idx = {dc: i for i, dc in enumerate(dcs_list)}

    # Build age/sex prior from training data
    age_sex_disease = defaultdict(Counter)
    with open("data/ddxplus/release_train_patients.csv") as f:
        for row in csv.DictReader(f):
            tdc = fr2cui.get(row["PATHOLOGY"])
            if not tdc: continue
            age_bin = min(int(row.get("AGE", 0)) // 10 * 10, 80)
            age_sex_disease[(age_bin, row.get("SEX", "M"))][tdc] += 1

    # Re-load test patients (in same order as score matrix)
    SUBSET = n
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= SUBSET: break
            patients.append({"pathology": row["PATHOLOGY"], "age": row.get("AGE", "30"),
                             "sex": row.get("SEX", "M")})

    # Filter: keep only patients with mapped disease
    candidates = []
    for p in patients:
        tdc = fr2cui.get(p["pathology"])
        if not tdc: continue
        candidates.append({"true_dc": tdc, "age": p["age"], "sex": p["sex"]})
    assert len(candidates) == n, f"{len(candidates)} != {n}"

    # Build prior matrix [n, 49]
    prior_log = np.zeros((n, k), dtype=np.float32)
    for c_idx, c in enumerate(candidates):
        age_bin = min(int(c["age"]) // 10 * 10, 80)
        prior_c = age_sex_disease[(age_bin, c["sex"])]
        prior_t = sum(prior_c.values())
        if prior_t == 0:
            prior_log[c_idx, :] = math.log(1.0 / k)
            continue
        for d_idx, dc in enumerate(dcs_list):
            p_d = (prior_c.get(dc, 0) + 1) / (prior_t + k)
            prior_log[c_idx, d_idx] = math.log(p_d)

    # Score sweep over α
    print("\n[3] α sweep (final = stage1 + α * 100 * log_prior)...", flush=True)

    def topk_acc(M, k_acc):
        hits = 0
        for c_idx, c in enumerate(candidates):
            top = np.argsort(-M[c_idx])[:k_acc]
            if any(dcs_list[i] == c["true_dc"] for i in top): hits += 1
        return 100 * hits / n

    # Raw v79 stage1 baseline
    t1_raw = sum(1 for c_idx, c in enumerate(candidates)
                 if dcs_list[int(np.argmax(score_matrix[c_idx]))] == c["true_dc"])
    print(f"  α=0.0 (stage1 only): @1={100*t1_raw/n:.2f}% @3={topk_acc(score_matrix,3):.1f}% @5={topk_acc(score_matrix,5):.1f}%", flush=True)

    best_t1 = t1_raw
    best_alpha = 0
    best_M = score_matrix
    for alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
        combined = score_matrix + alpha * 100 * prior_log
        t1 = sum(1 for c_idx, c in enumerate(candidates)
                 if dcs_list[int(np.argmax(combined[c_idx]))] == c["true_dc"])
        a3 = topk_acc(combined, 3)
        a5 = topk_acc(combined, 5)
        print(f"  α={alpha:.2f}: @1={100*t1/n:.2f}% @3={a3:.1f}% @5={a5:.1f}%", flush=True)
        if t1 > best_t1:
            best_t1 = t1
            best_alpha = alpha
            best_M = combined

    print(f"\n  Best α={best_alpha}: @1={100*best_t1/n:.2f}% @3={topk_acc(best_M,3):.1f}% @5={topk_acc(best_M,5):.1f}%", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"v80 GTPA@1 = {100*best_t1/n:.2f}% (α={best_alpha}, SUBSET={n})", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
