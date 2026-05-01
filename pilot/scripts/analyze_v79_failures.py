#!/usr/bin/env python3
"""v79 실패 케이스 분석.

v79_stage1.npy + 정답을 비교 → top10 안에 있지만 1위가 안 되는 케이스에서
어떤 distractor가 잘못 1위로 뽑혔는지, 정답과 distractor의 KG features 차이
분석.
"""
from __future__ import annotations
import ast, csv, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_v3_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}
GENERIC_TERMS = {'symptom', 'sign', 'pain', 'patient', 'disease', 'syndrome', 'condition'}

def main():
    score_matrix = np.load("pilot/results/v79_stage1.npy")
    n, k = score_matrix.shape
    print(f"v79 stage1 score matrix: {n} patients × {k} candidates")

    cp = {}
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG" and p[2] == "P" and p[0] not in cp:
                cp[p[0]] = p[14].strip()

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open(KG_CACHE) as f: cache = json.load(f)

    pc = Counter()
    for kk, v in cache["pair_counts"]: pc[tuple(kk)] = v

    diseases = {}; fr2cui = {}; cui2name = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]; diseases[dn] = {"cui": dc}
        fr2cui[info.get("cond-name-fr", "")] = dc; cui2name[dc] = dn
    dcs = set(d["cui"] for d in diseases.values())
    dcs_list = sorted(dcs)

    # Build disease features (same as v79)
    ds = defaultdict(dict)
    for (a, b), cnt in pc.items():
        if a in NOISE or b in NOISE: continue
        if a in dcs: ds[a][b] = cnt
        if b in dcs: ds[b][a] = cnt

    disease_features = {}
    TOP_K_FEATURES = 8
    for dc in dcs_list:
        feats = ds.get(dc, {})
        top_cuis = sorted(feats.items(), key=lambda x: -x[1])[:TOP_K_FEATURES * 3]
        names = []
        seen = set()
        for cui, cnt in top_cuis:
            n_ = cp.get(cui, cui)
            nl = n_.lower().strip()
            if not nl or nl in seen or nl in GENERIC_TERMS: continue
            if len(nl) < 3 or len(nl) > 50: continue
            seen.add(nl)
            names.append(n_)
            if len(names) >= TOP_K_FEATURES: break
        disease_features[dc] = ", ".join(names) if names else "—"

    # Re-load test patients (same order as score matrix)
    candidates = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n: break
            tdc = fr2cui.get(row["PATHOLOGY"])
            if not tdc: continue
            candidates.append({"true_dc": tdc, "patho": row["PATHOLOGY"]})

    assert len(candidates) == n

    # Confusion matrix: (true → predicted) 분석
    confusion = Counter()
    correct = Counter()
    in_top10_not_top1 = []  # (c_idx, true_dc, predicted_dc, true_rank, true_score, top1_score)

    for c_idx, c in enumerate(candidates):
        scores = score_matrix[c_idx]
        ranked = np.argsort(-scores)
        top1_dc = dcs_list[ranked[0]]
        true_dc = c["true_dc"]
        true_idx = dcs_list.index(true_dc)
        true_rank = int(np.where(ranked == true_idx)[0][0])

        if top1_dc == true_dc:
            correct[true_dc] += 1
        else:
            confusion[(true_dc, top1_dc)] += 1
            if true_rank < 10:
                in_top10_not_top1.append({
                    "c_idx": c_idx, "true_dc": true_dc, "pred_dc": top1_dc,
                    "true_rank": true_rank,
                    "true_score": float(scores[true_idx]),
                    "top1_score": float(scores[ranked[0]]),
                })

    n_correct = sum(correct.values())
    print(f"\nOverall @1: {100*n_correct/n:.2f}%")
    print(f"In top10 but not top1: {len(in_top10_not_top1)} ({100*len(in_top10_not_top1)/n:.1f}%)")

    # Per-disease accuracy
    disease_total = Counter()
    for c in candidates: disease_total[c["true_dc"]] += 1

    print("\n=== Per-disease accuracy (worst 15) ===")
    accs = [(dc, correct.get(dc, 0), disease_total[dc]) for dc in disease_total]
    accs.sort(key=lambda x: x[1] / x[2] if x[2] else 0)
    for dc, cor, tot in accs[:15]:
        print(f"  {cui2name.get(dc, dc)[:40]:42s} {cor}/{tot} = {100*cor/tot:.1f}%")

    # Top confusion pairs
    print("\n=== Top 20 confusion pairs (true → predicted) ===")
    for (t, p), c in confusion.most_common(20):
        print(f"  {c:4d}  {cui2name.get(t, t)[:30]:32s} → {cui2name.get(p, p)[:30]:32s}")
        print(f"        true KG:  {disease_features[t][:120]}")
        print(f"        pred KG:  {disease_features[p][:120]}")
        print()

    # Score gap analysis: how far behind is the true answer when missed?
    gaps = [c["top1_score"] - c["true_score"] for c in in_top10_not_top1]
    print(f"\n=== Score gap (top1 − true) for in_top10 misses ===")
    print(f"  median: {np.median(gaps):.1f}, mean: {np.mean(gaps):.1f}, max: {np.max(gaps):.1f}")
    print(f"  gap ≤ 5: {sum(1 for g in gaps if g <= 5)} ({100*sum(1 for g in gaps if g <= 5)/len(gaps):.1f}%)")
    print(f"  gap ≤ 10: {sum(1 for g in gaps if g <= 10)} ({100*sum(1 for g in gaps if g <= 10)/len(gaps):.1f}%)")

    # KG feature overlap analysis: do confused diseases share many features?
    def feat_set(dc):
        return set(f.strip().lower() for f in disease_features[dc].split(","))
    print(f"\n=== Top confused pairs: KG feature overlap ===")
    for (t, p), c in confusion.most_common(10):
        ft = feat_set(t); fp = feat_set(p)
        overlap = ft & fp
        print(f"  {cui2name.get(t,t)[:25]:27s} → {cui2name.get(p,p)[:25]:27s} cnt={c} overlap={len(overlap)}/{min(len(ft),len(fp))}")
        if overlap:
            print(f"        shared: {', '.join(sorted(overlap))[:120]}")


if __name__ == "__main__":
    main()
