#!/usr/bin/env python3
"""KG 노이즈 제거: Dunning's G² + Benjamini-Hochberg FDR.

학술 표준:
  - Dunning (1993): Log-likelihood ratio test
  - Benjamini & Hochberg (1995): FDR for multiple testing

각 disease-symptom 쌍에 대해 G² 통계량 계산 → BH-FDR로 q-value < 0.05 필터.
"""
from __future__ import annotations
import json, math
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

INPUT_KG = Path("pilot/results/kg_v3_cache.json")
OUTPUT_KG = Path("pilot/results/kg_v3_statistical.json")
UMLS_DIR = Path("data/umls_extracted")


def g2_test(a, b, c, d):
    """Dunning's G² (log-likelihood ratio).

    2x2 contingency:
                    Has symptom S | No symptom S
        Disease D       a              c
        NOT Disease D   b              d
    """
    n = a + b + c + d
    if n == 0: return 0
    # Expected counts under independence
    e_a = (a + b) * (a + c) / n
    e_b = (a + b) * (b + d) / n
    e_c = (c + d) * (a + c) / n
    e_d = (c + d) * (b + d) / n

    g2 = 0
    for obs, exp in [(a, e_a), (b, e_b), (c, e_c), (d, e_d)]:
        if obs > 0 and exp > 0:
            g2 += 2 * obs * math.log(obs / exp)
    return g2


def chi2_pvalue(g2, df=1):
    """Approximate p-value from G² (chi-square distribution)."""
    # Survival function of chi-square with 1 dof
    # P(X > g2) = erfc(sqrt(g2/2))
    return math.erfc(math.sqrt(g2 / 2))


def bh_fdr(pvalues, alpha=0.05):
    """Benjamini-Hochberg FDR correction.
    Returns array of q-values."""
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_p = np.array(pvalues)[sorted_idx]
    qvalues = np.zeros(n)
    # BH adjustment
    for i in range(n - 1, -1, -1):
        rank = i + 1
        if i == n - 1:
            qvalues[sorted_idx[i]] = sorted_p[i]
        else:
            qvalues[sorted_idx[i]] = min(sorted_p[i] * n / rank, qvalues[sorted_idx[i+1]])
    return qvalues


def main():
    print("=" * 80, flush=True)
    print("KG 통계적 노이즈 제거 (Dunning G² + BH FDR)", flush=True)
    print("=" * 80, flush=True)

    print("\n[1] 로드...", flush=True)
    cp = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG" and p[2] == "P" and p[0] not in cp:
                cp[p[0]] = p[14].strip()

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open(INPUT_KG) as f: cache = json.load(f)

    pc = Counter()
    for k, v in cache["pair_counts"]: pc[tuple(k)] = v

    dcs = set()
    cui2name = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]
        dcs.add(dc); cui2name[dc] = dn

    # 각 질환의 총 LLM "yes" count, 각 증상의 총 count
    disease_total = Counter()  # C(D, ·) for each disease
    symptom_total = Counter()  # C(·, S) for each symptom
    grand_total = 0
    pair_data = {}  # (D, S) -> count

    for (a, b), cnt in pc.items():
        # Disease 식별
        if a in dcs and b not in dcs:
            d, s = a, b
        elif b in dcs and a not in dcs:
            d, s = b, a
        else:
            continue
        pair_data[(d, s)] = cnt
        disease_total[d] += cnt
        symptom_total[s] += cnt
        grand_total += cnt

    print(f"  쌍: {len(pair_data):,}, total count: {grand_total:,}", flush=True)
    print(f"  질환: {len(disease_total)}, 증상: {len(symptom_total):,}", flush=True)

    # G² 통계 계산
    print("\n[2] G² 계산...", flush=True)
    results = []
    for (d, s), c_ds in pair_data.items():
        # 2x2 table
        a = c_ds                                  # D + S
        b = symptom_total[s] - c_ds               # NOT-D + S
        c = disease_total[d] - c_ds               # D + NOT-S
        dd = grand_total - a - b - c              # NOT-D + NOT-S
        if a == 0 or dd <= 0: continue

        g2 = g2_test(a, b, c, dd)

        # observed/expected ratio
        expected = (disease_total[d] * symptom_total[s]) / grand_total if grand_total > 0 else 0
        oe_ratio = a / expected if expected > 0 else 0

        # 양의 연관성만 (a > expected)
        if oe_ratio < 1:
            g2 = -g2  # 음의 연관 (NOT 증상)

        results.append({"d": d, "s": s, "count": c_ds, "g2": g2, "oe": oe_ratio})

    print(f"  계산 완료: {len(results):,}", flush=True)

    # G² 분포
    g2_values = [r["g2"] for r in results if r["g2"] > 0]
    print(f"\n  G² 분포 (양의 연관만):", flush=True)
    print(f"    >  3.84 (p<0.05): {sum(1 for g in g2_values if g > 3.84):,}", flush=True)
    print(f"    >  6.63 (p<0.01): {sum(1 for g in g2_values if g > 6.63):,}", flush=True)
    print(f"    > 10.83 (p<0.001): {sum(1 for g in g2_values if g > 10.83):,}", flush=True)

    # p-values + BH FDR
    print("\n[3] BH-FDR 다중검정 보정...", flush=True)
    pos_results = [r for r in results if r["g2"] > 0]
    pvalues = [chi2_pvalue(r["g2"]) for r in pos_results]
    qvalues = bh_fdr(pvalues, alpha=0.05)
    for r, q in zip(pos_results, qvalues):
        r["q"] = q

    # Filter q < 0.05
    significant = [r for r in pos_results if r["q"] < 0.05]
    print(f"  q < 0.05: {len(significant):,} / {len(pos_results):,} ({100*len(significant)/len(pos_results):.0f}%)", flush=True)
    print(f"  q < 0.01: {sum(1 for r in pos_results if r['q'] < 0.01):,}", flush=True)

    # 저장 (q < 0.05인 쌍만)
    pair_counts_filtered = Counter()
    for r in significant:
        pair_counts_filtered[tuple(sorted([r["d"], r["s"]]))] = r["count"]

    save_data = {
        "pair_counts": [[list(k), v] for k, v in pair_counts_filtered.most_common()],
        "diseases": cache.get("diseases", {}),
        "stats": {
            "original_pairs": len(pair_data),
            "filtered_pairs": len(significant),
            "method": "Dunning G² + BH-FDR (q<0.05)",
        },
    }
    with open(OUTPUT_KG, "w") as f: json.dump(save_data, f)
    print(f"\n  저장: {OUTPUT_KG}", flush=True)

    # 샘플: 가장 강한 연관 (높은 G²) vs 약한 연관
    print("\n[4] 샘플:", flush=True)
    pos_results.sort(key=lambda x: -x["g2"])
    print("  Top 10 G²:", flush=True)
    for r in pos_results[:10]:
        dn = cui2name.get(r["d"], r["d"])
        sn = cp.get(r["s"], r["s"])
        print(f"    G²={r['g2']:>6.0f}, q={r['q']:.2e}, count={r['count']:>4} | {dn[:30]:<30} - {sn[:35]}", flush=True)

    # 가장 약한 (q 큰 것 중 통과한 것)
    significant.sort(key=lambda x: -x["q"])
    print("\n  Borderline q-values (q ~ 0.05):", flush=True)
    for r in significant[:5]:
        dn = cui2name.get(r["d"], r["d"])
        sn = cp.get(r["s"], r["s"])
        print(f"    G²={r['g2']:>6.1f}, q={r['q']:.3f}, count={r['count']:>4} | {dn[:30]:<30} - {sn[:35]}", flush=True)


if __name__ == "__main__":
    main()
