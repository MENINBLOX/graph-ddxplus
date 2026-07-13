#!/usr/bin/env python3
"""KG topology inspection — coverage 이외의 quality checks.

원칙 1 보강:
1. Coverage = vocab 포함 여부 (이전)
2. **Connectivity = 각 evidence가 얼마나 다양한 disease에 연결되는지** (이번)

체크 항목:
- Disease 노드 degree 분포 (per disease evidence 수)
- Evidence 노드 degree 분포 (이 evidence가 몇 개 disease에 등장)
- Disconnected / singleton evidence (1 disease만 연결)
- Hub evidence (>N diseases)
- Coefficient of variation per benchmark evidence set
- IDF 분포 + skew

해석:
- Evidence가 단일 disease 연결 → 희귀병 specific (정상) or IE 부족 (문제)
- Hub evidence (모든 disease) → 비특이 (Pain, Fever) — 정상
- 분포 균형 (적당한 mid-degree 우세) = 건강한 KG
"""
from __future__ import annotations
import json, math, pickle, argparse, sys
from pathlib import Path
from collections import defaultdict, Counter
import statistics

PR_UNIVERSE = "pilot/data/pr_universe.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))
    allowed = {"patient_reportable", "history", "demographic"}

    # Collect HAS_PHENOTYPE edges
    disease_evs = defaultdict(set)  # disease → {ev cuis}
    ev_diseases = defaultdict(set)  # ev → {disease cuis}
    n_edges = 0
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") != "HAS_PHENOTYPE": continue
        cat = ed.get("category")
        if cat is None:
            if v not in pr: continue
        elif cat not in allowed: continue
        disease_evs[u].add(v)
        ev_diseases[v].add(u)
        n_edges += 1

    print(f"=== KG topology inspection: {args.graph} ===")
    print(f"HAS_PHENOTYPE edges (lay mode): {n_edges:,}")
    print(f"Unique diseases with profile: {len(disease_evs):,}")
    print(f"Unique evidence CUIs (vocab): {len(ev_diseases):,}\n")

    # Disease degree distribution
    print("--- Disease degree (evidences per disease) ---")
    dis_deg = [len(s) for s in disease_evs.values()]
    print(f"  mean={statistics.mean(dis_deg):.1f}  median={statistics.median(dis_deg):.0f}  "
          f"min={min(dis_deg)}  max={max(dis_deg)}  stdev={statistics.stdev(dis_deg):.1f}")
    buckets = [0, 10, 30, 60, 100, 200, 500, 999999]
    for i in range(len(buckets)-1):
        lo, hi = buckets[i], buckets[i+1]
        n = sum(1 for d in dis_deg if lo <= d < hi)
        hi_str = f"<{hi}" if hi < 999999 else "≥500"
        print(f"  [{lo:>3}-{hi_str:>5}): {n:>5} disease ({100*n/len(dis_deg):.1f}%)")

    # Evidence degree distribution
    print("\n--- Evidence degree (diseases per evidence) ---")
    ev_deg = [len(s) for s in ev_diseases.values()]
    print(f"  mean={statistics.mean(ev_deg):.1f}  median={statistics.median(ev_deg):.0f}  "
          f"min={min(ev_deg)}  max={max(ev_deg)}  stdev={statistics.stdev(ev_deg):.1f}")
    buckets_e = [1, 2, 3, 5, 10, 30, 100, 300, 999999]
    for i in range(len(buckets_e)-1):
        lo, hi = buckets_e[i], buckets_e[i+1]
        n = sum(1 for d in ev_deg if lo <= d < hi)
        hi_str = f"<{hi}" if hi < 999999 else "≥300"
        print(f"  [{lo:>3}-{hi_str:>5}): {n:>5} evidence ({100*n/len(ev_deg):.1f}%)")

    # Singletons (evidence connected to only 1 disease)
    singletons = [e for e, s in ev_diseases.items() if len(s) == 1]
    print(f"\n--- Singleton evidence (degree=1) ---")
    print(f"  Total: {len(singletons):,} ({100*len(singletons)/len(ev_diseases):.1f}% of all evidence)")
    print(f"  Interpretation: rare disease-specific OR IE incompleteness (cannot tell from KG alone)")

    # Hub evidence (overly generic)
    hubs = sorted(ev_diseases.items(), key=lambda x: -len(x[1]))[:20]
    print(f"\n--- Top-20 hub evidence (potential generic CUIs) ---")
    for ecui, dis_set in hubs:
        # Try to get name
        nname = G.nodes[ecui].get("name", ecui) if ecui in G else ecui
        print(f"  {ecui} ({nname[:60]}): in {len(dis_set):,} diseases")

    # Per-benchmark evidence connectivity
    print(f"\n=== Per-benchmark evidence connectivity ===")
    # SymCat
    sym_map = json.load(open("data/symcat/symptom_umls_mapping.json"))["mapping"]
    sym_cuis = set(v["umls_cui"] for v in sym_map.values() if v.get("umls_cui"))
    # DDXPlus
    value_cuis = json.load(open("/mnt/medkg/kg/ddxplus_evidence_value_cuis.json"))
    ddx_cuis = set()
    for m in value_cuis.values():
        for k, v in m.items():
            if isinstance(v, list): ddx_cuis.update(v)
    # RareBench HPO
    try:
        hpo_map = json.load(open("data/rarebench/hpo_umls_mapping.json"))["mapping"]
        rb_cuis = set(v["umls_cui"] for v in hpo_map.values() if v.get("umls_cui"))
    except: rb_cuis = set()

    for name, cuis in [("DDXPlus", ddx_cuis), ("SymCat", sym_cuis), ("RareBench(HPO)", rb_cuis)]:
        in_kg = cuis & set(ev_diseases.keys())
        degs = [len(ev_diseases[c]) for c in in_kg]
        if not degs:
            print(f"\n  {name}: no evidence in KG"); continue
        ds = sum(1 for d in degs if d == 1)
        hub_n = sum(1 for d in degs if d > 100)
        print(f"\n  {name}: {len(in_kg)}/{len(cuis)} CUI in KG ({100*len(in_kg)/max(len(cuis),1):.1f}%)")
        print(f"    degree: mean={statistics.mean(degs):.1f}  median={statistics.median(degs):.0f}  "
              f"min={min(degs)}  max={max(degs)}")
        print(f"    singletons (deg=1): {ds} ({100*ds/len(degs):.1f}%) — "
              f"검증 필요 (희귀 vs IE 부족)")
        print(f"    hubs (deg>100):    {hub_n} ({100*hub_n/len(degs):.1f}%)")

    # IDF dist
    print(f"\n=== IDF distribution (lay-mode profile) ===")
    N = len(disease_evs)
    df = Counter()
    for e, s in ev_diseases.items(): df[e] = len(s)
    idfs = [math.log((N+1)/(d+1))+1.0 for d in df.values()]
    print(f"  IDF: min={min(idfs):.2f}  median={statistics.median(idfs):.2f}  "
          f"mean={statistics.mean(idfs):.2f}  max={max(idfs):.2f}")
    print(f"  Low-IDF (<2.0, generic): {sum(1 for x in idfs if x<2.0):,}")
    print(f"  High-IDF (>6.0, specific): {sum(1 for x in idfs if x>6.0):,}")

    # Connectivity warning
    print(f"\n=== KG quality flags ===")
    sing_frac = len(singletons) / len(ev_diseases)
    if sing_frac > 0.5:
        print(f"  ⚠️  SINGLETON-DOMINANT: {100*sing_frac:.1f}% evidence connected to only 1 disease")
        print(f"     → IE incompleteness 가능성 — 추가 IE source 필요")
    elif sing_frac > 0.3:
        print(f"  ⚠️  many singletons: {100*sing_frac:.1f}% — partial concern")
    else:
        print(f"  ✓ singleton fraction OK: {100*sing_frac:.1f}%")

    isolated_dis = sum(1 for s in disease_evs.values() if len(s) == 0)
    if isolated_dis > 0:
        print(f"  ⚠️  {isolated_dis} diseases have NO evidence")
    poor_dis = sum(1 for s in disease_evs.values() if 0 < len(s) <= 5)
    print(f"  {poor_dis} diseases with ≤5 evidences (poor profile)")


if __name__ == "__main__":
    main()
