#!/usr/bin/env python3
"""SymCat KG coverage audit.

Measures:
1. Disease CUI coverage: SymCat 801 disease → CUI → in KG?
2. Symptom CUI coverage: SymCat 474 symptom → CUI → in any disease profile?
3. (Disease, Symptom) pair coverage: SymCat 9162 pairs → edge exists in KG?
4. Profile-quality: per disease, how many of its SymCat symptoms are
   actually in our KG profile (HAS_PHENOTYPE edges)?
"""
from __future__ import annotations
import json, pickle, argparse, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, "pilot/scripts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    args = ap.parse_args()

    parsed = json.load(open("data/symcat/symcat_parsed_full.json"))
    sym_map = json.load(open("data/symcat/symptom_umls_mapping.json"))["mapping"]
    dis_map = json.load(open("data/symcat/disease_umls_mapping.json"))["mapping"]
    sym2cui = {n: v["umls_cui"] for n, v in sym_map.items() if v.get("umls_cui")}
    dis2cui = {n: v["umls_cui"] for n, v in dis_map.items() if v.get("umls_cui")}

    print(f"=== KG: {args.graph} ===")
    G = pickle.load(open(args.graph, "rb"))
    print(f"KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)

    pr = set(json.load(open("pilot/data/pr_universe.json")))
    allowed_cats = {"patient_reportable", "history", "demographic"}

    pairs = parsed["disease_symptom_pairs"]

    # 1. Disease coverage
    dis_total = len(pairs)
    dis_mapped = sum(1 for d in pairs if dis2cui.get(d))
    dis_in_kg = sum(1 for d in pairs if dis2cui.get(d) and dis2cui[d] in G)

    print(f"\n--- Disease coverage ---")
    print(f"  SymCat 801 → UMLS CUI mapped: {dis_mapped}/{dis_total} ({100*dis_mapped/dis_total:.1f}%)")
    print(f"  → present as node in KG: {dis_in_kg}/{dis_total} ({100*dis_in_kg/dis_total:.1f}%)")

    # 2. Symptom coverage (does any disease profile contain it?)
    all_sym_cuis = set(v for v in sym2cui.values())
    print(f"\n--- Symptom coverage ---")
    print(f"  SymCat 474 → UMLS CUI mapped: {len(sym2cui)}/{len(sym_map)} "
          f"({100*len(sym2cui)/len(sym_map):.1f}%)")

    # Symptoms appearing in any disease profile (lay mode)
    profile_evs = set()
    for d, dcui in dis2cui.items():
        if not dcui or dcui not in G: continue
        for _, p, ed in G.out_edges(dcui, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            cat = ed.get("category")
            if cat is None:
                if p not in pr: continue
            elif cat not in allowed_cats: continue
            profile_evs.add(p)
    sym_in_profile = len(all_sym_cuis & profile_evs)
    print(f"  → present in any SymCat-disease KG profile: {sym_in_profile}/{len(all_sym_cuis)} "
          f"({100*sym_in_profile/len(all_sym_cuis):.1f}%)")

    # 3. (Disease, Symptom) pair coverage
    total_pairs = 0
    pairs_in_kg = 0
    per_dis_total = defaultdict(int)
    per_dis_in_kg = defaultdict(int)
    for d, syms in pairs.items():
        dcui = dis2cui.get(d)
        if not dcui or dcui not in G:
            for s, _ in syms:
                if sym2cui.get(s): total_pairs += 1
            continue
        # Build profile_for_disease (CUIs)
        prof_cuis = set()
        for _, p, ed in G.out_edges(dcui, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            cat = ed.get("category")
            if cat is None:
                if p not in pr: continue
            elif cat not in allowed_cats: continue
            prof_cuis.add(p)
        for s, _ in syms:
            scui = sym2cui.get(s)
            if not scui: continue
            total_pairs += 1
            per_dis_total[d] += 1
            if scui in prof_cuis:
                pairs_in_kg += 1
                per_dis_in_kg[d] += 1

    print(f"\n--- (Disease, Symptom) pair coverage ---")
    print(f"  Total mapped pairs in SymCat: {total_pairs}")
    print(f"  Pairs present as KG edge (lay mode): {pairs_in_kg} ({100*pairs_in_kg/total_pairs:.2f}%)")

    # 4. Per-disease coverage histogram
    per_disease_rates = []
    for d in pairs:
        t = per_dis_total[d]
        if t == 0: continue
        rate = per_dis_in_kg[d] / t
        per_disease_rates.append(rate)
    import statistics
    print(f"\n--- Per-disease coverage stats (only diseases with mapped symptoms) ---")
    print(f"  Diseases analyzed: {len(per_disease_rates)}")
    if per_disease_rates:
        print(f"  Mean coverage: {100*statistics.mean(per_disease_rates):.2f}%")
        print(f"  Median: {100*statistics.median(per_disease_rates):.2f}%")
        # Bucket
        buckets = [0, 25, 50, 75, 100]
        for i in range(len(buckets)-1):
            lo, hi = buckets[i], buckets[i+1]
            n = sum(1 for r in per_disease_rates if 100*r >= lo and 100*r < hi)
            print(f"  [{lo:>3}-{hi:>3}%): {n} disease ({100*n/len(per_disease_rates):.1f}%)")
        # 100% bucket
        n100 = sum(1 for r in per_disease_rates if r >= 1.0)
        print(f"  [100%   ]: {n100} disease ({100*n100/len(per_disease_rates):.1f}%)")
        n0 = sum(1 for r in per_disease_rates if r == 0)
        print(f"  Zero coverage: {n0} disease ({100*n0/len(per_disease_rates):.1f}%)")


if __name__ == "__main__":
    main()
