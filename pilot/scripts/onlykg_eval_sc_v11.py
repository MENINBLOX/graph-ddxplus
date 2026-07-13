#!/usr/bin/env python3
"""only-KG Stage 1 evaluation on SymCat (cross-benchmark).

SymCat provides disease → [(symptom, freq%)] from CDC/Mayo data. We simulate
patients per disease using inclusion probability. Patient CUIs come from
SymCat's symptom_umls_mapping.json. Run graph traversal on v2 graph.

This measures cross-benchmark consistency of the only-KG architecture.
"""
from __future__ import annotations
import sys, json, math, time, random, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

import networkx as nx

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v11.pkl"
SEM_CLASS_PATH = MEDKG_ROOT / "kg" / "evidence_cui_semantic_class.json"

random.seed(42)


def build_extended_phenotypes(G, disease_cuis, hop2_decay=0.5):
    extended = {}
    for d in disease_cuis:
        if d not in G:
            extended[d] = {}; continue
        phen_w = {}
        for _, p, edata in G.out_edges(d, data=True):
            if edata.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + edata.get("weight", 0)
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, edata2 in G.out_edges(p_direct, data=True):
                if edata2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + hop2_decay * dw * edata2.get("weight", 0)
        extended[d] = phen_w
    return extended


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients_per_disease", type=int, default=50)
    ap.add_argument("--hop2_decay", type=float, default=0.5)
    ap.add_argument("--filter_categories", default="ALL",
                    help="ALL or comma-list of categories to keep (SYMPTOM,ANATOMY,...)")
    args = ap.parse_args()

    print("Loading v2 graph + SymCat data...")
    with GRAPH.open("rb") as f:
        G = pickle.load(f)
    sc = json.load(open("data/symcat/symcat_parsed.json"))
    sym_umls = json.load(open("data/symcat/symptom_umls_mapping.json"))["mapping"]
    dis_umls = json.load(open("data/symcat/disease_umls_mapping.json"))["mapping"]

    # Diseases with CUI and in graph
    candidate_diseases = {}
    for dn, info in dis_umls.items():
        cui = info.get("umls_cui")
        if not cui: continue
        if cui in G:
            candidate_diseases[dn] = cui
    dcs_list = sorted(set(candidate_diseases.values()))
    print(f"  SymCat diseases with CUI in v2 KG: {len(candidate_diseases)} (unique CUIs: {len(dcs_list)})")

    # Symptom name → CUI
    sym2cui = {}
    for sn, info in sym_umls.items():
        c = info.get("umls_cui")
        if c: sym2cui[sn] = c

    # Disease → simulated patient CUI sets
    print("Simulating patients...")
    pairs = sc["disease_symptom_pairs"]
    patients = []  # (true_disease_cui, set_of_patient_cuis)
    diseases_used = set()
    for dn, sym_list in pairs.items():
        if dn not in candidate_diseases: continue
        true_cui = candidate_diseases[dn]
        if not sym_list: continue
        sym_cuis = [(sym2cui.get(s), f) for s, f in sym_list if sym2cui.get(s)]
        if not sym_cuis: continue
        diseases_used.add(true_cui)
        for _ in range(args.patients_per_disease):
            pcuis = set()
            for cui, freq in sym_cuis:
                if random.random() * 100 < freq:
                    pcuis.add(cui)
            if pcuis:
                patients.append((true_cui, pcuis))
    print(f"  Simulated patients: {len(patients)}  (diseases: {len(diseases_used)})")

    # Optional category filter
    if args.filter_categories != "ALL":
        keep = set(args.filter_categories.split(","))
        sem_class = json.load(open(SEM_CLASS_PATH))
        print(f"  Filter categories: {keep} (DDXPlus-derived classification)")
        filtered = []
        for tc, pcuis in patients:
            pcs = {c for c in pcuis if sem_class.get(c, {}).get("category") in keep}
            if pcs: filtered.append((tc, pcs))
        patients = filtered
        print(f"  After filter: {len(patients)} patients with ≥1 CUI")

    print("Pre-computing extended phenotypes...")
    extended = build_extended_phenotypes(G, dcs_list, args.hop2_decay)
    print(f"  Computed for {len(dcs_list)} diseases")

    print(f"\nEvaluating only-KG Stage 1 on SymCat...")
    t0 = time.time()
    n = 0; c1=c3=c5=c10=0; rr_sum=0
    for true_cui, pcuis in patients:
        scores = {}
        for d in dcs_list:
            ext = extended.get(d, {})
            s = sum(ext.get(p, 0) for p in pcuis)
            norm = math.sqrt(len(ext)) if ext else 1
            scores[d] = s / max(norm, 1)
        ranked = sorted(dcs_list, key=lambda d: -scores.get(d, 0))
        n += 1
        try: rank = ranked.index(true_cui) + 1
        except: rank = len(dcs_list)
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        rr_sum += 1.0 / rank

    print(f"\n=== only-KG Stage 1 on SymCat ({n} patients, {len(diseases_used)} diseases) ===")
    print(f"  GTPA@1  = {100*c1/n:.2f}%")
    print(f"  GTPA@3  = {100*c3/n:.2f}%")
    print(f"  GTPA@5  = {100*c5/n:.2f}%")
    print(f"  GTPA@10 = {100*c10/n:.2f}%")
    print(f"  MRR     = {rr_sum/n:.4f}")
    print(f"  Filter:    {args.filter_categories}")
    print(f"  hop2_decay={args.hop2_decay}")


if __name__ == "__main__":
    main()
