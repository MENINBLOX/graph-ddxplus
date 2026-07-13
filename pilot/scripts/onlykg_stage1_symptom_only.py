#!/usr/bin/env python3
"""only-KG Stage 1 with symptom-only evidence (filter context CUIs).

Evidence CUIs are classified by UMLS Semantic Type. Only SYMPTOM-class CUIs
(T184/T033/T046/T047/T048/T191/T037/T039/T067) are used for graph traversal.
Context CUIs (ANATOMY/GEO_TEMP/DEMOGRAPHIC/OTHER) are discarded.

Rationale: context CUIs are questionnaire metadata absent from academic literature
co-occurrence patterns. This filter is benchmark-AGNOSTIC (only uses UMLS structure).

Compares against v2 multi-hop (20.46%) — measures the symptom-only ceiling.
"""
from __future__ import annotations
import sys, json, csv, ast, math, time, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

import networkx as nx

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v2.pkl"
EVIDENCE_CUI = MEDKG_ROOT / "kg" / "ddxplus_evidence_cuis.json"
SEM_CLASS = MEDKG_ROOT / "kg" / "evidence_cui_semantic_class.json"


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
    ap.add_argument("--hop2_decay", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--filter", default="SYMPTOM",
                    help="comma-separated categories to KEEP (default SYMPTOM)")
    args = ap.parse_args()

    keep_cats = set(args.filter.split(","))
    print(f"Filter: keep CUIs with category in {keep_cats}")

    print(f"Loading v2 graph...")
    with GRAPH.open("rb") as f:
        G = pickle.load(f)
    print(f"  Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")

    ev_cuis = json.load(open(EVIDENCE_CUI))
    sem_class = json.load(open(SEM_CLASS))

    # Filter ev_cuis to keep only SYMPTOM CUIs per evidence
    filtered_ev_cuis = {}
    n_kept_cuis = 0; n_dropped_cuis = 0
    for ev, cs in ev_cuis.items():
        kept = [c for c in cs if sem_class.get(c, {}).get("category") in keep_cats]
        filtered_ev_cuis[ev] = kept
        n_kept_cuis += len(kept)
        n_dropped_cuis += len(cs) - len(kept)
    print(f"  Evidence CUI filter: kept {n_kept_cuis}, dropped {n_dropped_cuis}")
    n_ev_with_kept = sum(1 for v in filtered_ev_cuis.values() if v)
    print(f"  Evidences with ≥1 kept CUI: {n_ev_with_kept}/{len(ev_cuis)}")

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))

    print("Pre-computing extended phenotypes...")
    extended = build_extended_phenotypes(G, dcs_list, args.hop2_decay)

    print(f"\nEvaluating (filter={keep_cats}, hop2_decay={args.hop2_decay})...")
    t0 = time.time()
    n = 0; c1=c3=c5=c10=0; rr_sum=0
    n_empty = 0
    fail_per_d = Counter(); total_per_d = Counter()
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            patient_cuis = set()
            for ev in evs:
                base = ev.split("_@_")[0]
                patient_cuis.update(filtered_ev_cuis.get(base, []))
            if not patient_cuis:
                n_empty += 1
                # If no symptom CUIs, fall back to uniform random — keep accounting fair
                ranked = dcs_list[:]
            else:
                scores = {}
                for d in dcs_list:
                    ext = extended.get(d, {})
                    s = sum(ext.get(p, 0) for p in patient_cuis)
                    norm = math.sqrt(len(ext)) if ext else 1
                    scores[d] = s / max(norm, 1)
                ranked = sorted(dcs_list, key=lambda d: -scores.get(d, 0))
            true_name = cui2name.get(true_cui, "?")
            n += 1; total_per_d[true_name] += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            else: fail_per_d[true_name] += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank
            if n % 5000 == 0:
                print(f"  {n}/{args.n} @1={100*c1/n:.2f}% @5={100*c5/n:.2f}% MRR={rr_sum/n:.3f} ({time.time()-t0:.0f}s)")

    print(f"\n=== only-KG Stage 1 (symptom-only filter={keep_cats}) ===")
    print(f"  GTPA@1  = {100*c1/n:.2f}%")
    print(f"  GTPA@3  = {100*c3/n:.2f}%")
    print(f"  GTPA@5  = {100*c5/n:.2f}%")
    print(f"  GTPA@10 = {100*c10/n:.2f}%")
    print(f"  MRR     = {rr_sum/n:.4f}")
    print(f"  n       = {n} (empty after filter: {n_empty})")

    print("\nTop failures:")
    for d, fc_ in fail_per_d.most_common(10):
        tot = total_per_d[d]
        print(f"  {d:35s}  {fc_}/{tot} ({100*fc_/tot:.1f}%)")


if __name__ == "__main__":
    main()
