#!/usr/bin/env python3
"""only-KG Phase 2c: Stage 1 with multi-hop + patient CUI hierarchy expansion + disease prior.

Enhancements over v2:
  - Patient CUIs are also expanded via 1-hop HIERARCHY (vocabulary normalization)
  - Disease prior P(D) from DDXPlus train added as log-prior
  - Score = w_prior * log P(D) + w_match * sum(extended_phenotype_weight ∩ expanded_patient_cuis)

Note: P(D) uses DDXPlus train labels — this is zero-shot ON TEST patients but uses
the population distribution from train (acceptable as it's a global distribution).
"""
from __future__ import annotations
import sys, json, csv, ast, math, time, pickle
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

import networkx as nx

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v2.pkl"
EVIDENCE_CUI = MEDKG_ROOT / "kg" / "ddxplus_evidence_cuis.json"
DISEASE_PRIOR = MEDKG_ROOT / "kg" / "disease_prior_by_cui.json"


def expand_patient_cuis(G, patient_cuis: set, hop_weight: float = 0.5) -> dict:
    """Expand patient CUIs via 1-hop hierarchy. Returns {cui: weight}."""
    expanded = {c: 1.0 for c in patient_cuis}
    for c in list(patient_cuis):
        if c not in G: continue
        for _, c2, edata in G.out_edges(c, data=True):
            if edata.get("etype") == "HIERARCHY":
                h_w = edata.get("weight", 0)
                expanded[c2] = max(expanded.get(c2, 0), hop_weight * h_w)
    return expanded


def build_extended_phenotypes(G, disease_cuis: list, hop2_decay: float = 0.7) -> dict:
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
                    h_w = edata2.get("weight", 0)
                    phen_w[p2] = phen_w.get(p2, 0) + hop2_decay * dw * h_w
        extended[d] = phen_w
    return extended


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop2_decay", type=float, default=0.7)
    ap.add_argument("--patient_hop", type=float, default=0.5)
    ap.add_argument("--w_prior", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=30000)
    args = ap.parse_args()

    print(f"Loading v2 graph + prior...")
    with GRAPH.open("rb") as f:
        G = pickle.load(f)
    ev_cuis = json.load(open(EVIDENCE_CUI))
    prior = json.load(open(DISEASE_PRIOR)) if DISEASE_PRIOR.exists() else {}

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))

    # Disease prior by CUI
    cui_prior = {}
    for fr_name, cui in fr2cui.items():
        cui_prior[cui] = prior.get(fr_name, 1.0/len(dcs_list))
    total_pr = sum(cui_prior.values()) or 1
    for c in cui_prior: cui_prior[c] /= total_pr

    print("Pre-computing extended phenotypes...")
    extended = build_extended_phenotypes(G, dcs_list, args.hop2_decay)
    print(f"  Done")

    print(f"\nEvaluating (hop2_decay={args.hop2_decay}, patient_hop={args.patient_hop}, w_prior={args.w_prior})...")
    t0 = time.time()
    n = 0; c1=c3=c5=c10=0; rr_sum=0
    fail_per_d = Counter(); total_per_d = Counter()
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            raw_pcuis = set()
            for ev in evs:
                base = ev.split("_@_")[0]
                raw_pcuis.update(ev_cuis.get(base, []))
            # Expand patient CUIs via hierarchy
            expanded_p = expand_patient_cuis(G, raw_pcuis, hop_weight=args.patient_hop)
            scores = {}
            for d in dcs_list:
                ext = extended.get(d, {})
                # Sum: for each expanded patient CUI, take min(patient_weight, disease_weight)?
                # Simpler: sum disease_weight * patient_weight where overlap
                s = 0.0
                for p_cui, p_w in expanded_p.items():
                    if p_cui in ext:
                        s += ext[p_cui] * p_w
                # Normalize by sqrt(|ext|)
                norm = math.sqrt(len(ext)) if ext else 1
                s = s / max(norm, 1)
                # Add log prior
                pr = cui_prior.get(d, 1/len(dcs_list))
                scores[d] = s + args.w_prior * math.log(pr + 1e-6)
            ranked = sorted(dcs_list, key=lambda d: -scores.get(d, -1e9))
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

    print(f"\n=== only-KG Stage 1 v3 ===")
    print(f"  GTPA@1  = {100*c1/n:.2f}%")
    print(f"  GTPA@3  = {100*c3/n:.2f}%")
    print(f"  GTPA@5  = {100*c5/n:.2f}%")
    print(f"  GTPA@10 = {100*c10/n:.2f}%")
    print(f"  MRR     = {rr_sum/n:.4f}")
    print(f"  n       = {n}")

    print("\nTop failures:")
    for d, fc_ in fail_per_d.most_common(10):
        tot = total_per_d[d]
        print(f"  {d:35s}  {fc_}/{tot} ({100*fc_/tot:.1f}%)")


if __name__ == "__main__":
    main()
