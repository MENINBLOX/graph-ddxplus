#!/usr/bin/env python3
"""only-KG Phase 2b: Stage 1 with 2-hop traversal via hierarchy edges.

Per-disease pre-computation:
  extended_phenotypes(D) = {
    p: weight (direct or 2-hop via hierarchy)
  }

Scoring:
  score(D | patient_cuis) = Σ_p∈patient_cuis extended_phenotypes(D)[p]

Equivalent Cypher:
  MATCH (d:Disease)-[r:HAS_PHENOTYPE]->(p1:Phenotype)
  OPTIONAL MATCH (p1)-[h:HIERARCHY]->(p2:Phenotype)
  WHERE p1.cui IN $patient_cuis OR p2.cui IN $patient_cuis
  RETURN d.cui, SUM(CASE
    WHEN p1.cui IN $patient_cuis THEN r.weight
    WHEN p2.cui IN $patient_cuis THEN r.weight * h.weight * 0.5
    ELSE 0 END) AS score
  ORDER BY score DESC LIMIT $top_k
"""
from __future__ import annotations
import sys, json, csv, ast, math, time, pickle
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

import networkx as nx
import numpy as np

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v3.pkl"
EVIDENCE_CUI = MEDKG_ROOT / "kg" / "ddxplus_evidence_cuis.json"


def build_extended_phenotypes(G, disease_cuis: list, hop2_decay: float = 0.5) -> dict:
    """For each disease, precompute extended phenotype weight map.

    extended[D][p] = direct_weight(D → p) + hop2_decay * Σ_q direct_weight(D → q) * hierarchy_weight(q → p)
    """
    extended = {}
    for d in disease_cuis:
        if d not in G:
            extended[d] = {}; continue
        phen_w = {}
        # 1-hop: direct HAS_PHENOTYPE
        for _, p, edata in G.out_edges(d, data=True):
            if edata.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + edata.get("weight", 0)
        # 2-hop: via HIERARCHY from each direct phenotype
        for p_direct in list(phen_w.keys()):
            direct_w = phen_w[p_direct]
            for _, p2, edata2 in G.out_edges(p_direct, data=True):
                if edata2.get("etype") == "HIERARCHY":
                    h_w = edata2.get("weight", 0)
                    add = hop2_decay * direct_w * h_w
                    phen_w[p2] = phen_w.get(p2, 0) + add
        extended[d] = phen_w
    return extended


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop2_decay", type=float, default=0.5)
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--n", type=int, default=30000)
    args = ap.parse_args()

    print(f"Loading v2 graph from {GRAPH}...")
    with GRAPH.open("rb") as f:
        G = pickle.load(f)
    print(f"  Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")

    print("Loading patient evidence CUIs...")
    ev_cuis = json.load(open(EVIDENCE_CUI))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))

    print(f"Pre-computing extended phenotypes (hop2_decay={args.hop2_decay})...")
    t0 = time.time()
    extended = build_extended_phenotypes(G, dcs_list, args.hop2_decay)
    sizes = [len(v) for v in extended.values()]
    sizes.sort()
    print(f"  Extended phenotypes/disease: median={sizes[len(sizes)//2]}, p90={sizes[int(len(sizes)*0.9)]}, max={sizes[-1]} ({time.time()-t0:.0f}s)")

    print(f"\nEvaluating on {args.n} patients...")
    t0 = time.time()
    n = 0; c1=c3=c5=c10=0; rr_sum=0
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
                patient_cuis.update(ev_cuis.get(base, []))
            scores = {}
            for d in dcs_list:
                ext = extended.get(d, {})
                s = sum(ext.get(p, 0) for p in patient_cuis)
                if args.normalize:
                    norm = math.sqrt(len(ext)) if ext else 1
                    s = s / max(norm, 1)
                scores[d] = s
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

    print(f"\n=== only-KG Stage 1 v2 (multi-hop, hop2_decay={args.hop2_decay}) ===")
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
