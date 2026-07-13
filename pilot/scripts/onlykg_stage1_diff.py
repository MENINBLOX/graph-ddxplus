#!/usr/bin/env python3
"""only-KG Phase 2: Stage 1 differential diagnosis (graph traversal, no LLM).

Cypher-equivalent:
  MATCH (d:Disease)-[r:HAS_PHENOTYPE]->(p:Phenotype)
  WHERE p.cui IN $patient_cuis
  RETURN d.cui AS disease, SUM(r.weight) AS score, COUNT(DISTINCT p) AS n_match
  ORDER BY score DESC
  LIMIT $top_k

Scoring variants tested:
  A. Sum of edge weights (matched patient_cui → disease)
  B. Normalized by disease degree (sqrt) — avoid broad-disease bias
  C. Personalized PageRank from patient_cui nodes
  D. Multi-hop reach (2-hop via phenotype-phenotype)  [requires alias edges]

Phase 2 evaluation: DDXPlus 30K with all evidence at once (NO interactive Q&A yet).
This establishes baseline graph-based differential diagnosis accuracy.
Phase 3 will add interactive Q&A simulation on top.
"""
from __future__ import annotations
import sys, json, csv, ast, math, time, pickle
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

import networkx as nx
import numpy as np

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph.pkl"
EVIDENCE_CUI = MEDKG_ROOT / "kg" / "ddxplus_evidence_cuis.json"


def stage1_score(G, patient_cuis: set, candidates: list[str], variant: str = "A"):
    """Score each candidate disease against patient CUIs.

    variant A: sum of weighted edges
    variant B: A / sqrt(disease_degree)
    variant C: count matches × log(idf) — simpler alternative
    """
    scores = {}
    for d in candidates:
        if d not in G:
            scores[d] = 0.0; continue
        s = 0.0; n_match = 0
        for _, p, edata in G.out_edges(d, data=True):
            if p in patient_cuis:
                s += edata.get("weight", 0.0)
                n_match += 1
        if variant == "B":
            deg = G.out_degree(d) or 1
            s = s / math.sqrt(deg)
        elif variant == "C":
            s = n_match
        scores[d] = s
    return scores


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A", choices=["A","B","C"])
    ap.add_argument("--n", type=int, default=30000)
    args = ap.parse_args()

    print(f"Loading graph from {GRAPH}...")
    with GRAPH.open("rb") as f:
        G = pickle.load(f)
    print(f"  Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")

    print(f"Loading patient evidence CUIs from {EVIDENCE_CUI}...")
    ev_cuis = json.load(open(EVIDENCE_CUI))
    print(f"  {len(ev_cuis)} unique evidences, {sum(len(v) for v in ev_cuis.values())} total CUIs")

    # DDXPlus mapping
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))
    print(f"  DDXPlus candidates: {len(dcs_list)}")

    # Eval
    print(f"\nEvaluating variant {args.variant} on {args.n} patients...")
    t0 = time.time()
    n = 0
    c1 = c3 = c5 = c10 = 0
    rr_sum = 0.0
    fail_per_d = Counter(); total_per_d = Counter()
    confidence_buckets = defaultdict(lambda: [0, 0])  # bin_idx -> [correct, total]

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
            scores = stage1_score(G, patient_cuis, dcs_list, variant=args.variant)
            # Rank
            ranked = sorted(dcs_list, key=lambda d: -scores.get(d, 0))
            true_name = cui2name.get(true_cui, "?")
            n += 1; total_per_d[true_name] += 1
            try:
                rank = ranked.index(true_cui) + 1
            except ValueError:
                rank = 50
            if rank == 1: c1 += 1
            else: fail_per_d[true_name] += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0 / rank
            # Calibration: top-1 score as confidence
            top_score = scores.get(ranked[0], 0)
            # normalize to [0, 1] using max score among candidates
            max_score = max(scores.values()) or 1
            conf = top_score / max_score if max_score > 0 else 0
            bin_idx = min(int(conf * 10), 9)
            confidence_buckets[bin_idx][1] += 1
            if rank == 1:
                confidence_buckets[bin_idx][0] += 1
            if n % 5000 == 0:
                print(f"  {n}/{args.n} @1={100*c1/n:.2f}% @5={100*c5/n:.2f}% MRR={rr_sum/n:.3f} ({time.time()-t0:.0f}s)")

    print(f"\n=== only-KG Stage 1 (variant {args.variant}) Results ===")
    print(f"  GTPA@1  = {100*c1/n:.2f}%")
    print(f"  GTPA@3  = {100*c3/n:.2f}%")
    print(f"  GTPA@5  = {100*c5/n:.2f}%")
    print(f"  GTPA@10 = {100*c10/n:.2f}%")
    print(f"  MRR     = {rr_sum/n:.4f}")
    print(f"  n       = {n}")

    # ECE (expected calibration error)
    ece = 0.0
    for b, (correct, total) in confidence_buckets.items():
        if total == 0: continue
        acc = correct / total
        conf = (b + 0.5) / 10
        ece += (total / n) * abs(acc - conf)
    print(f"  ECE     = {ece:.4f}")

    print(f"\nTop failures:")
    for d, fc_ in fail_per_d.most_common(10):
        tot = total_per_d[d]
        print(f"  {d:35s}  {fc_}/{tot} ({100*fc_/tot:.1f}%)")


if __name__ == "__main__":
    main()
