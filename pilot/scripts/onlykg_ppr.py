#!/usr/bin/env python3
"""only-KG Personalized PageRank scoring.

Seed: patient evidence CUIs (uniform mass)
Diffuse: through KG edges (HAS_PHENOTYPE + HIERARCHY) with edge weights
Score disease: PPR visit probability

This uses pure graph structure — no benchmark-specific Q.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, time
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

import numpy as np

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--alpha", type=float, default=0.15, help="restart probability")
    ap.add_argument("--n_iter", type=int, default=20)
    args = ap.parse_args()

    print("Loading graph...")
    G = pickle.load(open(GRAPH, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))

    # Build undirected adjacency with weights
    print("Building adjacency...")
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    print(f"  {n_nodes} nodes")

    # Edge weight: HAS_PHENOTYPE × 1.0, HIERARCHY × 0.5
    out_w = {n: [] for n in nodes}  # n -> [(neighbor, weight)]
    for u, v, edata in G.edges(data=True):
        et = edata.get("etype", "")
        w = edata.get("weight", 0)
        if et == "HAS_PHENOTYPE":
            out_w[u].append((v, w))
            out_w[v].append((u, w))  # bidirectional
        elif et == "HIERARCHY":
            out_w[u].append((v, w * 0.5))
            out_w[v].append((u, w * 0.5))

    # Normalize
    for n, edges in out_w.items():
        tot = sum(w for _, w in edges) or 1
        out_w[n] = [(nb, w/tot) for nb, w in edges]

    def get_pcuis(evs):
        cuis = set()
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                m = value_cuis.get(base, {})
                cuis.update(m.get("_question", []))
                cuis.update(m.get(val, []))
            else:
                m = value_cuis.get(ev, {})
                cuis.update(m.get("_question", []))
        return cuis

    def ppr(seeds_set, alpha=args.alpha, n_iter=args.n_iter):
        """Simple power iteration PPR."""
        seeds = [c for c in seeds_set if c in node_idx]
        if not seeds: return {}
        # restart distribution
        r = np.zeros(n_nodes)
        for s in seeds: r[node_idx[s]] = 1.0
        r /= r.sum()
        p = r.copy()
        for _ in range(n_iter):
            new_p = np.zeros(n_nodes)
            for n, edges in out_w.items():
                if p[node_idx[n]] == 0: continue
                share = p[node_idx[n]] * (1 - alpha)
                for nb, w in edges:
                    new_p[node_idx[nb]] += share * w
            new_p += alpha * r
            p = new_p
        return p

    print(f"\nEvaluating PPR (alpha={args.alpha}, iter={args.n_iter})...")
    t0 = time.time()
    n = 0; c1=c3=c5=c10=0; rr_sum=0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis = get_pcuis(evs)
            p = ppr(pcuis)
            if len(p) == 0:
                ranked = dcs_list[:]
            else:
                ranked = sorted(dcs_list, key=lambda d: -p[node_idx.get(d, 0)] if d in node_idx else 0)
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank
            if n % 200 == 0:
                print(f"  {n}/{args.n} @1={100*c1/n:.2f}% @5={100*c5/n:.2f}% MRR={rr_sum/n:.3f} ({time.time()-t0:.0f}s)")

    print(f"\n=== PPR (alpha={args.alpha}, iter={args.n_iter}) ===")
    print(f"  @1={100*c1/n:.2f}%  @5={100*c5/n:.2f}%  @10={100*c10/n:.2f}%  MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
