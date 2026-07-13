#!/usr/bin/env python3
"""Sparse-matrix Personalized PageRank for only-KG."""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, time
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

import numpy as np
from scipy.sparse import csr_matrix

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--n_iter", type=int, default=15)
    ap.add_argument("--hier_weight", type=float, default=0.5)
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

    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    print(f"  {n_nodes} nodes")

    print("Building sparse transition matrix...")
    rows, cols, vals = [], [], []
    for u, v, edata in G.edges(data=True):
        et = edata.get("etype", "")
        w = edata.get("weight", 0)
        if w <= 0: continue
        if et == "HIERARCHY": w *= args.hier_weight
        elif et != "HAS_PHENOTYPE": continue
        rows.append(node_idx[u]); cols.append(node_idx[v]); vals.append(w)
        rows.append(node_idx[v]); cols.append(node_idx[u]); vals.append(w)

    # Normalize by row sum to make column-stochastic
    A = csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    print(f"  Sparse matrix nnz: {A.nnz:,}")
    # Compute row sums
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    # Inverse row sums diagonal
    from scipy.sparse import diags
    inv_d = diags(1.0 / row_sums)
    # M = inv_d @ A (each row sums to 1)
    M = inv_d @ A
    print(f"  Built transition matrix")
    # Transpose for column-stochastic (used in p ← (1-α) M^T p + α r)
    Mt = M.T.tocsr()

    # Disease indices
    d_idx = np.array([node_idx[d] for d in dcs_list if d in node_idx])

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

    def ppr(seeds_set):
        seeds = [node_idx[c] for c in seeds_set if c in node_idx]
        if not seeds: return None
        r = np.zeros(n_nodes)
        for s in seeds: r[s] = 1.0
        r /= r.sum()
        p = r.copy()
        for _ in range(args.n_iter):
            p = (1 - args.alpha) * (Mt @ p) + args.alpha * r
        return p

    print(f"\nEvaluating PPR (alpha={args.alpha}, iter={args.n_iter}, hier_w={args.hier_weight})...")
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
            if p is None:
                ranked = dcs_list[:]
            else:
                disease_scores = [(d, p[node_idx[d]] if d in node_idx else 0) for d in dcs_list]
                disease_scores.sort(key=lambda x: -x[1])
                ranked = [d for d, _ in disease_scores]
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank
            if n % 500 == 0:
                print(f"  {n}/{args.n} @1={100*c1/n:.2f}% @5={100*c5/n:.2f}% MRR={rr_sum/n:.3f} ({time.time()-t0:.0f}s)")

    print(f"\n=== Sparse PPR (alpha={args.alpha}, iter={args.n_iter}, hier_w={args.hier_weight}) ===")
    print(f"  @1={100*c1/n:.2f}%  @3={100*c3/n:.2f}%  @5={100*c5/n:.2f}%  @10={100*c10/n:.2f}%  MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
