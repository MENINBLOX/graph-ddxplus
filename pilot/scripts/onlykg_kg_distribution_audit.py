#!/usr/bin/env python3
"""KG node distribution audit + data source review."""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v13.pkl"


def main():
    G = pickle.load(open(GRAPH, "rb"))
    print("=" * 70)
    print("v13 KG distribution audit")
    print("=" * 70)
    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")

    # 1. Node type distribution
    print("\n[1] Node type distribution:")
    n_types = Counter()
    n_phen_by_source = Counter()
    for n, attrs in G.nodes(data=True):
        n_types[attrs.get("ntype", "?")] += 1
        if attrs.get("ntype") == "Phenotype":
            n_phen_by_source[attrs.get("source", "?")] += 1
    for t, c in n_types.most_common():
        print(f"  {t}: {c:,}")
    print(f"\n  Phenotype source distribution:")
    for s, c in n_phen_by_source.most_common():
        print(f"    {s}: {c:,}")

    # 2. Edge type distribution
    print("\n[2] Edge type distribution:")
    e_types = Counter()
    for u, v, e in G.edges(data=True):
        e_types[e.get("etype", "?")] += 1
    for t, c in e_types.most_common():
        print(f"  {t}: {c:,}")

    # 3. Disease coverage stats
    print("\n[3] Disease nodes — phenotype distribution:")
    disease_phen_counts = []
    for n, attrs in G.nodes(data=True):
        if attrs.get("ntype") == "Disease":
            n_phen = sum(1 for _,_,e in G.out_edges(n, data=True) if e.get("etype")=="HAS_PHENOTYPE")
            disease_phen_counts.append(n_phen)
    if disease_phen_counts:
        disease_phen_counts.sort()
        n = len(disease_phen_counts)
        print(f"  Total disease nodes: {n}")
        print(f"  Min phens: {disease_phen_counts[0]}")
        print(f"  p25: {disease_phen_counts[n//4]}")
        print(f"  Median: {disease_phen_counts[n//2]}")
        print(f"  p75: {disease_phen_counts[3*n//4]}")
        print(f"  Max: {disease_phen_counts[-1]}")
        n_zero = sum(1 for c in disease_phen_counts if c == 0)
        n_sparse = sum(1 for c in disease_phen_counts if c < 10)
        n_well = sum(1 for c in disease_phen_counts if c >= 50)
        print(f"  Zero-phen diseases: {n_zero:,}")
        print(f"  Sparse (<10 phens): {n_sparse:,}")
        print(f"  Well-covered (≥50 phens): {n_well:,}")

    # 4. Phenotype connectivity (indegree)
    print("\n[4] Phenotype indegree distribution:")
    phen_in = Counter()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            phen_in[v] += 1
    indeg_buckets = Counter()
    for v in phen_in.values():
        if v == 1: indeg_buckets["1 (singleton)"] += 1
        elif v <= 5: indeg_buckets["2-5"] += 1
        elif v <= 20: indeg_buckets["6-20"] += 1
        elif v <= 100: indeg_buckets["21-100"] += 1
        else: indeg_buckets[">100"] += 1
    for b, c in indeg_buckets.most_common():
        print(f"  {b}: {c:,}")
    print(f"  Total phen nodes: {len(phen_in):,}")

    # 5. DDXPlus 49 specifically
    print("\n[5] DDXPlus 49 disease coverage:")
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    ddx_cuis = {info["cui"]: dn for dn, info in icd.items() if "cui" in info}
    ddx_phens = []
    for cui, dn in ddx_cuis.items():
        if cui in G:
            n_phen = sum(1 for _,_,e in G.out_edges(cui, data=True) if e.get("etype")=="HAS_PHENOTYPE")
            ddx_phens.append((dn, n_phen))
    ddx_phens.sort(key=lambda x: x[1])
    print(f"  Total: {len(ddx_phens)}")
    print(f"  Min: {ddx_phens[0]}")
    print(f"  Max: {ddx_phens[-1]}")
    print(f"  Median: {ddx_phens[len(ddx_phens)//2]}")
    print(f"\n  Bottom 5 sparse:")
    for dn, n in ddx_phens[:5]:
        print(f"    {dn:40s} {n} phens")
    print(f"\n  Top 5 well-covered:")
    for dn, n in ddx_phens[-5:]:
        print(f"    {dn:40s} {n} phens")


if __name__ == "__main__":
    main()
