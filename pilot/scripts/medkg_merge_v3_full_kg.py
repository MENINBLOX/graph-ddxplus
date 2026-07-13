#!/usr/bin/env python3
"""Merge FULL v3 PubMed IE edges into v42_full_universal KG.

Start from v39 baseline (universal-only, no benchmark priority contamination).
Add 1.8M CUI-mapped categorized edges from full PubMed IE.

Output: /mnt/medkg/kg/onlykg_graph_v42_full_universal.pkl
"""
from __future__ import annotations
import json, pickle, math
from pathlib import Path
from collections import defaultdict

IN_KG = "/mnt/medkg/kg/onlykg_graph_v39_history.pkl"
IN_EDGES = "/mnt/medkg/processed/edges_v3_full_pubmed_cui.jsonl"
OUT_KG = "/mnt/medkg/kg/onlykg_graph_v42_full_universal.pkl"


def main():
    print(f"Loading STRICT baseline: {IN_KG}", flush=True)
    G = pickle.load(open(IN_KG, "rb"))
    n0 = G.number_of_nodes(); e0 = G.number_of_edges()
    print(f"  v39: {n0:,} nodes, {e0:,} edges", flush=True)

    print(f"Reading {IN_EDGES}...", flush=True)
    triples = defaultdict(lambda: defaultdict(int))
    n_in = 0
    for line in open(IN_EDGES):
        e = json.loads(line)
        d = e["umls_cui"]; ec = e["evidence_cui"]; cat = e["category"]
        if d == ec: continue
        triples[(d, ec)][cat] += 1
        n_in += 1
        if n_in % 200_000 == 0:
            print(f"  read {n_in:,}, unique pairs={len(triples):,}", flush=True)
    print(f"  total: {n_in:,} edges → {len(triples):,} unique (d,e) pairs", flush=True)

    print("Adding to KG...", flush=True)
    n_new_nodes = 0; n_new_edges = 0
    cat_counts = defaultdict(int)
    for (d, e), cats in triples.items():
        for c in (d, e):
            if c not in G:
                G.add_node(c); n_new_nodes += 1
        primary = max(cats.items(), key=lambda x: x[1])[0]
        weight = math.log(1 + sum(cats.values())) * 0.6
        G.add_edge(d, e, etype="HAS_PHENOTYPE", weight=weight,
                   freq=sum(cats.values()), category=primary,
                   categories_all=dict(cats),
                   source="ie_v3_full_pubmed",
                   selection_origin="universal_full")
        n_new_edges += 1
        cat_counts[primary] += 1

    print(f"  after: {G.number_of_nodes():,} nodes (+{n_new_nodes}), "
          f"{G.number_of_edges():,} edges (+{n_new_edges})", flush=True)
    print(f"  by category: {dict(cat_counts)}", flush=True)

    print(f"Saving: {OUT_KG}", flush=True)
    Path(OUT_KG).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_KG, "wb") as f:
        pickle.dump(G, f)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
