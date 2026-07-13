#!/usr/bin/env python3
"""Merge IE v3 categorized edges into existing KG.

Input:
  - /mnt/medkg/kg/onlykg_graph_v39_history.pkl (existing KG)
  - /mnt/medkg/processed/edges_universal_v3_cui.jsonl (new categorized edges)

Output:
  - /mnt/medkg/kg/onlykg_graph_v40_categorized.pkl

For each new (disease_cui, evidence_cui, category) triple:
  - Add edge HAS_PHENOTYPE with weight based on frequency
  - Tag edge with 'category' attribute for eval-time filtering
  - Source = 'ie_v3_categorized'
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
from collections import defaultdict
import math
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

IN_KG = MEDKG_ROOT / "kg" / "onlykg_graph_v39_history.pkl"
IN_EDGES = MEDKG_ROOT / "processed" / "edges_universal_v3_cui.jsonl"
OUT_KG = MEDKG_ROOT / "kg" / "onlykg_graph_v40_categorized.pkl"


def main():
    print(f"Loading KG: {IN_KG}", flush=True)
    G = pickle.load(open(IN_KG, "rb"))
    n_nodes_before = G.number_of_nodes()
    n_edges_before = G.number_of_edges()
    print(f"  before: {n_nodes_before:,} nodes, {n_edges_before:,} edges", flush=True)

    # Aggregate IE v3 triples: (disease_cui, evidence_cui, category) -> count
    print(f"Reading new edges: {IN_EDGES}", flush=True)
    triples = defaultdict(lambda: defaultdict(int))
    # triples[(d_cui, e_cui)][category] = count
    n_in = 0
    for line in open(IN_EDGES):
        e = json.loads(line)
        d_cui = e["umls_cui"]
        e_cui = e["evidence_cui"]
        cat = e["category"]
        if d_cui == e_cui: continue  # self-loop
        triples[(d_cui, e_cui)][cat] += 1
        n_in += 1
    print(f"  read {n_in} mapped edges, aggregated to {len(triples)} unique (d,e) pairs",
          flush=True)

    # Add to KG
    n_new_edges = 0
    n_new_nodes = 0
    cat_counts = defaultdict(int)
    for (d_cui, e_cui), cat_counter in triples.items():
        # Add nodes if not present
        for c in (d_cui, e_cui):
            if c not in G:
                G.add_node(c)
                n_new_nodes += 1
        # Determine primary category (most frequent)
        primary_cat = max(cat_counter.items(), key=lambda x: x[1])[0]
        total_count = sum(cat_counter.values())
        # Weight: log(1+count) * source_weight (0.6 = consistent with PubMed prov)
        weight = math.log(1 + total_count) * 0.6
        # Add edge with category tag
        G.add_edge(d_cui, e_cui,
                   etype="HAS_PHENOTYPE",
                   weight=weight,
                   freq=total_count,
                   category=primary_cat,
                   categories_all=dict(cat_counter),
                   source="ie_v3_categorized")
        n_new_edges += 1
        cat_counts[primary_cat] += 1

    n_nodes_after = G.number_of_nodes()
    n_edges_after = G.number_of_edges()
    print(f"  after:  {n_nodes_after:,} nodes (+{n_new_nodes}), "
          f"{n_edges_after:,} edges (+{n_new_edges})", flush=True)
    print(f"  new edges by primary category: {dict(cat_counts)}", flush=True)

    print(f"Saving: {OUT_KG}", flush=True)
    OUT_KG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_KG, "wb") as f:
        pickle.dump(G, f)
    print(f"Done.", flush=True)


if __name__ == "__main__":
    main()
