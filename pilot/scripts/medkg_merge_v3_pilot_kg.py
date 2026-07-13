#!/usr/bin/env python3
"""Merge BOTH v3 IE outputs (priority + pilot1k) into v41_universal KG.

CRITICAL: Start from v39 (universal-only baseline), NOT v40_categorized.
v40_categorized included benchmark-priority (159 CUIs) — violation of strict principles.
v41_universal = v39 + IE v3 categorized edges from BOTH:
  - 159 priority CUIs (benchmark-derived selection, marked as such)
  - 326 pilot1k CUIs (random sample from UMLS DISO focused gap, benchmark-blind)

Edges from each source are tagged with `selection_origin` so we can ablation-test.
"""
from __future__ import annotations
import json, pickle, sys, math
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

IN_KG = MEDKG_ROOT / "kg" / "onlykg_graph_v39_history.pkl"
IN_PILOT = MEDKG_ROOT / "processed" / "edges_universal_v3_pilot1k_cui.jsonl"
IN_PRIORITY = MEDKG_ROOT / "processed" / "edges_universal_v3_cui.jsonl"
OUT_KG = MEDKG_ROOT / "kg" / "onlykg_graph_v41_universal.pkl"


def aggregate(path, origin):
    """Aggregate (d_cui, e_cui, category) -> count from file."""
    triples = defaultdict(lambda: defaultdict(int))
    n = 0
    for line in open(path):
        e = json.loads(line)
        d, ec, cat = e["umls_cui"], e["evidence_cui"], e["category"]
        if d == ec: continue
        triples[(d, ec)][cat] += 1
        n += 1
    print(f"  {origin}: {n} mapped edges → {len(triples)} unique (d,e) pairs", flush=True)
    return triples


def main():
    print(f"Loading STRICT baseline KG: {IN_KG}", flush=True)
    G = pickle.load(open(IN_KG, "rb"))
    print(f"  v39 baseline: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges",
          flush=True)

    print("Loading IE v3 categorized edges...", flush=True)
    pilot_tri = aggregate(IN_PILOT, "pilot1k(universal)")
    priority_tri = aggregate(IN_PRIORITY, "priority159(benchmark-driven)")

    n_new_edges = 0
    n_new_nodes = 0
    cat_counts = defaultdict(int)
    origin_counts = defaultdict(int)

    # Process pilot1k (universal-only) edges first → tagged 'universal'
    for (d, e), cats in pilot_tri.items():
        for c in (d, e):
            if c not in G:
                G.add_node(c); n_new_nodes += 1
        primary = max(cats.items(), key=lambda x: x[1])[0]
        weight = math.log(1 + sum(cats.values())) * 0.6
        G.add_edge(d, e, etype="HAS_PHENOTYPE", weight=weight,
                   freq=sum(cats.values()),
                   category=primary,
                   categories_all=dict(cats),
                   source="ie_v3",
                   selection_origin="universal")
        n_new_edges += 1
        cat_counts[primary] += 1
        origin_counts["universal"] += 1

    # Then priority (benchmark-driven) edges → tagged 'benchmark_priority'
    for (d, e), cats in priority_tri.items():
        for c in (d, e):
            if c not in G:
                G.add_node(c); n_new_nodes += 1
        primary = max(cats.items(), key=lambda x: x[1])[0]
        weight = math.log(1 + sum(cats.values())) * 0.6
        G.add_edge(d, e, etype="HAS_PHENOTYPE", weight=weight,
                   freq=sum(cats.values()),
                   category=primary,
                   categories_all=dict(cats),
                   source="ie_v3",
                   selection_origin="benchmark_priority")
        n_new_edges += 1
        cat_counts[primary] += 1
        origin_counts["benchmark_priority"] += 1

    print(f"\n  after: {G.number_of_nodes():,} nodes (+{n_new_nodes}), "
          f"{G.number_of_edges():,} edges (+{n_new_edges})", flush=True)
    print(f"  by category: {dict(cat_counts)}", flush=True)
    print(f"  by origin:   {dict(origin_counts)}", flush=True)

    OUT_KG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_KG, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved: {OUT_KG}", flush=True)


if __name__ == "__main__":
    main()
