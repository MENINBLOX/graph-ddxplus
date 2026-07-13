#!/usr/bin/env python3
"""v84 — Merge v80 (generic IE) + v83 (discriminative N=10) edges.

Hypothesis: Two different prompts capture different aspects of disease
phenotypes. Merging both gives broader coverage. KG starts from v45 base.

Edge merging: if same (disease, phen) pair appears in both, take max prob.
"""
from __future__ import annotations
import json, argparse, pickle, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, "pilot/scripts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_kg", default="pilot/data/onlykg_graph_v49_v5_full.pkl")
    ap.add_argument("--llm_edges_a", required=True)
    ap.add_argument("--llm_edges_b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=3.0)
    args = ap.parse_args()

    # Merge edges from both sources
    merged = defaultdict(dict)  # dcui → {phen_cui: max_prob}
    n_a = n_b = 0
    for path, tag in [(args.llm_edges_a, 'a'), (args.llm_edges_b, 'b')]:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                dcui = r["dcui"]
                for phen_cui, prob in r["edges"].items():
                    cur = merged[dcui].get(phen_cui, 0)
                    merged[dcui][phen_cui] = max(cur, float(prob))
                    if tag == 'a': n_a += 1
                    else: n_b += 1
    print(f"Source A edges: {n_a}, Source B edges: {n_b}")
    print(f"Merged diseases: {len(merged)}, unique edges: {sum(len(v) for v in merged.values())}")

    print(f"Loading base KG {args.base_kg}...", flush=True)
    G = pickle.load(open(args.base_kg, "rb"))
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}", flush=True)

    n_added = 0
    for dcui, edges in merged.items():
        if dcui not in G:
            G.add_node(dcui, type="disease")
        for phen_cui, prob in edges.items():
            if phen_cui == dcui: continue
            if phen_cui not in G:
                G.add_node(phen_cui, type="phenotype")
            w = prob * args.scale
            G.add_edge(dcui, phen_cui, etype="HAS_PHENOTYPE",
                       weight=w, category="patient_reportable", source="v84_llm_merged")
            n_added += 1

    print(f"LLM merged edges added: {n_added}")
    print(f"Final KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(G, open(args.out, "wb"))
    print(f"Saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
