#!/usr/bin/env python3
"""v80 step 3 — Integrate LLM-derived edges into KG.

Takes v45 base KG + v80_cui_edges.jsonl and builds a new KG with:
- All existing v45 edges (PubMed/Wiki/MedlinePlus)
- New edges from LLM IE (with weight derived from LLM prevalence rating)

Weight scheme:
- LLM rates phen at probability p (0-1)
- Convert to additive edge weight: w_llm = p * scale
- Add to existing PubMed weight (augment, not replace)
- scale is hyper-parameter (default 30 so LLM 'always' (~0.95) gives weight 28.5,
  matching strong PubMed edges)

Category: All LLM-derived edges marked as 'patient_reportable' since LLM was
asked for typical patient findings (lay + clinical mix). v71 lay-mode eval
will use them.
"""
from __future__ import annotations
import json, argparse, pickle, sys
from pathlib import Path
sys.path.insert(0, "pilot/scripts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_kg", default="pilot/data/onlykg_graph_v49_v5_full.pkl")
    ap.add_argument("--llm_edges", default="pilot/data/cache/v80_cui_edges.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=30.0,
                    help="LLM prob → edge weight scale")
    ap.add_argument("--category", default="patient_reportable")
    args = ap.parse_args()

    print(f"Loading base KG {args.base_kg}...", flush=True)
    G = pickle.load(open(args.base_kg, "rb"))
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}", flush=True)

    n_added = 0; n_existing = 0; n_new_node = 0; n_disease = 0
    with open(args.llm_edges) as f:
        for line in f:
            r = json.loads(line)
            dcui = r["dcui"]
            n_disease += 1
            # Ensure disease node exists
            if dcui not in G:
                G.add_node(dcui, type="disease")
            for phen_cui, prob in r["edges"].items():
                if phen_cui == dcui: continue  # self-loop
                w = float(prob) * args.scale
                # Ensure phen node
                if phen_cui not in G:
                    G.add_node(phen_cui, type="phenotype")
                    n_new_node += 1
                # Check existing HAS_PHENOTYPE edge
                existing_w = 0.0
                if G.has_edge(dcui, phen_cui):
                    for k, ed in G[dcui][phen_cui].items():
                        if ed.get("etype") == "HAS_PHENOTYPE":
                            existing_w = max(existing_w, ed.get("weight", 0))
                            n_existing += 1
                            break
                # ADD new LLM edge (augment alongside existing)
                G.add_edge(dcui, phen_cui, etype="HAS_PHENOTYPE",
                           weight=w, category=args.category, source="v80_llm_ie")
                n_added += 1

    print(f"\nKG augmentation summary:")
    print(f"  Diseases processed: {n_disease}")
    print(f"  LLM edges added: {n_added}")
    print(f"  Of which already had PubMed edge: {n_existing}")
    print(f"  New phenotype nodes created: {n_new_node}")
    print(f"  Final KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(G, open(args.out, "wb"))
    print(f"  Saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
