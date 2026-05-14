#!/usr/bin/env python3
"""v29: 2-hop UMLS Q-expansion (deeper bridging).

v28 used 1-hop relations only. v29 adds 2-hop expansions: phen → related → Q.
Decay differently: 1-hop * 0.5, 2-hop * 0.25.
"""
from __future__ import annotations
import sys, json, math, pickle, time, argparse
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph_v23_sota.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v29_2hop.pkl"
PHEN_TO_Q_2HOP = Path("pilot/data/phen_to_q_umls_2hop.json")
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_in", default=str(GRAPH_IN))
    ap.add_argument("--graph_out", default=str(GRAPH_OUT))
    ap.add_argument("--decay_1hop", type=float, default=0.5)
    ap.add_argument("--decay_2hop", type=float, default=0.25)
    args = ap.parse_args()

    print(f"Loading v23 SOTA from {args.graph_in}")
    G = pickle.load(open(args.graph_in, "rb"))
    print(f"  v23: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    Q = set()
    value_cuis = json.load(open(VALUE_CUIS))
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)

    phen_data = json.load(open(PHEN_TO_Q_2HOP))
    print(f"  Loaded phen→Q (1hop+2hop): {len(phen_data):,} entries")

    existing = set((u,v) for u,v,e in G.edges(data=True) if e.get("etype")=="HAS_PHENOTYPE")
    n_1hop = 0; n_2hop = 0; merged_1 = 0; merged_2 = 0
    for u, v, e in list(G.edges(data=True)):
        if e.get("etype") != "HAS_PHENOTYPE": continue
        if v in Q: continue
        data = phen_data.get(v)
        if not data: continue
        orig_w = e.get("weight", 1.0)
        for q_cui in data.get("1hop", []):
            new_w = orig_w * args.decay_1hop
            if (u, q_cui) in existing:
                cur = G[u][q_cui]
                if cur.get("etype") == "HAS_PHENOTYPE":
                    cur["weight"] = cur.get("weight", 0) + new_w
                    merged_1 += 1
                    continue
            if q_cui not in G:
                G.add_node(q_cui, ntype="Phenotype", name=q_cui, source="v29_2hop")
            G.add_edge(u, q_cui, etype="HAS_PHENOTYPE", weight=new_w, source="qexpand_1hop")
            existing.add((u, q_cui))
            n_1hop += 1
        for q_cui in data.get("2hop", []):
            new_w = orig_w * args.decay_2hop
            if (u, q_cui) in existing:
                cur = G[u][q_cui]
                if cur.get("etype") == "HAS_PHENOTYPE":
                    cur["weight"] = cur.get("weight", 0) + new_w
                    merged_2 += 1
                    continue
            if q_cui not in G:
                G.add_node(q_cui, ntype="Phenotype", name=q_cui, source="v29_2hop")
            G.add_edge(u, q_cui, etype="HAS_PHENOTYPE", weight=new_w, source="qexpand_2hop")
            existing.add((u, q_cui))
            n_2hop += 1

    print(f"\nAdded 1-hop: {n_1hop:,} new, {merged_1:,} merged")
    print(f"Added 2-hop: {n_2hop:,} new, {merged_2:,} merged")
    print(f"\nFinal v29: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\nSaving to {args.graph_out}")
    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)
    print("Done.")


if __name__ == "__main__":
    main()
