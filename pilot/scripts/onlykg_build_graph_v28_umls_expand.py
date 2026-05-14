#!/usr/bin/env python3
"""v28: expand v23 SOTA phens to Q-CUIs via UMLS RB/RN/RO/SY/PAR/CHD relations.

Problem: only 17% of v23 phens are in Q (questionnaire universe) → 83% don't
contribute to scoring. GT KG has 100% Q coverage (72 Q-phens/disease).

Solution: For each phen P in our KG that is NOT in Q but has UMLS relation
(RB/RN/RO/SY/PAR/CHD) to a Q-CUI Q', add a direct (disease, Q') edge with
weight = original_weight * decay (decay < 1.0 to reflect indirect link).

Expansion: avg 21 → 55 Q-phens/disease (vs GT 72).
"""
from __future__ import annotations
import sys, json, math, pickle, time, argparse
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph_v23_sota.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v28_qexpand.pkl"
PHEN_TO_Q = Path("pilot/data/phen_to_q_umls.json")
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_in", default=str(GRAPH_IN))
    ap.add_argument("--graph_out", default=str(GRAPH_OUT))
    ap.add_argument("--decay", type=float, default=0.5, help="weight decay for expanded edges")
    ap.add_argument("--accept_rel", default="RB,RN,RO,SY,PAR,CHD", help="UMLS relations to use")
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
    print(f"  Q: {len(Q)}")

    if not PHEN_TO_Q.exists():
        # Build phen→Q on the fly
        print("Building phen→Q expansion from MRREL...")
        # Get all phens (HAS_PHENOTYPE targets)
        all_phens = set()
        for u, v, e in G.edges(data=True):
            if e.get("etype") == "HAS_PHENOTYPE":
                all_phens.add(v)
        print(f"  all phens: {len(all_phens):,}")
        accept_rel = set(args.accept_rel.split(","))
        phen_to_q = defaultdict(set)
        t0 = time.time()
        with open(UMLS_DIR / 'MRREL.RRF') as f:
            for line in f:
                parts = line.split('|')
                if len(parts) < 5: continue
                if parts[3] not in accept_rel: continue
                c1, c2 = parts[0], parts[4]
                if c1 in all_phens and c2 in Q and c1 != c2:
                    phen_to_q[c1].add(c2)
                if c2 in all_phens and c1 in Q and c1 != c2:
                    phen_to_q[c2].add(c1)
        print(f"  built: {len(phen_to_q):,} phens with Q relations ({time.time()-t0:.0f}s)")
    else:
        phen_to_q_save = json.load(open(PHEN_TO_Q))
        phen_to_q = {k: set(v) for k, v in phen_to_q_save.items()}
        print(f"  Loaded phen→Q expansion: {len(phen_to_q):,} entries")

    # Add edges: for each (disease, phen_NOT_in_Q) edge, expand to (disease, Q-CUI) edges
    added = 0; merged = 0
    existing = set((u,v) for u,v,e in G.edges(data=True) if e.get("etype")=="HAS_PHENOTYPE")
    n_disease_edges = 0
    for u, v, e in list(G.edges(data=True)):
        if e.get("etype") != "HAS_PHENOTYPE": continue
        n_disease_edges += 1
        # Only expand if v is NOT in Q (otherwise direct match)
        if v in Q: continue
        related_q = phen_to_q.get(v)
        if not related_q: continue
        orig_w = e.get("weight", 1.0)
        for q_cui in related_q:
            new_w = orig_w * args.decay
            if (u, q_cui) in existing:
                # Merge weight
                cur = G[u][q_cui]
                if cur.get("etype") == "HAS_PHENOTYPE":
                    cur["weight"] = cur.get("weight", 0) + new_w
                    merged += 1
                    continue
            if q_cui not in G:
                G.add_node(q_cui, ntype="Phenotype", name=q_cui, source="v28_qexpand")
            G.add_edge(u, q_cui, etype="HAS_PHENOTYPE", weight=new_w, source="qexpand_umls")
            existing.add((u, q_cui))
            added += 1
    print(f"\nExisting HAS_PHENOTYPE edges scanned: {n_disease_edges:,}")
    print(f"Added {added:,} new Q-expanded edges")
    print(f"Merged {merged:,} into existing")
    print(f"\nFinal v28: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\nSaving to {args.graph_out}")
    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)
    print("Done.")


if __name__ == "__main__":
    main()
