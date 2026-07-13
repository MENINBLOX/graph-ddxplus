#!/usr/bin/env python3
"""v94 — KG cleanup: noise hub CUI 제거.

원칙 11 적용: demographic/temporal/qualifier CUI가 phenotype으로 inject되어
IDF discrimination 약화. 다음 CUI들을 HAS_PHENOTYPE 관계에서 제거:

- Demographic: Woman, Man, Male, Female, Child, Adult, Patient
- Temporal: year, month, day, week
- Qualifier: Severe, Mild, Moderate, History
- Generic body: Age, "Symptoms"
"""
from __future__ import annotations
import pickle, argparse
from pathlib import Path


# Conservative noise list — only temporal/qualifier/meta CUIs.
# Keep demographic (Woman/Adult/Child/Male/Female) — provides disease-specificity.
# v94 (all 16) regressed DDXPlus -1.3%p; v94b removes only proven non-phenotype.
NOISE_CUIS = {
    "C0001779",  # Age (17,550) - meta
    "C0439234",  # year (14,661) - temporal
    "C0030705",  # Patients (7,903) - meta
    "C1457887",  # Symptoms (5,683) - meta
    "C0205082",  # Severe (severity modifier) (5,500) - qualifier
    "C0019664",  # History (4,507) - meta
    "C0392756",  # Reduced (4,060) - qualifier
    "C0241889",  # Family history (finding) (3,272) - meta
    "C0439231",  # month (3,078) - temporal
    # Kept (demographic = disease-discriminative):
    #   C0043210 Woman, C0086582 Males, C0008059 Child, C0025266 Male pop,
    #   C0001675 Adult, C0332239 Young, C0086287 Females
    # Kept (legitimate phenotype/anatomy):
    #   C0030193 Pain, C0015392 Eye, C0018787 Heart, C0041657 Unconscious
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_graph", required=True)
    ap.add_argument("--out_graph", required=True)
    args = ap.parse_args()

    print(f"Loading {args.in_graph}...", flush=True)
    G = pickle.load(open(args.in_graph, "rb"))
    print(f"  before: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)

    n_edges_removed = 0
    n_nodes_isolated = 0
    edges_to_remove = []
    for u, v, k, ed in G.edges(keys=True, data=True):
        if ed.get("etype") != "HAS_PHENOTYPE": continue
        if v in NOISE_CUIS:
            edges_to_remove.append((u, v, k))
    for u, v, k in edges_to_remove:
        G.remove_edge(u, v, k)
        n_edges_removed += 1

    # Remove isolated phenotype nodes
    for cui in NOISE_CUIS:
        if cui in G and G.degree(cui) == 0:
            G.remove_node(cui)
            n_nodes_isolated += 1

    print(f"\n  HAS_PHENOTYPE edges removed: {n_edges_removed:,}")
    print(f"  Isolated noise CUI nodes removed: {n_nodes_isolated}")
    print(f"  after: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)

    Path(args.out_graph).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(G, open(args.out_graph, "wb"))
    print(f"  Saved → {args.out_graph}", flush=True)


if __name__ == "__main__":
    main()
