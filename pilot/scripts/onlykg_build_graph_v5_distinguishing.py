#!/usr/bin/env python3
"""only-KG v5 graph: heavy emphasis on is_distinguishing phenotypes.

is_distinguishing=True flag in IE output marks phenotypes that appear in
'distinguishing' source (differential diagnosis text). These are
patient-reportable symptoms that discriminate the disease from cluster peers.

Strategy: build graph using v4 as base, BOOST is_distinguishing phenotype
edges by 5x; downweight non-distinguishing by 0.5x.
"""
from __future__ import annotations
import sys, json, math, time, pickle, re
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

import networkx as nx

FEATURES = MEDKG_ROOT / "kg" / "disease_features_dual_v2_by_cui.json"
GRAPH_V4 = MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v5.pkl"
PHEN_LINKS = MEDKG_ROOT / "kg" / "phenotype_scispacy_links.json"

PREFERRED_SABS = ["HPO", "SNOMEDCT_US", "MSH", "MEDCIN", "NCI", "ICD10CM"]
SAB_PRIORITY = {s: i for i, s in enumerate(PREFERRED_SABS)}


def normalize(text):
    t = text.lower().strip()
    t = re.sub(r'[()\[\]{}]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist_boost", type=float, default=5.0)
    ap.add_argument("--nondist_dampen", type=float, default=0.5)
    args = ap.parse_args()

    print("=" * 70)
    print(f"only-KG v5: is_distinguishing-weighted graph (boost={args.dist_boost}, dampen={args.nondist_dampen})")
    print("=" * 70)

    print("\n[1] Loading features + v4 graph...")
    features = json.load(open(FEATURES))
    G = pickle.load(open(GRAPH_V4, "rb"))
    phen_links = json.load(open(PHEN_LINKS)) if PHEN_LINKS.exists() else {}

    # Remove ALL HAS_PHENOTYPE edges (we'll rebuild)
    to_remove = [(u, v, k) for u, v, k, edata in G.edges(keys=True, data=True)
                 if edata.get("etype") == "HAS_PHENOTYPE"]
    for u, v, k in to_remove:
        G.remove_edge(u, v, k)
    print(f"  Removed {len(to_remove):,} existing HAS_PHENOTYPE edges")

    # Build MRCONSO index for cleaner matching (fast lookup)
    # Reuse v3's mappings - load directly from MRCONSO
    print("\n[2] Building MRCONSO string index...")
    phen_strings = set()
    for dcui, phens in features.items():
        for p in phens:
            phen_strings.add(normalize(p["phenotype"]))
    str2cui = {}
    t0 = time.time()
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < 15 or parts[1] != "ENG": continue
            norm = normalize(parts[14])
            if norm not in phen_strings: continue
            cui = parts[0]; sab = parts[11]
            prio = SAB_PRIORITY.get(sab, 99)
            if norm in str2cui:
                old_cui, old_prio = str2cui[norm]
                if prio < old_prio: str2cui[norm] = (cui, prio)
            else:
                str2cui[norm] = (cui, prio)
    print(f"  {len(str2cui):,} strings linked ({time.time()-t0:.0f}s)")

    # Collect (disease, phen_cui_set, weight_multiplier) from each entry
    print("\n[3] Collecting edges with is_distinguishing weighting...")
    edges = []  # list of (disease_cui, phen_cui, weight_components)
    n_dist = 0; n_nondist = 0
    for dcui, phens in features.items():
        for p in phens:
            norm = normalize(p["phenotype"])
            # Collect possible CUIs: MRCONSO direct + scispaCy decomposed
            phen_cuis = set()
            direct = str2cui.get(norm)
            if direct: phen_cuis.add(direct[0])
            for cui, _, _ in phen_links.get(norm, []):
                phen_cuis.add(cui)
            if not phen_cuis: continue

            is_dist = p.get("is_distinguishing", False)
            base_score = p.get("score", 0.5)
            n_sources = p.get("n_sources", 1)
            source_agreement = 0.5 + 0.5 * min(n_sources, 5) / 5

            mult = args.dist_boost if is_dist else args.nondist_dampen
            if is_dist: n_dist += 1
            else: n_nondist += 1

            for pcui in phen_cuis:
                if pcui == dcui: continue
                edges.append({
                    "disease": dcui, "phen": pcui,
                    "score": base_score * mult,
                    "is_dist": is_dist,
                    "source_agreement": source_agreement,
                })

    print(f"  Distinguishing entries: {n_dist:,}, non-dist: {n_nondist:,}")
    print(f"  Total edge proto: {len(edges):,}")

    # IDF over 49 disease universe (only for emphasis)
    # Note: IDF computed on 49 DDXPlus diseases would be benchmark-aware
    # Instead, compute on ALL features keys (universal)
    N = len(features)
    phen_freq = Counter(e["phen"] for e in edges)
    idf = {p: math.log(N / max(c, 1)) for p, c in phen_freq.items()}

    # Add edges to graph
    print("\n[4] Adding edges to graph...")
    edge_acc = {}  # (disease, phen) -> aggregated weight
    for e in edges:
        key = (e["disease"], e["phen"])
        w = math.log1p(e["score"] * 10) * e["source_agreement"] * idf.get(e["phen"], 1.0)
        edge_acc[key] = edge_acc.get(key, 0) + w  # accumulate

    for (d, p), w in edge_acc.items():
        if p not in G:
            G.add_node(p, ntype="Phenotype", name=p, source="v5")
        G.add_edge(d, p, etype="HAS_PHENOTYPE", weight=w)
    print(f"  Added {len(edge_acc):,} unique HAS_PHENOTYPE edges")
    print(f"  Final: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\n[5] Saving to {GRAPH_OUT}...")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)

    PN = "C0694504"
    if PN in G:
        phens = [(p, G.nodes[p].get("name","?"), edata.get("weight",0))
                 for _, p, edata in G.out_edges(PN, data=True)
                 if edata.get("etype") == "HAS_PHENOTYPE"]
        phens.sort(key=lambda x: -x[2])
        print(f"\nSample Pneumonia top 15:")
        for pcui, name, w in phens[:15]:
            print(f"  [{w:6.2f}] {name:<40s} {pcui}")


if __name__ == "__main__":
    main()
