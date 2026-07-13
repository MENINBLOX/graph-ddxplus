#!/usr/bin/env python3
"""only-KG Phase 1b: Add UMLS CUI hierarchy edges to graph.

Loads MRREL.RRF, adds Phenotype-Phenotype edges of types:
  - SY (synonym)
  - PAR/CHD (parent/child) → IS_A
  - RB/RN (broader/narrower) → IS_A_BROADER / IS_A_NARROWER
  - RT (related, mainly HPO/SNOMED)
  - RO (related other, restricted to clinical SABs)

Filters MRREL to:
  - CUIs that appear in our KG (Phenotype CUIs only — not Disease CUIs)
  - Clinical SABs: HPO, SNOMEDCT_US, MSH, MEDLINEPLUS, MEDCIN

Output: replaces /mnt/medkg/kg/onlykg_graph.pkl with hierarchy-enriched version
"""
from __future__ import annotations
import sys, json, math, time, pickle
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

import networkx as nx

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v2.pkl"

# Clinical SABs (vocabularies considered relevant for phenotype matching)
CLINICAL_SABS = {"HPO", "SNOMEDCT_US", "MSH", "MEDLINEPLUS", "MEDCIN", "ICD10CM", "MEDDRA"}
ACCEPT_RELS = {"SY", "PAR", "CHD", "RB", "RN", "RT"}  # exclude RO (too noisy)


def main():
    print("=" * 70)
    print("only-KG Phase 1b: Add UMLS hierarchy edges")
    print("=" * 70)

    print(f"\n[1] Loading existing graph from {GRAPH_IN}...")
    with GRAPH_IN.open("rb") as f:
        G = pickle.load(f)
    print(f"    Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")

    # Get set of Phenotype CUIs (the ones we want to expand)
    pheno_cuis = set()
    for n, attrs in G.nodes(data=True):
        if attrs.get("ntype") == "Phenotype":
            pheno_cuis.add(n)
    print(f"    Phenotype CUIs in KG: {len(pheno_cuis):,}")

    # Also include patient evidence CUIs (we want to bridge to KG)
    print("\n[2] Loading patient evidence CUIs...")
    ev_cuis = json.load(open(MEDKG_ROOT / "kg" / "ddxplus_evidence_cuis.json"))
    patient_cuis = set()
    for cuis in ev_cuis.values():
        patient_cuis.update(cuis)
    print(f"    Patient evidence CUIs (DDXPlus questions): {len(patient_cuis):,}")
    print(f"    Patient ∩ KG Phenotype: {len(patient_cuis & pheno_cuis):,}")
    print(f"    Patient only (need bridging): {len(patient_cuis - pheno_cuis):,}")

    # Combined set: all CUIs we want to bridge via MRREL
    target_cuis = pheno_cuis | patient_cuis
    print(f"\n[3] Total target CUIs for hierarchy: {len(target_cuis):,}")

    # Stream MRREL
    print(f"\n[4] Streaming MRREL.RRF (2.98M relations)...")
    t0 = time.time()
    n_kept = 0; n_total = 0
    hierarchy_edges = defaultdict(set)  # (cui1, cui2) → set of (rel, sab)

    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            n_total += 1
            if n_total % 500000 == 0:
                print(f"    {n_total:,} processed, {n_kept:,} kept ({time.time()-t0:.0f}s)")
            parts = line.split("|")
            if len(parts) < 12: continue
            cui1, _, _, rel, cui2, _, _, _, _, _, sab, _ = parts[:12]
            if rel not in ACCEPT_RELS: continue
            if sab not in CLINICAL_SABS: continue
            # Both ends must be in target set
            if cui1 not in target_cuis or cui2 not in target_cuis: continue
            if cui1 == cui2: continue
            hierarchy_edges[(cui1, cui2)].add((rel, sab))
            n_kept += 1
    print(f"    Done. {n_total:,} relations scanned, {n_kept:,} kept ({len(hierarchy_edges):,} unique pairs)")

    # Add to graph
    print(f"\n[5] Adding hierarchy edges to graph...")
    added = 0
    # Pre-add bridge nodes (patient-only CUIs not yet in graph)
    bridge_nodes = patient_cuis - pheno_cuis
    for cui in bridge_nodes:
        if cui not in G:
            G.add_node(cui, ntype="Phenotype", name=cui, source="patient_bridge")
    print(f"    Added {len(bridge_nodes):,} bridge Phenotype nodes")

    for (c1, c2), rel_set in hierarchy_edges.items():
        # Use weight = 1.0 for SY, 0.6 for PAR/CHD/RB/RN, 0.4 for RT
        max_w = 0
        for rel, sab in rel_set:
            if rel == "SY": w = 1.0
            elif rel in {"PAR","CHD","RB","RN"}: w = 0.6
            else: w = 0.4
            if w > max_w: max_w = w
        G.add_edge(c1, c2, etype="HIERARCHY",
                   weight=max_w, rels=list(rel_set))
        added += 1
    print(f"    Added {added:,} HIERARCHY edges")
    print(f"    Final: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Save
    print(f"\n[6] Saving to {GRAPH_OUT}...")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)
    print(f"    Saved")

    # Sample
    print("\n[7] Sample: bridging URTI phenotypes via hierarchy")
    URTI = "C0041912"
    print(f"  URTI direct phenotypes (1-hop, HAS_PHENOTYPE):")
    for _, p, e in list(G.out_edges(URTI, data=True))[:5]:
        if e.get("etype") == "HAS_PHENOTYPE":
            name = G.nodes[p].get("name", p)
            print(f"    - {name} ({p})")
    print(f"  URTI 2-hop via HIERARCHY:")
    URTI_phenos = set(p for _, p, e in G.out_edges(URTI, data=True) if e.get("etype") == "HAS_PHENOTYPE")
    bridged = set()
    for p in URTI_phenos:
        for _, p2, e in G.out_edges(p, data=True):
            if e.get("etype") == "HIERARCHY":
                bridged.add(p2)
    print(f"    {len(bridged):,} CUIs reachable in 2-hop")


if __name__ == "__main__":
    main()
