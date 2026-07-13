#!/usr/bin/env python3
"""only-KG v3 graph build: re-link phenotype text → CUI via UMLS MRCONSO.

The v2 graph used scispaCy-linked CUIs only. Many lay phenotypes
("shortness of breath", "chest pain", "fever") failed scispaCy linking.
This build re-links via direct UMLS MRCONSO string match (case-insensitive,
normalized), which recovers ~50%+ of lay phenotypes.

This is benchmark-agnostic — only uses UMLS structure + existing IE output.
"""
from __future__ import annotations
import sys, json, math, time, pickle, re
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

import networkx as nx

FEATURES = MEDKG_ROOT / "kg" / "disease_features_dual_v2_by_cui.json"
GRAPH_V2 = MEDKG_ROOT / "kg" / "onlykg_graph_v2.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v3.pkl"

# Preferred SABs for disambiguation (clinical priority)
PREFERRED_SABS = ["HPO", "SNOMEDCT_US", "MSH", "MEDCIN", "NCI", "ICD10CM", "MEDDRA", "CHV", "MEDLINEPLUS"]
SAB_PRIORITY = {s: i for i, s in enumerate(PREFERRED_SABS)}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = text.lower().strip()
    t = re.sub(r'[\(\)\[\]\{\}]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def main():
    print("=" * 70)
    print("only-KG v3: re-link phenotype text → CUI via MRCONSO")
    print("=" * 70)

    print("\n[1] Loading dual_v2 features...")
    features = json.load(open(FEATURES))
    print(f"    diseases: {len(features)}")

    # Collect unique phenotype strings (normalized)
    phen_strings = set()
    for dcui, phens in features.items():
        for p in phens:
            phen_strings.add(normalize(p["phenotype"]))
    print(f"    unique normalized phenotype strings: {len(phen_strings):,}")

    print("\n[2] Building MRCONSO string → CUI index (filter to candidate strings)...")
    # We only need entries whose normalized string is in phen_strings (saves memory)
    str2cui = {}  # normalized_string → (cui, sab_priority)
    t0 = time.time()
    n_total = 0; n_kept = 0
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            n_total += 1
            if n_total % 2000000 == 0:
                print(f"    {n_total:,} lines processed, {len(str2cui):,} strings indexed ({time.time()-t0:.0f}s)")
            parts = line.split("|")
            if len(parts) < 15: continue
            if parts[1] != "ENG": continue
            text = parts[14]
            norm = normalize(text)
            if norm not in phen_strings: continue
            cui = parts[0]; sab = parts[11]
            prio = SAB_PRIORITY.get(sab, 99)
            if norm in str2cui:
                # Keep highest priority (lowest number)
                old_cui, old_prio = str2cui[norm]
                if prio < old_prio:
                    str2cui[norm] = (cui, prio)
            else:
                str2cui[norm] = (cui, prio)
                n_kept += 1
    print(f"    Done: {n_total:,} MRCONSO lines, {len(str2cui):,} phenotype strings linked ({time.time()-t0:.0f}s)")

    coverage = len(str2cui) / len(phen_strings) * 100
    print(f"    String → CUI coverage: {coverage:.1f}%")

    print("\n[3] Building edges (disease, phenotype_cui, weight)...")
    edges = []  # (disease_cui, phen_cui, weight, sources)
    phen_node_names = {}
    disease_freq = Counter()
    phen_freq = Counter()  # for IDF
    skipped = 0
    for dcui, phens in features.items():
        for p in phens:
            norm = normalize(p["phenotype"])
            cui_info = str2cui.get(norm)
            if cui_info is None:
                skipped += 1; continue
            pcui, _ = cui_info
            if pcui == dcui: continue  # self-loop
            sources = p.get("sources", [])
            n_sources = p.get("n_sources", len(sources))
            score = p.get("score", 0.5)
            # weight: log1p(score) × source_agreement × IDF
            source_agreement = 0.5 + 0.5 * min(n_sources, 5) / 5
            # IDF computed after first pass
            edges.append({
                "disease": dcui, "phen": pcui,
                "score": score, "sources": sources,
                "n_sources": n_sources,
                "source_agreement": source_agreement,
            })
            phen_node_names[pcui] = p["phenotype"]  # keep original text
            disease_freq[dcui] += 1
            phen_freq[pcui] += 1
    print(f"    Edges: {len(edges):,}, unique phenotype CUIs: {len(phen_node_names):,}")
    print(f"    Skipped (no CUI match): {skipped:,}")

    # IDF: log(N_diseases / freq)
    N = max(len(disease_freq), 1)
    idf = {p: math.log(N / phen_freq[p]) for p in phen_freq}

    print("\n[4] Loading v2 graph as base (UNION mode — keep all v2 edges)...")
    with GRAPH_V2.open("rb") as f:
        G = pickle.load(f)
    n_v2_phen = sum(1 for _,_,e in G.edges(data=True) if e.get("etype")=="HAS_PHENOTYPE")
    print(f"    Kept {n_v2_phen:,} v2 HAS_PHENOTYPE edges")

    print("\n[5] Adding v3 edges (union with v2)...")
    # Track existing (disease, phen) pairs to avoid duplicates
    existing_pairs = set()
    for u, v, edata in G.edges(data=True):
        if edata.get("etype") == "HAS_PHENOTYPE":
            existing_pairs.add((u, v))
    added = 0; dedup = 0
    for e in edges:
        d, p = e["disease"], e["phen"]
        if (d, p) in existing_pairs:
            dedup += 1; continue
        if p not in G:
            G.add_node(p, ntype="Phenotype", name=phen_node_names.get(p, p), source="v3_relink")
        w = math.log1p(e["score"] * 10) * e["source_agreement"] * idf.get(p, 1.0)
        G.add_edge(d, p, etype="HAS_PHENOTYPE", weight=w,
                   score=e["score"], sources=e["sources"])
        added += 1
    print(f"    Added {added:,} new v3 HAS_PHENOTYPE edges (skipped {dedup:,} duplicates with v2)")
    print(f"    Final: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\n[6] Saving v3 graph to {GRAPH_OUT}...")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)
    print(f"    Saved")

    # Verify: Pneumonia phenotypes
    PN = "C0694504"
    if PN in G:
        phens = [(p, G.nodes[p].get("name","?"), edata.get("weight",0))
                 for _, p, edata in G.out_edges(PN, data=True)
                 if edata.get("etype") == "HAS_PHENOTYPE"]
        phens.sort(key=lambda x: -x[2])
        print(f"\n[7] Sample Pneumonia (C0694504) top 15 phenotypes:")
        for pcui, name, w in phens[:15]:
            print(f"    [{w:6.2f}] {name:<40s} {pcui}")


if __name__ == "__main__":
    main()
