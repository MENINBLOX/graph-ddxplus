#!/usr/bin/env python3
"""v103 property graph KG builder.

Disease → Phenotype edges with attribute distributions (per disease).
Aggregated from PubMed source documents via v103_run_shard.py.

학술적 grounding:
- HPO modifiers (HP:0410014 location, HP:0012824 severity, HP:0003674 onset)
- Phenopackets v2 (evidence-grounded, excluded field for negation)
- SNOMED CT qualifiers

Each edge attribute is a probability distribution over enum values,
empirically derived from multi-source IE (not LLM-hallucinated).
"""
from __future__ import annotations
import json, pickle, argparse, glob
from pathlib import Path
from collections import defaultdict
import networkx as nx


def name_to_cui_map():
    """Build phenotype-name → UMLS CUI mapping from existing v85/v95 KG."""
    # Reuse existing scispaCy/UMLS direct match
    # For prototype: use simple heuristic + curated dict
    import re

    # Try v92 enhanced mapping
    mapping = {}
    for path in ["pilot/data/cache/v85_cui_edges.jsonl",
                 "pilot/data/cache/v92_cui_edges.jsonl"]:
        if not Path(path).exists(): continue
        # These have disease→{cui: prob} format, no direct phen name
        pass

    # Use existing UMLS phenotype string table (MRCONSO + MRSTY)
    # Load on demand for prototype
    return mapping


def build_kg(per_disease_dir, out_path):
    """Build property graph KG from per-disease aggregated phenotypes."""
    G = nx.MultiDiGraph()

    n_diseases = 0
    n_edges = 0
    n_phens_total = 0
    for path in glob.glob(f"{per_disease_dir}/*.json"):
        with open(path) as f:
            data = json.load(f)
        disease_name = data["disease"]
        disease_cui = data["cui"]
        aggregated = data["aggregated"]

        # Add disease node
        G.add_node(disease_cui, ntype="disease", name=disease_name)
        n_diseases += 1

        # For each phenotype, add node + edge
        for phen_name, phen_data in aggregated.items():
            # Use phen_name as node ID for prototype (CUI mapping later)
            phen_id = phen_name.lower().strip()
            if phen_id not in G:
                G.add_node(phen_id, ntype="phenotype", name=phen_name)
            G.add_edge(disease_cui, phen_id,
                       etype="HAS_PHENOTYPE",
                       n_mentions=phen_data["n_mentions"],
                       frequency=phen_data["frequency_in_abstracts"],
                       location_dist=phen_data["location_dist"],
                       severity_dist=phen_data["severity_dist"],
                       onset_dist=phen_data["onset_dist"],
                       character_dist=phen_data["character_dist"])
            n_edges += 1
            n_phens_total += 1

    print(f"v103 KG built: {n_diseases} diseases, {n_edges} edges, "
          f"{G.number_of_nodes()} nodes total", flush=True)
    print(f"  Avg phenotypes/disease: {n_phens_total/max(n_diseases,1):.1f}", flush=True)

    # Sample edge inspection
    for disease, phen, ed in list(G.edges(data=True))[:3]:
        print(f"\n  [{G.nodes[disease].get('name','?')}] → [{phen}]", flush=True)
        print(f"    freq={ed['frequency']:.2f}, mentions={ed['n_mentions']}")
        if ed['location_dist']: print(f"    location: {dict(list(ed['location_dist'].items())[:3])}")
        if ed['severity_dist']: print(f"    severity: {dict(ed['severity_dist'])}")
        if ed['character_dist']: print(f"    character: {dict(list(ed['character_dist'].items())[:3])}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(G, open(out_path, "wb"))
    print(f"\nSaved → {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="pilot/data/cache/v103_per_disease")
    ap.add_argument("--out", default="pilot/data/onlykg_graph_v103.pkl")
    args = ap.parse_args()
    build_kg(args.in_dir, args.out)


if __name__ == "__main__":
    main()
