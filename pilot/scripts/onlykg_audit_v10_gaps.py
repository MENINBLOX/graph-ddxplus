#!/usr/bin/env python3
"""Audit v10 KG for gaps that need targeted re-IE.

1. Per-disease coverage analysis:
   - Q∩phens count
   - In-degree distribution
2. Singleton phenotype detection (indegree=1, weak signal)
3. Low-coverage DDXPlus diseases — these need more raw text IE
4. Generate priority list: which diseases × what missing phenotypes
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v10.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"

OUT = Path("pilot/results/v10_gap_audit.json")


def main():
    print("Loading v10 graph...")
    G = pickle.load(open(GRAPH, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    ddx_cuis = {info["cui"]: dn for dn, info in icd.items() if "cui" in info}

    # Q universe
    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)
    print(f"|Q| = {len(Q)}")

    # 1. Per-disease coverage
    print("\n=== 49 DDXPlus disease coverage in v10 ===")
    disease_stats = {}
    low_q = []
    for cui, name in ddx_cuis.items():
        if cui not in G:
            disease_stats[cui] = {"name": name, "in_graph": False}
            continue
        phens_list = [(p, e.get("weight", 0)) for _, p, e in G.out_edges(cui, data=True)
                       if e.get("etype") == "HAS_PHENOTYPE"]
        n_edges = len(phens_list)
        q_phens = [(p, w) for p, w in phens_list if p in Q]
        n_q_phens = len(q_phens)
        # Singleton phens (only this disease points to them — low IDF utility)
        # Compute later globally
        disease_stats[cui] = {
            "name": name,
            "in_graph": True,
            "n_edges": n_edges,
            "n_q_phens": n_q_phens,
            "top_q_phens": sorted(q_phens, key=lambda x: -x[1])[:10]
        }
        if n_q_phens < 10:
            low_q.append((name, cui, n_edges, n_q_phens))

    print(f"Disease in graph: {sum(1 for v in disease_stats.values() if v.get('in_graph'))}/49")
    print(f"Diseases with Q∩phens<10 (CRITICAL): {len(low_q)}/49")
    for name, cui, ne, qp in sorted(low_q, key=lambda x: x[3]):
        print(f"  {name:40s} edges={ne:3d}  Q∩phens={qp}")

    # 2. Singleton phenotype detection
    print("\n=== Singleton phenotype analysis (indegree=1) ===")
    phen_indegree = Counter()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            phen_indegree[v] += 1
    indeg_dist = Counter(phen_indegree.values())
    print(f"Phenotype indegree distribution (top 10):")
    for k in sorted(indeg_dist.keys())[:10]:
        print(f"  indegree={k}: {indeg_dist[k]:,} nodes")
    n_singleton = indeg_dist.get(1, 0)
    print(f"\nSingleton phens (indegree=1): {n_singleton:,}")
    print("These add no multi-disease discrimination signal.")
    print("If isolated (no HIERARCHY edges either), they're effectively useless.")

    # 3. Check which DDXPlus disease phens are singletons (lose them = lose coverage)
    print("\n=== DDXPlus disease phens that are singletons ===")
    ddx_singletons = Counter()
    for cui, name in ddx_cuis.items():
        if cui not in G: continue
        for _, p, e in G.out_edges(cui, data=True):
            if e.get("etype") == "HAS_PHENOTYPE" and phen_indegree[p] == 1:
                ddx_singletons[name] += 1
    print(f"Top 10 diseases with most singleton phens (these are disease-specific signals):")
    for n, c in ddx_singletons.most_common(10):
        print(f"  {n}: {c} singletons")

    # 4. Save audit
    audit = {
        "n_diseases_in_graph": sum(1 for v in disease_stats.values() if v.get("in_graph")),
        "n_low_q_coverage": len(low_q),
        "low_q_diseases": [
            {"name": name, "cui": cui, "n_edges": ne, "n_q_phens": qp}
            for name, cui, ne, qp in low_q
        ],
        "singleton_count": n_singleton,
        "indegree_distribution": dict(indeg_dist),
        "ddx_disease_singletons": dict(ddx_singletons),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(audit, f, indent=2)
    print(f"\nSaved audit to {OUT}")

    print("\n=== Recommended re-IE targets ===")
    print(f"Priority 1 (Q∩phens < 5): {sum(1 for x in low_q if x[3] < 5)} diseases")
    print(f"Priority 2 (Q∩phens 5-9): {sum(1 for x in low_q if 5 <= x[3] < 10)} diseases")
    print(f"\nThese need additional raw text + LLM IE for patient-questionnaire-style phenotypes.")


if __name__ == "__main__":
    main()
