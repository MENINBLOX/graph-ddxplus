#!/usr/bin/env python3
"""only-KG vocabulary gap diagnostic v2: edge-level analysis.

Patient CUIs are all KG nodes (via patient_bridge). The real gap is:
  - Do disease nodes have HAS_PHENOTYPE edges to patient CUIs (direct)?
  - Or only via HIERARCHY (indirect)?
  - For each true (disease, patient_cui) pair from test data, classify:
       DIRECT     : disease -[HAS_PHENOTYPE]-> patient_cui
       HIERARCHY  : disease -[HAS_PHENOTYPE]-> p' -[HIERARCHY]-> patient_cui
       UNREACHABLE: no path within 2 hops

Then: per-disease recall = |edges found| / |patient CUIs from true cases|.
"""
from __future__ import annotations
import sys, json, csv, ast, pickle
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH_V2 = MEDKG_ROOT / "kg" / "onlykg_graph_v2.pkl"
EVIDENCE_CUI = MEDKG_ROOT / "kg" / "ddxplus_evidence_cuis.json"


def main():
    print("=" * 70)
    print("only-KG edge-level vocabulary gap diagnostic")
    print("=" * 70)

    print("\n[1] Loading v2 graph...")
    with GRAPH_V2.open("rb") as f:
        G = pickle.load(f)
    print(f"    Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")

    print("\n[2] Loading DDXPlus mapping...")
    ev_cuis = json.load(open(EVIDENCE_CUI))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}

    print("\n[3] For each disease, build phenotype reachability map...")
    disease_direct = {}     # disease_cui -> set(patient_cuis directly connected)
    disease_hier = {}       # disease_cui -> set(patient_cuis via 1-hop HIERARCHY from direct)
    for d_cui in set(fr2cui.values()):
        if d_cui not in G:
            disease_direct[d_cui] = set(); disease_hier[d_cui] = set(); continue
        direct_phens = set()
        for _, p, edata in G.out_edges(d_cui, data=True):
            if edata.get("etype") == "HAS_PHENOTYPE":
                direct_phens.add(p)
        # Hierarchy expansion
        hier_phens = set()
        for p in direct_phens:
            for _, p2, edata in G.out_edges(p, data=True):
                if edata.get("etype") == "HIERARCHY":
                    hier_phens.add(p2)
            for p2, _, edata in G.in_edges(p, data=True):
                if edata.get("etype") == "HIERARCHY":
                    hier_phens.add(p2)
        disease_direct[d_cui] = direct_phens
        disease_hier[d_cui] = hier_phens - direct_phens

    print("\n[4] Walk test patients, classify each (true_disease, patient_cui) pair...")
    counter_direct = 0
    counter_hier = 0
    counter_unreach = 0
    unreach_by_disease = defaultdict(Counter)  # disease -> Counter(patient_cui -> miss_count)
    total_pairs_by_disease = Counter()
    direct_by_disease = Counter()
    hier_by_disease = Counter()
    n_patients = 0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n_patients >= 5000: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in disease_direct: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            patient_cuis = set()
            for ev in evs:
                patient_cuis.update(ev_cuis.get(ev.split("_@_")[0], []))
            true_name = cui2name.get(true_cui, "?")
            for pc in patient_cuis:
                total_pairs_by_disease[true_name] += 1
                if pc in disease_direct[true_cui]:
                    counter_direct += 1; direct_by_disease[true_name] += 1
                elif pc in disease_hier[true_cui]:
                    counter_hier += 1; hier_by_disease[true_name] += 1
                else:
                    counter_unreach += 1
                    unreach_by_disease[true_name][pc] += 1
            n_patients += 1

    total = counter_direct + counter_hier + counter_unreach
    print(f"\n[5] Pair-level reachability across {n_patients} patients ({total:,} (true_disease, patient_cui) pairs):")
    print(f"    DIRECT      (HAS_PHENOTYPE):     {counter_direct:,} ({100*counter_direct/total:.1f}%)")
    print(f"    HIERARCHY   (2-hop via HIERARCHY): {counter_hier:,} ({100*counter_hier/total:.1f}%)")
    print(f"    UNREACHABLE (gap):                {counter_unreach:,} ({100*counter_unreach/total:.1f}%)")

    print(f"\n[6] Per-disease breakdown (top 20 by sample count):")
    print(f"    {'Disease':<40s} {'Direct%':>8s} {'Hier%':>8s} {'Miss%':>8s}")
    for d, tot in total_pairs_by_disease.most_common(20):
        dr = 100*direct_by_disease[d]/tot
        hr = 100*hier_by_disease[d]/tot
        ms = 100 - dr - hr
        print(f"    {d:<40s} {dr:>7.1f}% {hr:>7.1f}% {ms:>7.1f}%")

    print(f"\n[7] Top unreachable patient CUIs (frequency across all diseases):")
    global_unreach = Counter()
    for d, cnts in unreach_by_disease.items():
        for pc, c in cnts.items():
            global_unreach[pc] += c
    for pc, c in global_unreach.most_common(20):
        name = G.nodes[pc].get("name", "?") if pc in G else "?"
        print(f"    {pc:12s} {name:<50s} miss_count={c}")

    print(f"\n[8] Interpretation:")
    print(f"    - DIRECT rate is upper-bound for graph-traversal precision")
    print(f"    - HIERARCHY rate is value-add of UMLS hierarchy (already in v2)")
    print(f"    - UNREACHABLE rate measures real semantic gap (LLM bridges this)")
    print(f"    - If UNREACHABLE is high but global_unreach top is concentrated:")
    print(f"      → multi-candidate linking or SY expansion at IE time may help")

    out = {
        "patients_analyzed": n_patients,
        "pairs_total": total,
        "direct_pct": 100*counter_direct/total,
        "hierarchy_pct": 100*counter_hier/total,
        "unreachable_pct": 100*counter_unreach/total,
        "top_unreachable_cuis": [(pc, G.nodes[pc].get("name", "?") if pc in G else "?", c) for pc, c in global_unreach.most_common(30)],
    }
    out_path = MEDKG_ROOT / "kg" / "onlykg_gap_diagnostic_v2.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n    Saved to {out_path}")


if __name__ == "__main__":
    main()
