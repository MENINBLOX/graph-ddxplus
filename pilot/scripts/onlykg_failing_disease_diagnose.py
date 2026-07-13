#!/usr/bin/env python3
"""Diagnose why specific diseases have 100% failure rate in only-KG.

For each failing disease:
  - Check disease node existence in KG
  - Count HAS_PHENOTYPE edges (out-degree)
  - List top phenotypes (by weight)
  - Check intersection with patient evidence CUIs (SYMPTOM+ANATOMY)
  - Identify if low-recall is due to (a) sparse KG content or (b) vocabulary mismatch
"""
from __future__ import annotations
import sys, json, csv, ast, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v2.pkl"
EVIDENCE_CUI = MEDKG_ROOT / "kg" / "ddxplus_evidence_cuis.json"
SEM_CLASS = MEDKG_ROOT / "kg" / "evidence_cui_semantic_class.json"

FAILING = [
    "Viral pharyngitis", "Anemia", "HIV (initial infection)",
    "Localized edema", "Pulmonary embolism", "Acute otitis media",
    "Influenza", "Pneumonia", "Bronchitis",
]


def main():
    with GRAPH.open("rb") as f:
        G = pickle.load(f)
    ev_cuis = json.load(open(EVIDENCE_CUI))
    sem_class = json.load(open(SEM_CLASS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    # Build english_name → cui (using icd_map keys = English)
    name2cui = {dn: icd_map[dn]["cui"] for dn in icd_map}

    # Collect typical patient evidence CUIs per disease (from test patients)
    print("Building typical patient evidence per failing disease...")
    disease_typical_cuis = {d: Counter() for d in FAILING}
    cui_filter = {c for c, info in sem_class.items()
                  if info.get("category") in {"SYMPTOM", "ANATOMY"}}
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            disease = None
            for dn, info in cond.items():
                if info.get("cond-name-fr") == row["PATHOLOGY"]:
                    disease = dn; break
            if disease not in FAILING: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            for ev in evs:
                base = ev.split("_@_")[0]
                for c in ev_cuis.get(base, []):
                    if c in cui_filter:
                        disease_typical_cuis[disease][c] += 1

    print("\n" + "=" * 80)
    print("Per-disease KG content diagnostic")
    print("=" * 80)
    for d_name in FAILING:
        d_cui = name2cui.get(d_name)
        print(f"\n  Disease: {d_name}  (CUI: {d_cui})")
        if d_cui not in G:
            print(f"    ❌ NOT IN KG")
            continue
        phens = [(p, edata.get("weight", 0))
                 for _, p, edata in G.out_edges(d_cui, data=True)
                 if edata.get("etype") == "HAS_PHENOTYPE"]
        phens.sort(key=lambda x: -x[1])
        print(f"    Phenotype edges: {len(phens)}")
        if not phens:
            print(f"    ❌ ZERO HAS_PHENOTYPE edges")
            continue
        print(f"    Top 10 KG phenotypes (by weight):")
        for p, w in phens[:10]:
            name = G.nodes[p].get("name", "?")
            print(f"      [{w:6.2f}] {name:<40s} {p}")
        # Patient evidence overlap
        typical = disease_typical_cuis[d_name]
        kg_phen_set = set(p for p, _ in phens)
        overlap = [c for c in typical if c in kg_phen_set]
        # Hierarchy bridge
        hier_overlap = set()
        for p_direct in kg_phen_set:
            for _, p2, edata in G.out_edges(p_direct, data=True):
                if edata.get("etype") == "HIERARCHY" and p2 in typical:
                    hier_overlap.add(p2)
        print(f"    Patient typical CUIs: {len(typical)}")
        print(f"    Direct overlap:     {len(overlap)} ({100*len(overlap)/max(len(typical),1):.1f}%)")
        print(f"    Hierarchy overlap:  {len(hier_overlap)} ({100*len(hier_overlap)/max(len(typical),1):.1f}%)")
        if not overlap and not hier_overlap:
            print(f"    Top patient CUIs NOT IN KG phenotypes:")
            for c, cnt in typical.most_common(5):
                name = sem_class.get(c, {}).get("name", c)
                cat = sem_class.get(c, {}).get("category", "?")
                print(f"      {c:12s} {name:<35s} [{cat}]  freq={cnt}")


if __name__ == "__main__":
    main()
