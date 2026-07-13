#!/usr/bin/env python3
"""For 10 sparse DDXPlus diseases, identify which Q-CUIs are missing.

For each disease in priority list:
  - Current Q∩phens
  - Common patient evidence CUIs (from test patients) NOT in disease's phens
  - These are the gaps to target with re-IE
"""
from __future__ import annotations
import sys, json, csv, ast, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v10.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"

SPARSE_DISEASES = {
    "Possible NSTEMI / STEMI",
    "Localized edema",
    "Spontaneous rib fracture",
    "Acute COPD exacerbation / infection",
    "Acute dystonic reactions",
    "Pancreatic neoplasm",
    "Inguinal hernia",
    "Boerhaave",
    "PSVT",
    "Whooping cough",
}


def main():
    G = pickle.load(open(GRAPH, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"]
              for dn,info in cond.items() if dn in icd}
    cui2name = {icd[dn]["cui"]: dn for dn in icd}

    # Q universe
    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)

    # CUI names for readability
    cui_names = {}
    for n, attrs in G.nodes(data=True):
        cui_names[n] = attrs.get("name", n)

    # For each sparse disease: typical patient CUIs from test data
    sparse_cuis = {dn: icd[dn]["cui"] for dn in icd if dn in SPARSE_DISEASES}
    name_to_fr = {info.get("cond-name-fr",""): dn for dn, info in cond.items() if dn in SPARSE_DISEASES}

    patient_cuis_per_disease = {dn: Counter() for dn in SPARSE_DISEASES}
    sample_count = {dn: 0 for dn in SPARSE_DISEASES}
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            fr_name = row["PATHOLOGY"]
            if fr_name not in name_to_fr: continue
            dn = name_to_fr[fr_name]
            if sample_count[dn] >= 100: continue
            sample_count[dn] += 1
            evs = ast.literal_eval(row["EVIDENCES"])
            for ev in evs:
                if "_@_" in ev:
                    base, val = ev.split("_@_", 1)
                    m = value_cuis.get(base, {})
                    for c in m.get("_question", []): patient_cuis_per_disease[dn][c] += 1
                    for c in m.get(val, []): patient_cuis_per_disease[dn][c] += 1
                else:
                    m = value_cuis.get(ev, {})
                    for c in m.get("_question", []): patient_cuis_per_disease[dn][c] += 1

    print("=" * 80)
    print("Per-disease gap analysis: common patient CUIs NOT in disease's phens")
    print("=" * 80)
    gap_summary = {}
    for dn in sorted(SPARSE_DISEASES):
        dcui = sparse_cuis.get(dn)
        if not dcui or dcui not in G: continue
        # Current disease phens
        d_phens = {p for _, p, e in G.out_edges(dcui, data=True) if e.get("etype")=="HAS_PHENOTYPE"}
        d_q_phens = d_phens & Q
        # Patient typical CUIs
        patient_typical = patient_cuis_per_disease[dn]
        # Gap = patient CUIs (high freq) NOT in disease's phens
        gap = []
        for c, freq in patient_typical.most_common(20):
            if c not in d_phens and c in Q:
                gap.append((c, freq, cui_names.get(c, c)))
        print(f"\n{dn}:")
        print(f"  Disease CUI: {dcui}, current Q∩phens: {len(d_q_phens)}")
        print(f"  Top gap CUIs (in patient typical, NOT in disease's KG):")
        for c, freq, name in gap[:15]:
            print(f"    {c}  {name:<40s} appears in {freq}/100 patients")
        gap_summary[dn] = {
            "dcui": dcui,
            "current_q_phens": len(d_q_phens),
            "n_samples": sample_count[dn],
            "gap_cuis": [{"cui": c, "name": cui_names.get(c, c), "freq": freq} for c, freq, _ in gap[:15]]
        }

    # Save
    out = Path("pilot/results/sparse_disease_gaps.json")
    with out.open("w") as f:
        json.dump(gap_summary, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
