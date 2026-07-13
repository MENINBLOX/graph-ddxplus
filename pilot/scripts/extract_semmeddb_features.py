#!/usr/bin/env python3
"""Extract SemMedDB AFFECTS/CAUSES/ASSOCIATED_WITH features for DDXPlus 49 diseases.

Output: pilot/results/semmeddb_disease_features.json — dict of disease_cui -> top-K feature CUIs (sorted by count).
"""
from __future__ import annotations
import gzip, csv, json
from collections import Counter, defaultdict
from pathlib import Path

UMLS_DIR = Path("data/umls_extracted")

# Relations indicating disease has symptoms / produces effects
SYMPTOM_RELATIONS = {
    "AFFECTS",          # disease affects body system
    "CAUSES",           # disease causes finding
    "ASSOCIATED_WITH",  # general association
    "COEXISTS_WITH",    # co-occurring features
    "MANIFESTATION_OF", # finding is manifestation of disease
    "PRODUCES",         # disease produces something
    "PRECEDES",         # may precede
}


def main():
    print("="*80)
    print("Extract SemMedDB features for DDXPlus diseases")
    print("="*80)

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    diseases = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        diseases[dn] = icd_map[dn]["cui"]
    dcs = set(diseases.values())
    print(f"DDXPlus diseases: {len(dcs)}")

    # Scan SemMedDB
    rel_features = defaultdict(Counter)  # disease_cui -> Counter(feature_cui)
    forward = 0
    reverse = 0
    n = 0
    with gzip.open('data/semmeddb/semmedVER43_2024_R_PREDICATION.csv.gz', 'rt', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 10: continue
            n += 1
            if n % 5000000 == 0: print(f"  scanned {n}M lines...")
            pred = row[3]
            if pred not in SYMPTOM_RELATIONS: continue
            subj_cui = row[4]
            obj_cui = row[8] if len(row) > 8 else None
            if not obj_cui: continue
            # Forward: disease -> feature
            if subj_cui in dcs and obj_cui != subj_cui:
                rel_features[subj_cui][obj_cui] += 1
                forward += 1
            # Reverse: feature -> disease (e.g., MANIFESTATION_OF: symptom -> disease)
            if obj_cui in dcs and subj_cui != obj_cui:
                rel_features[obj_cui][subj_cui] += 1
                reverse += 1

    print(f"\nTotal scanned: {n}")
    print(f"Disease-feature edges: forward={forward}, reverse={reverse}")
    print(f"Diseases with features: {sum(1 for v in rel_features.values() if v)}/{len(dcs)}")

    # Save
    out = {dc: dict(rel_features[dc].most_common(30)) for dc in dcs}
    with open("pilot/results/semmeddb_disease_features.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to pilot/results/semmeddb_disease_features.json")

    # Show sample for problem diseases
    cp = {}
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG" and p[2] == "P" and p[0] not in cp:
                cp[p[0]] = p[14].strip()

    cui2name = {info["cui"]: dn for dn, info in icd_map.items()}
    for dc in dcs:
        feats = rel_features[dc]
        if not feats: continue
        name = cui2name.get(dc, dc)
        if name not in ['URTI', 'Influenza', 'Pneumonia', 'Sarcoidosis', 'PSVT', 'Pericarditis']:
            continue
        print(f"\n{name} ({dc}): {sum(feats.values())} edges, top:")
        for cui, cnt in feats.most_common(10):
            print(f"  {cnt:4d}  {cp.get(cui, cui)[:50]}")


if __name__ == "__main__":
    main()
