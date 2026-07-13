#!/usr/bin/env python3
"""Classify DDXPlus evidence CUIs by UMLS Semantic Type.

Goal: separate (a) symptom CUIs (matchable in only-KG) from (b) context CUIs
(anatomical/temporal/geographic — questionnaire metadata that academic literature
does not co-mention with diseases).

This classification is benchmark-agnostic: it uses only UMLS MRSTY (universal
semantic types), not DDXPlus-specific information.

Output: /mnt/medkg/kg/evidence_cui_semantic_class.json
  {
    cui: { tui: [...], category: "SYMPTOM"|"ANATOMY"|"GEO_TEMP"|"DEMOGRAPHIC"|"OTHER" }
  }
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

EVIDENCE_CUI = MEDKG_ROOT / "kg" / "ddxplus_evidence_cuis.json"
OUT = MEDKG_ROOT / "kg" / "evidence_cui_semantic_class.json"

# UMLS Semantic Types
SYMPTOM_TUIS = {
    "T184",  # Sign or Symptom
    "T033",  # Finding
    "T046",  # Pathologic Function
    "T047",  # Disease or Syndrome
    "T048",  # Mental or Behavioral Dysfunction
    "T191",  # Neoplastic Process
    "T037",  # Injury or Poisoning
    "T039",  # Physiologic Function
    "T067",  # Phenomenon or Process
}
ANATOMY_TUIS = {
    "T023",  # Body Part, Organ, or Organ Component
    "T024",  # Tissue
    "T029",  # Body Location or Region
    "T030",  # Body Space or Junction
    "T031",  # Body Substance
    "T025",  # Cell
    "T026",  # Cell Component
    "T017",  # Anatomical Structure
    "T018",  # Embryonic Structure
    "T021",  # Fully Formed Anatomical Structure
}
GEO_TEMP_TUIS = {
    "T079",  # Temporal Concept
    "T082",  # Spatial Concept
    "T083",  # Geographic Area
    "T080",  # Qualitative Concept (often spatial)
    "T081",  # Quantitative Concept
}
DEMOGRAPHIC_TUIS = {
    "T098",  # Population Group
    "T100",  # Age Group
    "T099",  # Family Group
    "T101",  # Patient or Disabled Group
    "T102",  # Group Attribute
}


def classify(tuis: set) -> str:
    if tuis & SYMPTOM_TUIS: return "SYMPTOM"
    if tuis & ANATOMY_TUIS: return "ANATOMY"
    if tuis & GEO_TEMP_TUIS: return "GEO_TEMP"
    if tuis & DEMOGRAPHIC_TUIS: return "DEMOGRAPHIC"
    return "OTHER"


def main():
    print("Loading DDXPlus evidence CUIs...")
    ev_cuis = json.load(open(EVIDENCE_CUI))
    all_cuis = set()
    for cs in ev_cuis.values():
        all_cuis.update(cs)
    print(f"  Unique CUIs: {len(all_cuis)}")

    print("Loading MRSTY (semantic types)...")
    cui2tuis = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            parts = line.split("|")
            cui, tui = parts[0], parts[1]
            if cui in all_cuis:
                cui2tuis[cui].add(tui)
    print(f"  CUIs with TUI info: {len(cui2tuis)}/{len(all_cuis)}")

    # Get names for reporting
    print("Loading MRCONSO names...")
    cui2name = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.split("|")
            cui = parts[0]
            if cui not in all_cuis or cui in cui2name: continue
            if parts[1] != "ENG": continue
            cui2name[cui] = parts[14]

    cat_counts = Counter()
    classification = {}
    for cui in all_cuis:
        tuis = cui2tuis.get(cui, set())
        cat = classify(tuis)
        cat_counts[cat] += 1
        classification[cui] = {
            "name": cui2name.get(cui, cui),
            "tuis": sorted(tuis),
            "category": cat,
        }

    print("\n=== Category distribution (CUIs) ===")
    for cat, c in cat_counts.most_common():
        print(f"  {cat:15s} {c:4d} ({100*c/len(all_cuis):.1f}%)")

    # Show samples per category
    for cat in ["SYMPTOM", "ANATOMY", "GEO_TEMP", "DEMOGRAPHIC", "OTHER"]:
        samples = [(c, cls) for c, cls in classification.items() if cls["category"] == cat][:6]
        print(f"\n  Sample {cat}:")
        for c, cls in samples:
            print(f"    {c}  {cls['name']:<45s}  TUIs={cls['tuis']}")

    # Per-evidence category breakdown
    print("\n=== Per-evidence-question category counts ===")
    ev_cat_dist = Counter()
    for ev, cs in ev_cuis.items():
        cats = [classification[c]["category"] for c in cs if c in classification]
        if not cats: ev_cat_dist["NO_CUI"] += 1; continue
        # majority category
        majority = Counter(cats).most_common(1)[0][0]
        ev_cat_dist[majority] += 1
    print(f"  Evidence questions ({len(ev_cuis)} total):")
    for cat, c in ev_cat_dist.most_common():
        print(f"    {cat:15s} {c} questions")

    print(f"\nSaving classification to {OUT}")
    with OUT.open("w") as f:
        json.dump(classification, f, indent=2)


if __name__ == "__main__":
    main()
