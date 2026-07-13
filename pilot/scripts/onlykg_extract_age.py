#!/usr/bin/env python3
"""Extract age preferences for each of 49 DDXPlus diseases from raw KG text.

For each disease, scan phenotype text for age mentions and build a soft
age preference distribution (infant/child/adult/elderly).
"""
from __future__ import annotations
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

OUT = MEDKG_ROOT / "kg" / "disease_age_preferences.json"


AGE_PATTERNS = {
    "infant": [r"\binfan(t|ts|cy)\b", r"\bneonate?s?\b", r"\bnewborn", r"\bunder\s+2\s+years?", r"\b< ?2\s+years?"],
    "child":  [r"\bchild(ren|hood)?\b", r"\bpediatric\b", r"\btoddler", r"\badolescen(t|ts|ce)\b",
               r"\bunder\s+(?:5|10|12|15|18)\s+years?", r"\bschool[- ]age"],
    "adult":  [r"\badult", r"\b(young\s+adult|middle[- ]aged)\b",
               r"\b(?:18|20|30|40|50)\s+to\s+\d+\s+years?",
               r"\b(?:18|20|25|30|35|40|45|50)\s+years?\b"],
    "elderly":[r"\belderly\b", r"\bgeriatric\b", r"\baged?\b", r"\bsenior", r"\bold[- ]age\b",
               r"\bover\s+(?:60|65|70|75|80)\s+years?", r"\b> ?(?:60|65|70)\s+years?"],
}


def parse_age_from_phens(phens):
    """Return dict: age_category → count of mentions."""
    counts = {k: 0 for k in AGE_PATTERNS}
    for p in phens:
        text = p.get("phenotype", "").lower()
        for cat, pats in AGE_PATTERNS.items():
            for pat in pats:
                if re.search(pat, text):
                    counts[cat] += 1
                    break
    return counts


def main():
    features = json.load(open(MEDKG_ROOT / "kg" / "disease_features_dual_v2_by_cui.json"))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    ddxplus_cuis = {info["cui"]: dn for dn, info in icd.items() if "cui" in info}

    out = {}
    for cui, name in ddxplus_cuis.items():
        phens = features.get(cui, [])
        counts = parse_age_from_phens(phens)
        total = sum(counts.values())
        # Normalize to preference distribution
        if total > 0:
            prefs = {k: v/total for k, v in counts.items()}
        else:
            # Neutral (uniform)
            prefs = {k: 0.25 for k in AGE_PATTERNS}
        out[cui] = {"name": name, "age_pref": prefs, "raw_counts": counts}

    # Show non-neutral
    print("Diseases with strong age preference:")
    for cui, info in out.items():
        prefs = info["age_pref"]
        max_pref = max(prefs.values())
        if max_pref > 0.4:
            top = sorted(prefs.items(), key=lambda x: -x[1])[:2]
            print(f"  {info['name']:35s} {top[0][0]}={top[0][1]:.2f} (counts={info['raw_counts']})")

    print(f"\nSaving to {OUT}")
    with OUT.open("w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
