#!/usr/bin/env python3
"""Build refined UMLS DISO seed list (~30-50K CUI) for medkg construction.

Filters applied:
1. Semantic type ∈ {T047, T191, T046, T037, T019, T048} — core disease types
2. Has mapping to at least one clinical vocabulary (SNOMED, ICD-10, MeSH, HPO, OMIM, ORPHA)
3. Has English preferred term (LAT='ENG', TS='P')
4. Active concept (SUPPRESS='N')

Output: $MEDKG_ROOT/seeds/umls_diso_refined.jsonl
"""
from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import UMLS_DIR, MEDKG_ROOT, ensure_dirs
import json
from collections import defaultdict

ensure_dirs()
SEEDS_DIR = MEDKG_ROOT / "seeds"
SEEDS_DIR.mkdir(parents=True, exist_ok=True)
OUT = SEEDS_DIR / "umls_diso_refined.jsonl"

CORE_DISEASE_TUI = {"T047", "T191", "T046", "T037", "T019", "T048"}
CLINICAL_SAB = {"SNOMEDCT_US", "ICD10", "ICD10CM", "MSH", "HPO", "OMIM", "ORPHANET",
                 "ICD9CM", "MEDLINEPLUS"}


def main():
    print(f"UMLS_DIR = {UMLS_DIR}")
    print(f"OUT      = {OUT}")
    print()

    mrsty = UMLS_DIR / "MRSTY.RRF"
    mrconso = UMLS_DIR / "MRCONSO.RRF"
    print(f"Reading {mrsty}...")
    diso_cuis = set()
    with mrsty.open() as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < 4: continue
            cui, tui = parts[0], parts[1]
            if tui in CORE_DISEASE_TUI:
                diso_cuis.add(cui)
    print(f"  Core DISO TUI CUIs: {len(diso_cuis):,}")
    print()

    print(f"Reading {mrconso}...")
    cui_clinical_sab = defaultdict(set)
    cui_pref_name = {}
    n_lines = 0
    with mrconso.open() as f:
        for line in f:
            n_lines += 1
            if n_lines % 1000000 == 0:
                print(f"  ... {n_lines:,} lines processed", flush=True)
            parts = line.split("|")
            if len(parts) < 17: continue
            cui = parts[0]
            if cui not in diso_cuis: continue
            lat = parts[1]
            ts  = parts[2]
            sab = parts[11]
            tty = parts[12]
            sup = parts[16]
            string_ = parts[14]
            if sup == "Y": continue   # suppressed
            if sab in CLINICAL_SAB:
                cui_clinical_sab[cui].add(sab)
            if lat == "ENG" and ts == "P" and cui not in cui_pref_name:
                cui_pref_name[cui] = string_

    print()
    print(f"  CUIs with clinical SAB mapping: {len(cui_clinical_sab):,}")
    print(f"  CUIs with English preferred name: {len(cui_pref_name):,}")
    print()

    # Final refined seed: clinical SAB + English preferred name + strong anchoring
    # Strong anchor = (SNOMED AND ICD10) OR (rare disease SAB) OR (multi-SAB ≥2 clinical)
    rare_sabs = {"OMIM", "ORPHANET", "HPO"}
    icd10_sabs = {"ICD10", "ICD10CM"}
    final_broad = []
    final_focused = []
    for cui in cui_clinical_sab:
        if cui not in cui_pref_name: continue
        sabs = cui_clinical_sab[cui]
        entry = {
            "cui": cui,
            "name": cui_pref_name[cui],
            "sabs": sorted(sabs),
        }
        final_broad.append(entry)
        # Focused: strong anchor
        has_snomed = "SNOMEDCT_US" in sabs
        has_icd10 = bool(sabs & icd10_sabs)
        has_rare = bool(sabs & rare_sabs)
        n_clinical_sab = len(sabs)
        if (has_snomed and has_icd10) or has_rare or n_clinical_sab >= 3:
            final_focused.append(entry)
    final_broad.sort(key=lambda x: x["cui"])
    final_focused.sort(key=lambda x: x["cui"])
    print(f"Broad refined DISO seed:    {len(final_broad):,} CUIs")
    print(f"Focused refined DISO seed:  {len(final_focused):,} CUIs (strong anchor)")

    with OUT.open("w") as out:
        for entry in final_broad:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nBroad   → {OUT}")

    OUT_FOCUSED = SEEDS_DIR / "umls_diso_focused.jsonl"
    with OUT_FOCUSED.open("w") as out:
        for entry in final_focused:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Focused → {OUT_FOCUSED}")

    # SAB distribution — focused subset
    sab_count = defaultdict(int)
    for e in final_focused:
        for sab in e["sabs"]:
            sab_count[sab] += 1
    print(f"\nFocused subset SAB distribution:")
    for sab, c in sorted(sab_count.items(), key=lambda x: -x[1]):
        print(f"  {sab:15s} {c:>8,}")


if __name__ == "__main__":
    main()
