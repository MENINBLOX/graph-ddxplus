#!/usr/bin/env python3
"""Inspector for MIMIC-RD initial-diff-diagnosis annotation set (RDMA repo).

Source: github.com/jhnwu3/RDMA, public_data/initial_diff_diagnosis_benchmark.json.
Provides patient-level annotations only (text requires MIMIC-IV credential).

Each top-level key = MIMIC patient HADM/admission id. Values include:
  - orpha_codes        : list of ORPHA / numeric IDs
  - disease_entities   : free-text disease strings
  - matched_phenotypes : list of dicts with hp_id (HPO term) + status
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

P = Path("/home/max/Graph-DDXPlus/data/external_benchmarks/mimic_rd/"
        "public_data/initial_diff_diagnosis_benchmark.json")


def main() -> None:
    print("=== MIMIC-RD initial-diff-diagnosis benchmark ===")
    d = json.load(open(P))
    print(f"  Patients (admission ids)   : {len(d)}")
    all_diseases = Counter()
    phen_counts = []
    phen_statuses = Counter()
    n_with_hp = 0
    for pid, rec in d.items():
        for did in rec.get("orpha_codes", []):
            all_diseases[did] += 1
        phs = rec.get("matched_phenotypes", [])
        hp_ids = [p.get("hp_id") for p in phs if p.get("hp_id")]
        phen_counts.append(len(hp_ids))
        if hp_ids:
            n_with_hp += 1
        for ph in phs:
            phen_statuses[ph.get("status", "?")] += 1
    print(f"  Unique disease IDs         : {len(all_diseases)}")
    print(f"  Patients with HPO-mapped   : {n_with_hp}")
    if phen_counts:
        print(f"  HPO terms/patient avg      : {sum(phen_counts)/len(phen_counts):.1f}  "
              f"(min={min(phen_counts)}, max={max(phen_counts)})")
    print(f"  Phenotype status mix       : {dict(phen_statuses.most_common(8))}")
    print(f"  Evidence format            : HPO IDs (via NER) + ORPHA disease codes")
    keys = list(d.keys())[:3]
    for k in keys:
        rec = d[k]
        hp_sample = [p.get("hp_id") for p in rec.get("matched_phenotypes", [])
                     if p.get("hp_id")][:5]
        print(f"  Sample patient {k}: orpha={rec.get('orpha_codes')}, "
              f"dx={rec.get('disease_entities')}, hpos={hp_sample}")


if __name__ == "__main__":
    main()
