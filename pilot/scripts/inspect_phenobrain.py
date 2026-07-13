#!/usr/bin/env python3
"""Inspector for PhenoBrain Zenodo subset (10.5281/zenodo.10774650).

Six de-identified rare-disease test sets. Each JSON is a list of cases,
each case = [phens, diseases] where:
  - phens    : list[str] of HPO IDs (e.g. "HP:0001225")
  - diseases : list[str] of disease IDs (OMIM/ORPHA/CCRD prefixed)
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

BASE = Path("/home/max/Graph-DDXPlus/data/external_benchmarks/phenobrain/Test cases")


def inspect_one(p: Path) -> None:
    d = json.load(open(p))
    if not isinstance(d, list):
        print(f"  {p.name}: UNEXPECTED structure"); return
    cases = [(c[0], c[1]) for c in d if isinstance(c, list) and len(c) >= 2]
    n_phens = [len(c[0]) for c in cases]
    n_dis = [len(c[1]) for c in cases]
    all_disease_ids = Counter()
    prefix_counts = Counter()
    for _, dis in cases:
        for did in dis:
            all_disease_ids[did] += 1
            pref = did.split(":")[0] if ":" in did else "NO_PREFIX"
            prefix_counts[pref] += 1
    print(f"\n--- {p.name} ---")
    print(f"  Cases                : {len(cases)}")
    print(f"  Phenotypes/case avg  : {sum(n_phens)/len(n_phens):.1f}  "
          f"(min={min(n_phens)}, max={max(n_phens)})")
    print(f"  Disease labels/case  : avg={sum(n_dis)/len(n_dis):.1f}")
    print(f"  Unique disease IDs   : {len(all_disease_ids)}")
    print(f"  ID prefix counts     : {dict(prefix_counts)}")
    print(f"  Evidence format      : HPO IDs (e.g. HP:0001225)")
    for i, (ph, di) in enumerate(cases[:3]):
        print(f"  Sample {i}: |hpo|={len(ph)}, hpos[:3]={ph[:3]}, dx={di}")


def main() -> None:
    print("=== PhenoBrain Zenodo subset (Mao & Huang 2024) ===")
    for fn in sorted(BASE.iterdir()):
        if fn.suffix == ".json":
            inspect_one(fn)


if __name__ == "__main__":
    main()
