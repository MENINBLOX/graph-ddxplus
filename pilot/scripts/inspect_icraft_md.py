#!/usr/bin/env python3
"""Inspector for raw iCRAFT-MD CSV release (rajpurkarlab/craft-md).

Two CSVs:
  - usmle_and_derm_dataset.csv : MCQ-form vignettes with 4 choices
  - nejm_imagechallenge_dataset.csv : NEJM image URLs only (no text)

Evidence format is free-text vignettes; no HPO/CUI codes. Adaptation to
the KG requires either text->CUI extraction (LLM IE) or use of the
MEDDxAgent-derived `facts` field instead.
"""
from __future__ import annotations
import csv
from collections import Counter
from pathlib import Path

BASE = Path("/home/max/Graph-DDXPlus/data/external_benchmarks/icraft_md/data")


def inspect_usmle_derm() -> None:
    print("\n=== iCRAFT-MD :: usmle_and_derm_dataset.csv ===")
    p = BASE / "usmle_and_derm_dataset.csv"
    with p.open() as f:
        rows = list(csv.DictReader(f))
    diseases = Counter(r["answer"] for r in rows)
    datasets = Counter(r.get("dataset", "?") for r in rows)
    cats = Counter(r.get("category", "?") for r in rows)
    print(f"  Cases                : {len(rows)}")
    print(f"  Unique answers (dx)  : {len(diseases)}")
    print(f"  Source datasets      : {dict(datasets)}")
    print(f"  Categories           : {dict(cats.most_common(10))}")
    print(f"  Evidence format      : free-text case vignette")
    for i, r in enumerate(rows[:3]):
        v = r["case_vignette"][:160].replace("\n", " ")
        print(f"  Sample {i}: dx={r['answer']!r}; ds={r['dataset']}; "
              f"vignette={v}...")


def inspect_nejm() -> None:
    print("\n=== iCRAFT-MD :: nejm_imagechallenge_dataset.csv ===")
    p = BASE / "nejm_imagechallenge_dataset.csv"
    with p.open() as f:
        rows = list(csv.DictReader(f))
    print(f"  Cases                : {len(rows)}")
    print(f"  Columns              : {list(rows[0].keys()) if rows else []}")
    print(f"  Evidence format      : IMAGE URLs only (no text labels)")
    if rows:
        print(f"  Sample 0: case_id={rows[0].get('case_id')}; "
              f"url={rows[0].get('case_url','?')[:80]}")


def main() -> None:
    inspect_usmle_derm()
    inspect_nejm()


if __name__ == "__main__":
    main()
