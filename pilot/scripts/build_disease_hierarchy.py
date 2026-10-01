"""Build a disease-CUI parent->children map from UMLS MRREL, restricted to the Disorders
semantic group. Used to resolve benchmark umbrella diseases (e.g. 'Migraine') to the
SPECIFIC subtype CUIs our rare-disease-inclusive KG actually indexes (Hemiplegic migraine,
Familial hemiplegic migraine, ...). This is UMLS structure, NOT benchmark data -> strict
zero-shot safe.

MRREL row: CUI1|AUI1|STYPE1|REL|CUI2|... ; REL=CHD means CUI2 is a child of CUI1.
We keep CHD (taxonomic child) and RN (narrower) where BOTH endpoints are disorders.
"""
from __future__ import annotations
import json, sys

STY = "pilot/cache/umls_maps/cui_sty.tsv"
MRREL = "data/umls_extracted/MRREL.RRF"
OUT = "pilot/cache/umls_maps/disease_children.json"

# Disorders semantic group (disease-level; excludes T184 sign/symptom, T033 finding)
DISO_TUI = {"T047", "T191", "T019", "T020", "T037", "T048", "T049", "T050", "T046", "T190"}


def main() -> None:
    diso = set()
    for line in open(STY, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[1] in DISO_TUI:
            diso.add(p[0])
    print(f"disorder CUIs: {len(diso)}", flush=True)

    children: dict[str, set] = {}
    kept = 0
    for i, line in enumerate(open(MRREL, encoding="utf-8")):
        c = line.split("|", 5)
        if len(c) < 5:
            continue
        cui1, rel, cui2 = c[0], c[3], c[4]
        if rel not in ("CHD", "RN"):
            continue
        if cui1 in diso and cui2 in diso and cui1 != cui2:
            children.setdefault(cui1, set()).add(cui2)
            kept += 1
        if (i + 1) % 500000 == 0:
            print(f"  {i+1} rows, kept {kept}", flush=True)

    out = {k: sorted(v) for k, v in children.items()}
    json.dump(out, open(OUT, "w"))
    print(f"parents with children: {len(out)}  edges: {kept}\n-> {OUT}")


if __name__ == "__main__":
    main()
