"""child-CUI -> parent-CUIs map over the finding/symptom semantic types, from UMLS MRREL.
Used to bridge phenotype granularity between benchmark evidence (often specific HPO terms)
and KG phenotypes (whatever granularity the IE produced). UMLS structure -> zero-shot safe.

MRREL row CUI1|.|.|REL|CUI2 ; REL=PAR means CUI2 is a parent of CUI1; RB = broader.
"""
from __future__ import annotations
import json

STY = "pilot/cache/umls_maps/cui_sty.tsv"
MRREL = "data/umls_extracted/MRREL.RRF"
OUT = "pilot/cache/umls_maps/phenotype_parents.json"

PHENO_TUI = {"T184", "T033", "T046", "T047", "T048", "T049", "T050", "T037",
             "T019", "T020", "T190", "T191"}


def main():
    ph = set()
    for line in open(STY, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[1] in PHENO_TUI:
            ph.add(p[0])
    print(f"phenotype CUIs: {len(ph)}", flush=True)

    parents = {}
    kept = 0
    for i, line in enumerate(open(MRREL, encoding="utf-8")):
        c = line.split("|", 5)
        if len(c) < 5:
            continue
        cui1, rel, cui2 = c[0], c[3], c[4]
        if rel not in ("PAR", "RB"):
            continue
        if cui1 in ph and cui2 in ph and cui1 != cui2:
            parents.setdefault(cui1, set()).add(cui2)
            kept += 1
        if (i + 1) % 500000 == 0:
            print(f"  {i+1} rows, kept {kept}", flush=True)

    json.dump({k: sorted(v) for k, v in parents.items()}, open(OUT, "w"))
    print(f"children with parents: {len(parents)}  edges: {kept}\n-> {OUT}")


if __name__ == "__main__":
    main()
