#!/usr/bin/env python3
"""Build CUI index for KG features using scispaCy UMLS linker.

For each disease in disease_features_dual_by_cui.json, run scispaCy on each
feature phrase → extract CUIs (top-1 per entity, threshold ≥0.85).

Output: $MEDKG_ROOT/kg/disease_kg_cuis.json — {disease_cui: [feature_cui, ...]}
"""
from __future__ import annotations
import json, sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

KG_FILE = MEDKG_ROOT / "kg" / "disease_features_dual_by_cui.json"
OUT = MEDKG_ROOT / "kg" / "disease_kg_cuis.json"


def main():
    print("Loading scispaCy en_core_sci_lg + UMLS linker...")
    import spacy, scispacy
    from scispacy.linking import EntityLinker
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, "linker_name": "umls",
        "k": 1, "threshold": 0.85
    })

    print(f"Loading KG features from {KG_FILE} ...")
    fc = json.load(open(KG_FILE))
    print(f"  {len(fc):,} diseases")

    out = {}
    t0 = time.time()
    n_done = 0
    for disease_cui, feats in fc.items():
        cuis = set()
        for f in feats:
            text = f.get("phenotype", "")
            if not text: continue
            doc = nlp(text)
            for ent in doc.ents:
                if ent._.kb_ents:
                    cui = ent._.kb_ents[0][0]
                    cuis.add(cui)
        out[disease_cui] = sorted(cuis)
        n_done += 1
        if n_done % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {n_done}/{len(fc)} ({elapsed:.0f}s)  median CUIs: {sorted([len(v) for v in out.values()])[len(out)//2]}")

    OUT.write_text(json.dumps(out, ensure_ascii=False))
    print(f"\nSaved → {OUT}  (elapsed {time.time()-t0:.0f}s)")
    sizes = sorted(len(v) for v in out.values())
    print(f"  CUIs per disease: median={sizes[len(sizes)//2]}, p90={sizes[int(len(sizes)*0.9)]}, max={sizes[-1]}")


if __name__ == "__main__":
    main()
