#!/usr/bin/env python3
"""Map IE v3 evidence text -> UMLS CUI via scispaCy linker.

Reads /mnt/medkg/processed/edges_universal_v3.jsonl (text edges),
outputs /mnt/medkg/processed/edges_universal_v3_cui.jsonl (with CUI added).

For each phenotype text, runs scispaCy UMLS linker (threshold 0.80, k=1).
"""
from __future__ import annotations
import json, sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

IN = MEDKG_ROOT / "processed" / "edges_universal_v3.jsonl"
OUT = MEDKG_ROOT / "processed" / "edges_universal_v3_cui.jsonl"


def main():
    print("Loading scispaCy en_core_sci_lg + UMLS linker...", flush=True)
    import spacy, scispacy
    from scispacy.linking import EntityLinker
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, "linker_name": "umls",
        "k": 1, "threshold": 0.80
    })

    n_in = sum(1 for _ in open(IN))
    print(f"Mapping {n_in} edges...", flush=True)
    t0 = time.time()

    n_out = 0; n_no_cui = 0
    with open(OUT, "w") as fout, open(IN) as fin:
        for i, line in enumerate(fin):
            e = json.loads(line)
            text = e["phenotype"]
            doc = nlp(text)
            cuis = []
            for ent in doc.ents:
                if ent._.kb_ents:
                    cuis.append(ent._.kb_ents[0][0])
            cuis = list(dict.fromkeys(cuis))  # preserve order, dedupe
            if not cuis:
                n_no_cui += 1
                continue
            for c in cuis:
                e2 = dict(e)
                e2["evidence_cui"] = c
                fout.write(json.dumps(e2, ensure_ascii=False) + "\n")
                n_out += 1
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{n_in} edges, mapped={n_out}, no_cui={n_no_cui} ({elapsed:.0f}s)",
                      flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s", flush=True)
    print(f"  In edges: {n_in}", flush=True)
    print(f"  No CUI extracted: {n_no_cui} ({100*n_no_cui/n_in:.1f}%)", flush=True)
    print(f"  Out edges (with CUI, after multi-CUI fan-out): {n_out}", flush=True)
    print(f"  → {OUT}", flush=True)


if __name__ == "__main__":
    main()
