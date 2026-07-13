#!/usr/bin/env python3
"""Map IE v3 pilot evidence text -> UMLS CUI via scispaCy linker."""
from __future__ import annotations
import json, sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

IN = MEDKG_ROOT / "processed" / "edges_universal_v3_pilot1k.jsonl"
OUT = MEDKG_ROOT / "processed" / "edges_universal_v3_pilot1k_cui.jsonl"


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
            doc = nlp(e["phenotype"])
            cuis = list(dict.fromkeys(ent._.kb_ents[0][0] for ent in doc.ents if ent._.kb_ents))
            if not cuis:
                n_no_cui += 1; continue
            for c in cuis:
                e2 = dict(e); e2["evidence_cui"] = c
                fout.write(json.dumps(e2, ensure_ascii=False) + "\n")
                n_out += 1
            if (i+1) % 1000 == 0:
                print(f"  {i+1}/{n_in} mapped={n_out} no_cui={n_no_cui} ({time.time()-t0:.0f}s)",
                      flush=True)
    print(f"Done in {time.time()-t0:.0f}s. In={n_in}, no_cui={n_no_cui}, out={n_out}",
          flush=True)


if __name__ == "__main__":
    main()
