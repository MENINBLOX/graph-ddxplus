#!/usr/bin/env python3
"""Map IE v3 FULL evidence text -> UMLS CUI via scispaCy linker. Parallel-able by --part/--of."""
from __future__ import annotations
import json, sys, warnings, time, argparse
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default=str(MEDKG_ROOT / "processed" / "edges_v3_full_pubmed.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args()

    print(f"[part {args.part}/{args.of}] Loading scispaCy...", flush=True)
    import spacy, scispacy
    from scispacy.linking import EntityLinker
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, "linker_name": "umls",
        "k": 1, "threshold": 0.80
    })

    # Cache phenotype text → CUIs (many duplicates)
    phen_cache = {}
    t0 = time.time()
    n_in = n_out = n_no_cui = 0
    n_cached = 0

    with open(args.out, "w") as fout, open(args.in_path) as fin:
        for i, line in enumerate(fin):
            if i % args.of != args.part: continue
            e = json.loads(line)
            text = e["phenotype"]
            if text in phen_cache:
                cuis = phen_cache[text]
                n_cached += 1
            else:
                doc = nlp(text)
                cuis = list(dict.fromkeys(ent._.kb_ents[0][0] for ent in doc.ents if ent._.kb_ents))
                phen_cache[text] = cuis
            n_in += 1
            if not cuis:
                n_no_cui += 1
                continue
            for c in cuis:
                e2 = dict(e); e2["evidence_cui"] = c
                fout.write(json.dumps(e2, ensure_ascii=False) + "\n")
                n_out += 1
            if n_in % 10000 == 0:
                rate = n_in / max(time.time() - t0, 1)
                print(f"  [part {args.part}] {n_in} processed, "
                      f"cached={n_cached} ({100*n_cached/n_in:.0f}%), "
                      f"mapped={n_out}, no_cui={n_no_cui}, "
                      f"rate={rate:.0f}/s", flush=True)
    print(f"Done. in={n_in}, no_cui={n_no_cui}, out={n_out}, cache_hits={n_cached}",
          flush=True)


if __name__ == "__main__":
    main()
