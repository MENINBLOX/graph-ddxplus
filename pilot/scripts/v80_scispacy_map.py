#!/usr/bin/env python3
"""v80 step 2 — Map free-form phenotype text to UMLS CUI via scispaCy.

Input: v80_freeform_ie_full.jsonl (disease → {phen_name: {prob, n_seen}})
Output: v80_cui_edges.jsonl (disease_cui → {phen_cui: prob})

Disease name → CUI uses:
- DDXPlus: data/ddxplus/disease_icd10_cui_mapping.json
- SymCat:  data/symcat/disease_umls_mapping.json
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
from collections import defaultdict


def load_disease_to_cui():
    """Union of DDXPlus + SymCat disease → CUI mappings."""
    name_to_cui = {}
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f:
        icd = json.load(f)
    for dn, info in icd.items():
        c = info.get("cui")
        if c: name_to_cui[dn] = c
    sym_dis = json.load(open("data/symcat/disease_umls_mapping.json"))["mapping"]
    for dn, info in sym_dis.items():
        c = info.get("umls_cui")
        if c and dn not in name_to_cui:
            name_to_cui[dn] = c
    return name_to_cui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--linker_threshold", type=float, default=0.80)
    ap.add_argument("--keep_top_per_phrase", type=int, default=1)
    args = ap.parse_args()

    name_to_cui = load_disease_to_cui()
    print(f"Loaded disease→CUI: {len(name_to_cui)}", flush=True)

    print("Loading scispaCy en_core_sci_lg + UMLS linker...", flush=True)
    import spacy
    from scispacy.linking import EntityLinker
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, "linker_name": "umls",
        "k": 3, "threshold": args.linker_threshold,
    })

    records = []
    with open(args.ie_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} disease records", flush=True)

    # Aggregate phen_name across all diseases first (one nlp.pipe call)
    all_phens = set()
    for r in records:
        all_phens.update(r["phenotypes"].keys())
    all_phens = list(all_phens)
    print(f"Unique phenotype phrases: {len(all_phens)}", flush=True)

    # Run scispaCy
    print("Running scispaCy NER + linker...", flush=True)
    phen_to_cui = {}
    n_mapped = 0
    for i, doc in enumerate(nlp.pipe(all_phens, batch_size=64)):
        phrase = all_phens[i]
        cuis = []
        for ent in doc.ents:
            for cui, score in ent._.kb_ents[:args.keep_top_per_phrase]:
                cuis.append((cui, score))
        # If no entity, try linking the whole phrase as one entity by re-NER
        if not cuis:
            # last resort: link the entire phrase against linker
            pass
        if cuis:
            # Take best
            cuis.sort(key=lambda x: -x[1])
            phen_to_cui[phrase] = cuis[0][0]
            n_mapped += 1
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(all_phens)} ({100*n_mapped/(i+1):.1f}% mapped)", flush=True)
    print(f"Mapped {n_mapped}/{len(all_phens)} phrases to CUI", flush=True)

    # Output: per disease_cui → {phen_cui: prob}
    n_records = 0; n_edges = 0
    unmapped_diseases = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            dname = r["disease"]
            dcui = name_to_cui.get(dname)
            if not dcui:
                unmapped_diseases += 1
                continue
            edges = {}
            for phen_name, info in r["phenotypes"].items():
                phen_cui = phen_to_cui.get(phen_name)
                if not phen_cui: continue
                # If multiple phrases map to same CUI, take max prob
                p = info["prob"]
                if phen_cui in edges:
                    edges[phen_cui] = max(edges[phen_cui], p)
                else:
                    edges[phen_cui] = p
            if not edges: continue
            f.write(json.dumps({
                "disease": dname, "dcui": dcui, "source": r.get("source",""),
                "edges": edges,
            }) + "\n")
            n_records += 1
            n_edges += len(edges)
    print(f"\nSaved {n_records} disease records → {args.out}")
    print(f"  Total edges: {n_edges} (avg {n_edges/max(n_records,1):.1f}/disease)")
    print(f"  Unmapped diseases (no disease CUI): {unmapped_diseases}")


if __name__ == "__main__":
    main()
