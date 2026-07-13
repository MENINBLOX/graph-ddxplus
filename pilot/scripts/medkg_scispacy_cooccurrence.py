#!/usr/bin/env python3
"""Same-document co-occurrence IE via scispaCy NER + UMLS linker.

For each PubMed abstract anchored to a DDXPlus disease:
1. Run scispaCy `en_core_sci_lg` NER → extract entities
2. UMLS linker → map each entity to CUI
3. Filter to medical semantic types (DISO, PHEN, BODY, FNDG, SOSY, etc.)
4. All entity pairs in the same abstract → co-occurrence edge

Output: /windows/data/medkg/processed/edges_cooccurrence_scispacy.jsonl
  Each line: {"src_cui", "dst_cui", "pmid", "src_disease_anchor"}

Universal: scispaCy + UMLS only, no DDXPlus-specific prompts.
CPU-only — runs alongside vLLM jobs.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

INPUT = Path(os.environ.get('COOCCUR_INPUT', str(MEDKG_ROOT / 'pubmed' / 'ddxplus_anchored_abstracts.jsonl')))
OUT = Path(os.environ.get('COOCCUR_OUTPUT', str(MEDKG_ROOT / 'processed' / 'edges_cooccurrence_scispacy.jsonl')))

# Acceptable semantic types (broad medical, exclude generic body parts and procedures)
# T184=Sign or Symptom, T033=Finding (broad), T047=Disease or Syndrome, T046=Pathologic Function,
# T037=Injury or Poisoning, T191=Neoplastic Process, T048=Mental or Behavioral Dysfunction,
# T190=Anatomical Abnormality, T019=Congenital Abnormality
ACCEPT_TUI = {
    "T184", "T033", "T034",  # finding, lab
    "T047", "T046", "T037", "T191", "T048", "T190", "T019",  # disease/disorder family
    "T020", "T024",  # cell pathology
    "T029", "T022", "T030", "T031",  # body cavity / tissue
}


def main():
    records = []
    with INPUT.open() as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records):,} abstracts", flush=True)

    print("Loading scispaCy en_core_sci_lg + UMLS linker (CPU)...", flush=True)
    import spacy
    from scispacy.linking import EntityLinker  # noqa: F401
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
        "k": 3,
        "threshold": 0.85,
        "filter_for_definitions": False,
    })
    linker = nlp.get_pipe("scispacy_linker")
    print("  loaded.", flush=True)

    t0 = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_edges = 0
    with OUT.open('w') as out:
        for i, r in enumerate(records):
            anchor = r['cui']; pmid = r['pmid']
            doc = nlp(r['text'])
            cuis = set()
            for ent in doc.ents:
                if not ent._.kb_ents: continue
                cui, score = ent._.kb_ents[0]
                if score < 0.85: continue
                # Filter by semantic type
                kb_entry = linker.kb.cui_to_entity.get(cui)
                if not kb_entry: continue
                tuis = set(kb_entry.types)
                if not (tuis & ACCEPT_TUI): continue
                cuis.add(cui)
            cuis.discard(anchor)
            # All pairs: anchor → entity AND entity₁ ↔ entity₂
            for c in cuis:
                out.write(json.dumps({"src_cui": anchor, "dst_cui": c, "pmid": pmid, "edge_type": "DOC_COOCCUR"}) + "\n")
                n_edges += 1
            # entity-entity co-occurrence
            ents = list(cuis)
            for a in range(len(ents)):
                for b in range(a+1, len(ents)):
                    out.write(json.dumps({"src_cui": ents[a], "dst_cui": ents[b], "pmid": pmid, "edge_type": "DOC_COOCCUR"}) + "\n")
                    out.write(json.dumps({"src_cui": ents[b], "dst_cui": ents[a], "pmid": pmid, "edge_type": "DOC_COOCCUR"}) + "\n")
                    n_edges += 2
            if (i+1) % 50 == 0:
                el = time.time() - t0
                rate = (i+1) / max(el, 1)
                eta = (len(records) - i - 1) / max(rate, 0.01) / 60
                print(f"  {i+1}/{len(records)}  edges={n_edges:,}  rate={rate:.1f}/s  ETA={eta:.0f}min", flush=True)
    print(f"Done. {n_edges:,} co-occurrence edges → {OUT}", flush=True)


if __name__ == "__main__":
    main()
