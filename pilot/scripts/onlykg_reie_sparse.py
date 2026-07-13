#!/usr/bin/env python3
"""Re-IE sparse diseases via scispaCy on existing text.

For each low-coverage disease, run scispaCy on existing pubmed_alt papers
with low threshold (0.65) to extract more CUIs as phenotype candidates.

This is a re-IE cycle: existing raw text → more aggressive entity linking.
"""
from __future__ import annotations
import sys, json, pickle, re, math, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

import scispacy, spacy
from scispacy.linking import EntityLinker

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph_v13.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v14.pkl"

# Sparse diseases (Q∩phens<10 or n_phens<50)
SPARSE_DDX_CUIS = {
    "C0013609": "Localized edema",
    "C0043168": "Whooping cough",
    "C0348343": "Pulmonary neoplasm",
    "C0478237": "Spontaneous rib fracture",
    "C0041912": "URTI",
    "C0001344": "Viral pharyngitis",
    "C0340044": "Acute COPD exacerbation / infection",
    "C0236832": "Acute dystonic reactions",
    "C0023066": "Larygospasm",
    "C0039240": "PSVT",
}

KEEP_TUIS = {
    "T184", "T033", "T046", "T047", "T048", "T191", "T037", "T039", "T067",
    "T023", "T024", "T029", "T030", "T031", "T017",
}


def main():
    print("Loading v13 graph + scispaCy...")
    G = pickle.load(open(GRAPH_IN, "rb"))
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
        "threshold": 0.70,
        "max_entities_per_mention": 2,
    })

    cui2tuis = {}
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.split("|")
            cui2tuis.setdefault(p[0], set()).add(p[1])

    # For each sparse disease, find related pubmed_alt files
    # pubmed_alt is keyed by CUI but may use different CUI than our eval CUI
    # Try direct match first, then check alias variants
    PUBMED_DIR = Path("/windows/data/medkg/pubmed_alt")

    # Map of additional CUI aliases for pubmed_alt lookup
    ALIAS_LOOKUP = {
        "C0010072": ["C0010072", "C1304447", "C0151744", "C0027051"],  # NSTEMI
        "C0478237": ["C0478237", "C0035525", "C0016659"],  # Rib fracture
        "C0346647": ["C0346647", "C0153466", "C0030297"],  # Pancreatic neoplasm
        "C0340044": ["C0340044", "C0741421", "C0024117"],  # Acute COPD
        "C0023066": ["C0023066", "C0023068"],  # Laryngospasm
    }

    total_added = 0
    for dcui, dname in SPARSE_DDX_CUIS.items():
        # Find pubmed text
        candidate_cuis = ALIAS_LOOKUP.get(dcui, [dcui])
        texts = []
        for c in candidate_cuis:
            fp = PUBMED_DIR / f"{c}.jsonl"
            if fp.exists():
                with fp.open() as f:
                    for line in f:
                        d = json.loads(line)
                        t = d.get("abstract", "")
                        if t: texts.append(t)
        if not texts:
            print(f"  {dname} ({dcui}): NO pubmed_alt text found")
            continue

        print(f"\n  {dname} ({dcui}): {len(texts)} papers")
        # Run scispaCy
        cui_counts = Counter()
        cui_names = {}
        for doc in nlp.pipe(texts, batch_size=8):
            for ent in doc.ents:
                if not ent._.kb_ents: continue
                for cui, score in ent._.kb_ents[:2]:
                    if score < 0.75: continue
                    tuis = cui2tuis.get(cui, set())
                    if tuis and not (tuis & KEEP_TUIS): continue
                    cui_counts[cui] += 1
                    if cui not in cui_names: cui_names[cui] = ent.text
                    break

        # Add top-K most frequent CUIs as new phenotypes
        existing = {p for _, p, e in G.out_edges(dcui, data=True) if e.get("etype") == "HAS_PHENOTYPE"}
        added = 0
        for cui, freq in cui_counts.most_common(100):
            if cui == dcui: continue
            if cui in existing: continue
            if cui not in G:
                G.add_node(cui, ntype="Phenotype", name=cui_names.get(cui, cui), source="v14_reie")
            w = math.log1p(freq) * 0.7
            G.add_edge(dcui, cui, etype="HAS_PHENOTYPE", weight=w, source="v14_reie")
            added += 1
            total_added += 1
        print(f"    Added {added} new edges")

    print(f"\nTotal added: {total_added}")
    print(f"Saving v14 to {GRAPH_OUT}")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)


if __name__ == "__main__":
    main()
