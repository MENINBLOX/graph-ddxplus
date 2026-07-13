#!/usr/bin/env python3
"""v15: integrate patient-focused gemma-4-E4B IE results into v14 graph.

466 new patient-vocab phenotype mentions from sparse diseases.
Map text → CUI via MRCONSO + scispaCy cache.
"""
from __future__ import annotations
import sys, json, math, pickle, re, time
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph_v14.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v15.pkl"
PATIENT_IE = MEDKG_ROOT / "processed" / "edges_patient_focused_ie.jsonl"


def normalize(text):
    t = text.lower().strip()
    t = re.sub(r'[()\[\]{}]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def main():
    print("Loading v14 graph + patient IE...")
    G = pickle.load(open(GRAPH_IN, "rb"))
    n_v14 = sum(1 for _,_,e in G.edges(data=True) if e.get("etype")=="HAS_PHENOTYPE")
    print(f"v14: {G.number_of_nodes():,} nodes, {n_v14:,} HAS_PHENOTYPE edges")

    # Load patient IE edges
    raw = [json.loads(l) for l in open(PATIENT_IE)]
    print(f"Patient IE edges: {len(raw)}")

    # Group by (disease, phenotype_text) and count pmids
    from collections import defaultdict
    edge_pmids = defaultdict(set)
    edge_disease_name = {}
    for r in raw:
        key = (r["eval_cui"], normalize(r["phenotype"]))
        edge_pmids[key].add(r.get("pmid", ""))
        edge_disease_name[r["eval_cui"]] = r["disease"]
    print(f"Unique (disease, phen_text) pairs: {len(edge_pmids)}")

    # Collect unique phen texts for CUI mapping
    phen_texts = {key[1] for key in edge_pmids}
    print(f"Unique phen text variants: {len(phen_texts)}")

    # MRCONSO lookup
    print("Loading MRCONSO subset...")
    str2cui = {}
    PREFERRED_SABS = ["HPO", "SNOMEDCT_US", "MSH", "MEDCIN", "NCI", "ICD10CM"]
    SAB_PRIORITY = {s: i for i, s in enumerate(PREFERRED_SABS)}
    t0 = time.time()
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < 15 or parts[1] != "ENG": continue
            norm = normalize(parts[14])
            if norm not in phen_texts: continue
            cui = parts[0]; sab = parts[11]
            prio = SAB_PRIORITY.get(sab, 99)
            if norm in str2cui:
                if prio < str2cui[norm][1]:
                    str2cui[norm] = (cui, prio)
            else:
                str2cui[norm] = (cui, prio)
    print(f"  Mapped {len(str2cui):,} strings → CUI ({time.time()-t0:.0f}s)")

    # Try scispaCy on unmapped
    unmapped = phen_texts - set(str2cui.keys())
    print(f"Unmapped: {len(unmapped)}, running scispaCy...")
    if unmapped:
        import warnings; warnings.filterwarnings("ignore")
        import scispacy, spacy
        from scispacy.linking import EntityLinker
        nlp = spacy.load("en_core_sci_lg")
        nlp.add_pipe("scispacy_linker", config={
            "resolve_abbreviations": True, "linker_name": "umls",
            "threshold": 0.70, "max_entities_per_mention": 2,
        })
        scispacy_links = {}
        for doc, text in zip(nlp.pipe(list(unmapped), batch_size=64), list(unmapped)):
            for ent in doc.ents:
                if not ent._.kb_ents: continue
                for cui, score in ent._.kb_ents[:1]:
                    if score < 0.75: continue
                    scispacy_links[text] = cui
                    break
                if text in scispacy_links: break
        print(f"  scispaCy mapped {len(scispacy_links)} additional strings")
        for t, c in scispacy_links.items():
            str2cui[t] = (c, 50)

    # Add edges to graph
    existing_pairs = set()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            existing_pairs.add((u, v))

    added = 0; skipped = 0
    for (dcui, phen_text), pmids in edge_pmids.items():
        if phen_text not in str2cui:
            skipped += 1; continue
        pcui = str2cui[phen_text][0]
        if pcui == dcui: continue
        if (dcui, pcui) in existing_pairs:
            skipped += 1; continue
        if pcui not in G:
            G.add_node(pcui, ntype="Phenotype", name=phen_text, source="v15_patient_ie")
        # Weight: log1p(n_pmids) × 1.0 (patient-focused IE = high quality)
        w = math.log1p(len(pmids)) * 1.0
        G.add_edge(dcui, pcui, etype="HAS_PHENOTYPE", weight=w, source="patient_ie")
        existing_pairs.add((dcui, pcui))
        added += 1
    print(f"\nAdded {added} new patient-IE edges (skipped {skipped})")
    print(f"Final: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\nSaving v15 to {GRAPH_OUT}")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)


if __name__ == "__main__":
    main()
