#!/usr/bin/env python3
"""v8: merge edges_pubmed_alt_ie.jsonl into KG.

The pubmed_alt IE was done specifically on 49 DDXPlus diseases with alt-search
PubMed queries but never merged into the main KG. Contains 1,849 disease-phen
edges covering 45/49 DDXPlus diseases, many with patient-reportable
vocabulary (fever, myalgia, etc.).
"""
from __future__ import annotations
import sys, json, math, time, pickle, re
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

import networkx as nx

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph_v7.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v8.pkl"
EDGES_FILE = Path("/windows/data/medkg/processed/edges_pubmed_alt_ie.jsonl")

PREFERRED_SABS = ["HPO", "SNOMEDCT_US", "MSH", "MEDCIN", "NCI", "ICD10CM"]
SAB_PRIORITY = {s: i for i, s in enumerate(PREFERRED_SABS)}


def normalize(text):
    t = text.lower().strip()
    t = re.sub(r'[()\[\]{}]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def lemma_word(w):
    if len(w) <= 3: return w
    for suffix in ["'s", "ies", "sses", "ches", "ses", "es", "ed", "ing", "s"]:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            base = w[:-len(suffix)]
            if suffix == "ies": return base + "y"
            return base
    return w


def lemmatize(text):
    return " ".join(lemma_word(w) for w in text.split())


def main():
    print("Loading v7 graph...")
    G = pickle.load(open(GRAPH_IN, "rb"))
    print(f"  Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")

    # Load edges
    print("\nLoading pubmed_alt_ie edges...")
    raw_edges = []
    phen_texts = set()
    disease_cuis = set()
    with open(EDGES_FILE) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("umls_cui"): continue
            raw_edges.append(d)
            phen_texts.add(normalize(d["phenotype"]))
            disease_cuis.add(d["umls_cui"])
    print(f"  Edges: {len(raw_edges):,}, unique phen texts: {len(phen_texts):,}, disease CUIs: {len(disease_cuis)}")

    # Build MRCONSO lookup for phen texts (with lemma)
    print("\nBuilding MRCONSO string→CUI index (with lemma)...")
    str2cui = {}
    targets = phen_texts | {lemmatize(t) for t in phen_texts}
    t0 = time.time()
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < 15 or parts[1] != "ENG": continue
            text = parts[14]
            norm = normalize(text)
            lemma = lemmatize(norm)
            cui = parts[0]; sab = parts[11]
            prio = SAB_PRIORITY.get(sab, 99)
            for variant in (norm, lemma):
                if variant not in targets: continue
                if variant in str2cui:
                    if prio < str2cui[variant][1]:
                        str2cui[variant] = (cui, prio)
                else:
                    str2cui[variant] = (cui, prio)
    print(f"  Indexed {len(str2cui):,} strings ({time.time()-t0:.0f}s)")

    # Also use scispaCy phen_links cache for difficult ones
    phen_links = json.load(open(MEDKG_ROOT / "kg" / "phenotype_scispacy_links.json"))

    # Compute new edge counts for IDF
    phen_per_edge_cuis = {}  # phen_text → set of CUI candidates
    for text in phen_texts:
        cuis = set()
        if text in str2cui: cuis.add(str2cui[text][0])
        lemma_t = lemmatize(text)
        if lemma_t in str2cui: cuis.add(str2cui[lemma_t][0])
        for cui, _, _ in phen_links.get(text, []):
            cuis.add(cui)
        phen_per_edge_cuis[text] = cuis

    # Existing pairs in v7
    existing_pairs = set()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            existing_pairs.add((u, v))

    # Aggregate edges (disease, phen) → count for IDF
    edge_proto = []
    for d in raw_edges:
        text = normalize(d["phenotype"])
        cuis = phen_per_edge_cuis.get(text, set())
        if not cuis: continue
        for pcui in cuis:
            edge_proto.append({
                "disease": d["umls_cui"], "phen": pcui, "text": d["phenotype"],
            })

    phen_freq = Counter(e["phen"] for e in edge_proto)
    N = 19000  # approximate disease count for IDF baseline
    idf = {p: math.log(N / max(c, 1)) for p, c in phen_freq.items()}

    added = 0; skipped = 0; resolved_cuis = set()
    for e in edge_proto:
        d, p = e["disease"], e["phen"]
        if p == d: continue
        if (d, p) in existing_pairs:
            skipped += 1; continue
        if p not in G:
            G.add_node(p, ntype="Phenotype", name=e["text"], source="v8_pubmed_alt")
        # weight: source-agreement assumed 1 (single source), score=0.6 default
        w = math.log1p(6) * 0.6 * idf.get(p, 1.0)
        G.add_edge(d, p, etype="HAS_PHENOTYPE", weight=w)
        existing_pairs.add((d, p))
        added += 1
        resolved_cuis.add(p)
    print(f"\nAdded {added:,} new edges, skipped {skipped:,} dupes")
    print(f"Unique new phenotype CUIs: {len(resolved_cuis):,}")
    print(f"Final: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\nSaving to {GRAPH_OUT}")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)

    # Sample low-coverage diseases after v8
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    ddx = {info["cui"]: dn for dn, info in icd.items() if "cui" in info}
    print("\nDDXPlus disease edge counts (v8):")
    for cui, name in list(ddx.items()):
        if cui not in G: continue
        n_e = sum(1 for _,_,e in G.out_edges(cui, data=True) if e.get("etype")=="HAS_PHENOTYPE")
        if n_e > 0 and n_e < 30 or name in {"HIV (initial infection)", "Pneumonia", "Anemia", "Viral pharyngitis"}:
            print(f"  {name:40s} {n_e} edges")


if __name__ == "__main__":
    main()
