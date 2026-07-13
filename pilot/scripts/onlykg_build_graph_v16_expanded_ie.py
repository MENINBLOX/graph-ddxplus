#!/usr/bin/env python3
"""v16: rebuild KG from expanded universal PubMed IE.

After Phase D incremental IE on ~16K new universal DISO CUIs,
edges_pubmed_ie.jsonl grows from 413K to ~700K+ edges, covering
~35K diseases instead of ~19K.

Strategy: start from v13 (with alias fixes), then add the NEW
edges (PubMed entries appended after timestamp T) as additional
HAS_PHENOTYPE links with proper IDF reweighting.

NOTE: Universal-only — uses UMLS preferred names, no DDXPlus 49
disease English names anywhere in the pipeline.
"""
from __future__ import annotations
import sys, json, math, pickle, re, time
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph_v13.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v16.pkl"
PUBMED_IE = Path("/windows/data/medkg/processed/edges_pubmed_ie.jsonl")

# CUIs already covered by v13 (snapshot count before Phase D ran)
# v13 covers 19,316 disease CUIs from PubMed.
V13_DISEASE_CUI_COUNT = 19316

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


def lemmatize(t): return " ".join(lemma_word(w) for w in t.split())


def main():
    print(f"Loading v13 from {GRAPH_IN}")
    G = pickle.load(open(GRAPH_IN, "rb"))
    print(f"  v13: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Existing disease CUIs in v13 (with phen edges)
    v13_disease_cuis = set()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            v13_disease_cuis.add(u)
    print(f"  v13 disease CUIs (with phen): {len(v13_disease_cuis):,}")

    # Load all PubMed IE edges; partition into already-in-v13 vs new
    print(f"\nLoading expanded PubMed IE from {PUBMED_IE}")
    all_edges = []
    with open(PUBMED_IE) as f:
        for line in f:
            try: d = json.loads(line)
            except: continue
            if d.get("umls_cui"): all_edges.append(d)
    print(f"  Total IE edges: {len(all_edges):,}")

    new_edges = [e for e in all_edges if e["umls_cui"] not in v13_disease_cuis]
    print(f"  New disease IE edges (not yet in v13): {len(new_edges):,}")
    new_disease_cuis = {e["umls_cui"] for e in new_edges}
    print(f"  New disease CUIs to add: {len(new_disease_cuis):,}")

    # Collect phen text variants
    targets = set()
    for e in new_edges:
        t = normalize(e["phenotype"])
        targets.add(t); targets.add(lemmatize(t))
    print(f"  Unique phen variants to resolve: {len(targets):,}")

    # MRCONSO string index
    str2cui = {}
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

    # scispaCy phen links cache (optional)
    scispacy_path = MEDKG_ROOT / "kg" / "phenotype_scispacy_links.json"
    if scispacy_path.exists():
        phen_links = json.load(open(scispacy_path))
    else:
        phen_links = {}

    existing_pairs = set()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            existing_pairs.add((u, v))

    # Aggregate new edges
    edge_proto = []
    new_phen_freq = Counter()
    src_freq = Counter()  # for evidence weighting
    for e in new_edges:
        text = normalize(e["phenotype"])
        cuis = set()
        if text in str2cui: cuis.add(str2cui[text][0])
        lt = lemmatize(text)
        if lt in str2cui: cuis.add(str2cui[lt][0])
        for cui, _, _ in phen_links.get(text, []):
            cuis.add(cui)
        for pcui in cuis:
            if pcui == e["umls_cui"]: continue
            edge_proto.append({"disease": e["umls_cui"], "phen": pcui, "text": e["phenotype"]})
            new_phen_freq[pcui] += 1
            src_freq[(e["umls_cui"], pcui)] += 1

    # Recompute IDF including new phens
    full_freq = Counter()
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") == "HAS_PHENOTYPE":
            full_freq[v] += 1
    for p, c in new_phen_freq.items():
        full_freq[p] += c
    N = max(len(v13_disease_cuis) + len(new_disease_cuis), 19000)
    idf = {p: math.log(N / max(c, 1)) for p, c in full_freq.items()}

    added = 0
    new_diseases_added = 0
    new_phens_added = 0
    seen_disease = set()
    for d, p in {(e["disease"], e["phen"]) for e in edge_proto}:
        if (d, p) in existing_pairs: continue
        if d not in G:
            G.add_node(d, ntype="Disease", name=d)
            new_diseases_added += 1
        if p not in G:
            G.add_node(p, ntype="Phenotype", name=p, source="v16")
            new_phens_added += 1
        co = src_freq[(d, p)]
        w = math.log1p(co) * 0.6 * idf.get(p, 1.0)
        G.add_edge(d, p, etype="HAS_PHENOTYPE", weight=w, source="pubmed_ie_v16")
        existing_pairs.add((d, p))
        added += 1
        seen_disease.add(d)

    print(f"\nAdded {added:,} new edges across {len(seen_disease):,} diseases")
    print(f"  New disease nodes: {new_diseases_added:,}")
    print(f"  New phen nodes: {new_phens_added:,}")
    print(f"Final v16: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\nSaving to {GRAPH_OUT}")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)
    print("Done.")


if __name__ == "__main__":
    main()
