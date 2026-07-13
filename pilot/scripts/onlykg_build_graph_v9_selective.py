#!/usr/bin/env python3
"""v9: selectively merge pubmed_alt only for sparse diseases.

Only add v8 (pubmed_alt) edges to diseases that have <T phens in v7.
This avoids diluting well-covered diseases while enriching sparse ones.
"""
from __future__ import annotations
import sys, json, math, pickle, re
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

GRAPH_V7 = MEDKG_ROOT / "kg" / "onlykg_graph_v7.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v9.pkl"
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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=50, help="only enrich diseases with <T phens")
    args = ap.parse_args()

    print(f"Loading v7 graph (threshold={args.threshold})...")
    G = pickle.load(open(GRAPH_V7, "rb"))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    ddx_cuis = {info["cui"] for info in icd.values() if "cui" in info}

    # Count v7 phens per DDXPlus disease
    v7_counts = {}
    for d in ddx_cuis:
        if d not in G: v7_counts[d] = 0; continue
        v7_counts[d] = sum(1 for _,_,e in G.out_edges(d, data=True) if e.get("etype")=="HAS_PHENOTYPE")
    sparse_diseases = {d for d, n in v7_counts.items() if n < args.threshold}
    print(f"  Sparse diseases (<{args.threshold} phens): {len(sparse_diseases)}")

    # Load pubmed_alt edges
    raw_edges = [json.loads(l) for l in open(EDGES_FILE) if l.strip()]
    raw_edges = [e for e in raw_edges if e.get("umls_cui") in sparse_diseases]
    print(f"  pubmed_alt edges for sparse diseases: {len(raw_edges):,}")

    # Build MRCONSO for these phen texts
    targets = set()
    for e in raw_edges:
        n = normalize(e["phenotype"])
        targets.add(n)
        targets.add(" ".join(lemma_word(w) for w in n.split()))

    str2cui = {}
    import time
    t0 = time.time()
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < 15 or parts[1] != "ENG": continue
            text = parts[14]
            norm = normalize(text)
            lemma = " ".join(lemma_word(w) for w in norm.split())
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

    phen_links = json.load(open(MEDKG_ROOT / "kg" / "phenotype_scispacy_links.json"))

    # Existing pairs
    existing_pairs = set()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            existing_pairs.add((u, v))

    added = 0; skipped = 0
    new_phen_freq = Counter()
    # First pass: collect new edges to compute IDF
    edge_proto = []
    for e in raw_edges:
        text = normalize(e["phenotype"])
        cuis = set()
        if text in str2cui: cuis.add(str2cui[text][0])
        lemma_t = " ".join(lemma_word(w) for w in text.split())
        if lemma_t in str2cui: cuis.add(str2cui[lemma_t][0])
        for cui, _, _ in phen_links.get(text, []):
            cuis.add(cui)
        for pcui in cuis:
            if pcui == e["umls_cui"]: continue
            if (e["umls_cui"], pcui) in existing_pairs: continue
            edge_proto.append({"disease": e["umls_cui"], "phen": pcui, "text": e["phenotype"]})
            new_phen_freq[pcui] += 1

    # IDF based on full graph (using all current edges + new)
    full_phen_freq = Counter()
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") == "HAS_PHENOTYPE":
            full_phen_freq[v] += 1
    for p, c in new_phen_freq.items():
        full_phen_freq[p] += c
    N = 19000
    idf = {p: math.log(N / max(c, 1)) for p, c in full_phen_freq.items()}

    # Add edges with boost for sparse-disease edges (score=0.7 instead of 0.6)
    for e in edge_proto:
        d, p = e["disease"], e["phen"]
        if p not in G:
            G.add_node(p, ntype="Phenotype", name=e["text"], source="v9_pubmed_alt_selective")
        w = math.log1p(7) * 0.7 * idf.get(p, 1.0)
        G.add_edge(d, p, etype="HAS_PHENOTYPE", weight=w)
        added += 1
    print(f"\nAdded {added:,} selective edges")
    print(f"Final: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\nSaving to {GRAPH_OUT}")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)

    print("\nDDXPlus disease counts after v9:")
    for cui, name in list({info["cui"]: dn for dn, info in icd.items() if "cui" in info}.items()):
        if cui not in G: continue
        n_e = sum(1 for _,_,e in G.out_edges(cui, data=True) if e.get("etype")=="HAS_PHENOTYPE")
        v7_n = v7_counts.get(cui, 0)
        delta = n_e - v7_n
        if delta > 0:
            print(f"  {name:40s} {v7_n}→{n_e} (+{delta})")


if __name__ == "__main__":
    main()
