#!/usr/bin/env python3
"""v10: merge all remaining processed edges files.

edges_categorized, edges_distinguishing, edges_epidemiology
contain IE outputs from disease-specific text sections (history, distinguishing,
epidemiology) that may not have been merged into the main KG.
"""
from __future__ import annotations
import sys, json, math, pickle, re, time
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

GRAPH_V9 = MEDKG_ROOT / "kg" / "onlykg_graph_v9.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v10.pkl"
EDGE_FILES = [
    "/windows/data/medkg/processed/edges_categorized.jsonl",
    "/windows/data/medkg/processed/edges_distinguishing.jsonl",
    "/windows/data/medkg/processed/edges_epidemiology.jsonl",
]

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
    print("Loading v9...")
    G = pickle.load(open(GRAPH_V9, "rb"))
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Load DDXPlus disease set for boost
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    ddx_cuis = {info["cui"] for info in icd.values() if "cui" in info}

    # Collect all phen texts
    print("\nLoading additional edge files...")
    raw_edges = []
    for fp in EDGE_FILES:
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                if d.get("umls_cui"):
                    raw_edges.append(d)
    print(f"  Total: {len(raw_edges):,} edges across {len(EDGE_FILES)} files")
    targets = set()
    for e in raw_edges:
        t = normalize(e["phenotype"])
        targets.add(t)
        targets.add(lemmatize(t))
    print(f"  Unique phen variants to resolve: {len(targets):,}")

    # MRCONSO index
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

    phen_links = json.load(open(MEDKG_ROOT / "kg" / "phenotype_scispacy_links.json"))

    existing_pairs = set()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            existing_pairs.add((u, v))

    # Process edges
    edge_proto = []
    new_phen_freq = Counter()
    for e in raw_edges:
        text = normalize(e["phenotype"])
        cuis = set()
        if text in str2cui: cuis.add(str2cui[text][0])
        lt = lemmatize(text)
        if lt in str2cui: cuis.add(str2cui[lt][0])
        for cui, _, _ in phen_links.get(text, []):
            cuis.add(cui)
        for pcui in cuis:
            if pcui == e["umls_cui"]: continue
            if (e["umls_cui"], pcui) in existing_pairs: continue
            edge_proto.append({"disease": e["umls_cui"], "phen": pcui, "text": e["phenotype"]})
            new_phen_freq[pcui] += 1

    # IDF (full graph + new)
    full_freq = Counter()
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") == "HAS_PHENOTYPE":
            full_freq[v] += 1
    for p, c in new_phen_freq.items():
        full_freq[p] += c
    N = 19000
    idf = {p: math.log(N / max(c, 1)) for p, c in full_freq.items()}

    added = 0
    for e in edge_proto:
        d, p = e["disease"], e["phen"]
        if (d, p) in existing_pairs: continue
        if p not in G:
            G.add_node(p, ntype="Phenotype", name=e["text"], source="v10")
        w = math.log1p(6) * 0.6 * idf.get(p, 1.0)
        G.add_edge(d, p, etype="HAS_PHENOTYPE", weight=w)
        existing_pairs.add((d, p))
        added += 1
    print(f"\nAdded {added:,} new edges")
    print(f"Final: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\nSaving to {GRAPH_OUT}")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)


if __name__ == "__main__":
    main()
