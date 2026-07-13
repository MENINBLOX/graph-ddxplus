#!/usr/bin/env python3
"""v17_partial: add Wikipedia IE edges to v13 (vocab-gap hypothesis test).

Top Wiki phens (fever/vomiting/fatigue/headache/nausea/cough/sore throat)
match DDXPlus questionnaire vocabulary exactly. This builds v13 + Wiki
edges to test whether the lay-vocab gap is the primary ceiling driver.
"""
from __future__ import annotations
import sys, json, math, pickle, re, time
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph_v13.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v17_wiki.pkl"
WIKI_IE = Path("/windows/data/medkg/processed/edges_wikipedia_ie.jsonl")

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

    print(f"\nLoading Wikipedia IE edges from {WIKI_IE}")
    wiki_edges = []
    with WIKI_IE.open() as f:
        for line in f:
            try: d = json.loads(line)
            except: continue
            if d.get("umls_cui"): wiki_edges.append(d)
    print(f"  Wiki edges: {len(wiki_edges):,}")

    # Phen text resolution targets
    targets = set()
    for e in wiki_edges:
        t = normalize(e["phenotype"])
        targets.add(t); targets.add(lemmatize(t))
    print(f"  Unique phen variants: {len(targets):,}")

    # MRCONSO index
    str2cui = {}
    t0 = time.time()
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < 15 or parts[1] != "ENG": continue
            text = parts[14]
            norm = normalize(text); lemma = lemmatize(norm)
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

    scispacy_path = MEDKG_ROOT / "kg" / "phenotype_scispacy_links.json"
    phen_links = json.load(open(scispacy_path)) if scispacy_path.exists() else {}

    existing_pairs = set()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            existing_pairs.add((u, v))

    edge_proto = []
    new_phen_freq = Counter()
    src_freq = Counter()
    for e in wiki_edges:
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

    print(f"  Resolved phen-CUI links: {sum(len(set([e['phen'] for e in edge_proto if e['disease']==d])) for d in set(e['disease'] for e in edge_proto))} unique pairs (raw {len(edge_proto)})")

    full_freq = Counter()
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") == "HAS_PHENOTYPE":
            full_freq[v] += 1
    for p, c in new_phen_freq.items():
        full_freq[p] += c
    N = max(20000, len(set(u for u,v,e in G.edges(data=True) if e.get("etype")=="HAS_PHENOTYPE")))
    idf = {p: math.log(N / max(c, 1)) for p, c in full_freq.items()}

    added = 0; new_phens_added = 0
    seen_pairs = set()
    for ep in edge_proto:
        d, p = ep["disease"], ep["phen"]
        if (d, p) in seen_pairs: continue
        seen_pairs.add((d, p))
        if (d, p) in existing_pairs: continue
        if d not in G: G.add_node(d, ntype="Disease", name=d)
        if p not in G:
            G.add_node(p, ntype="Phenotype", name=ep["text"], source="v17_wiki")
            new_phens_added += 1
        co = src_freq[(d, p)]
        w = math.log1p(co) * 1.0 * idf.get(p, 1.0)  # weight 1.0 for Wiki (lay-vocab)
        G.add_edge(d, p, etype="HAS_PHENOTYPE", weight=w, source="wikipedia_ie")
        existing_pairs.add((d, p))
        added += 1

    print(f"\nAdded {added:,} new edges from Wikipedia")
    print(f"  New phen nodes: {new_phens_added:,}")
    print(f"Final v17_wiki: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\nSaving to {GRAPH_OUT}")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)
    print("Done.")


if __name__ == "__main__":
    main()
