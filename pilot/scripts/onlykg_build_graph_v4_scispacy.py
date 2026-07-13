#!/usr/bin/env python3
"""only-KG v4 graph: scispaCy NER + UMLS linker on unmatched phenotype phrases.

Decomposes multi-word phenotype phrases ("skin lesions located on the buttocks")
into component entities ("skin lesions" → C0037284, "buttocks" → C0006497) and
adds each as a HAS_PHENOTYPE edge.

Union with v3 (which has MRCONSO direct matches) → v4 final.
Threshold 0.85 (scispaCy default for high precision).
"""
from __future__ import annotations
import sys, json, math, time, pickle, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

import scispacy, spacy
from scispacy.linking import EntityLinker
import networkx as nx

FEATURES = MEDKG_ROOT / "kg" / "disease_features_dual_v2_by_cui.json"
GRAPH_V3 = MEDKG_ROOT / "kg" / "onlykg_graph_v3.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl"
CACHE = MEDKG_ROOT / "kg" / "phenotype_scispacy_links.json"

THRESHOLD = 0.85


def normalize(text):
    t = text.lower().strip()
    t = re.sub(r'[()\[\]{}]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def main():
    print("=" * 70)
    print("only-KG v4: scispaCy NER decomposition of phenotype phrases")
    print("=" * 70)

    features = json.load(open(FEATURES))
    phen_strings = set()
    for dcui, phens in features.items():
        for p in phens:
            phen_strings.add(normalize(p["phenotype"]))
    print(f"  Unique phenotype strings: {len(phen_strings):,}")

    # Check existing cache
    if CACHE.exists():
        cache = json.load(CACHE.open())
        cached = set(cache.keys())
        todo = [s for s in phen_strings if s not in cached]
        print(f"  Cached: {len(cache):,}, TODO: {len(todo):,}")
    else:
        cache = {}
        todo = list(phen_strings)
        print(f"  No cache, TODO all: {len(todo):,}")

    if todo:
        print("Loading scispaCy en_core_sci_lg + UMLS linker (threshold=0.70)...")
        nlp = spacy.load("en_core_sci_lg")
        nlp.add_pipe("scispacy_linker", config={
            "resolve_abbreviations": True,
            "linker_name": "umls",
            "threshold": 0.70,  # extract candidates above 0.70
            "max_entities_per_mention": 3,
        })
        print("  Loaded")

        t0 = time.time()
        for i, doc in enumerate(nlp.pipe(todo, batch_size=128, n_process=1)):
            ents = []
            for ent in doc.ents:
                for cui, score in ent._.kb_ents[:2]:
                    if score >= THRESHOLD:
                        ents.append((cui, float(score), ent.text))
                        break  # take top per entity
            cache[todo[i]] = ents
            if (i + 1) % 5000 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(todo) - i - 1) / rate
                print(f"    {i+1:,}/{len(todo):,}  rate={rate:.0f}/s  eta={eta:.0f}s")
        print(f"  Done ({time.time()-t0:.0f}s)")
        with CACHE.open("w") as f:
            json.dump(cache, f)
        print(f"  Saved cache to {CACHE}")

    # Stats
    n_linked = sum(1 for v in cache.values() if v)
    print(f"  Phrases with ≥1 linked CUI: {n_linked:,} / {len(cache):,} ({100*n_linked/len(cache):.1f}%)")

    # Build v4 graph: v3 + new scispaCy edges
    print("\nLoading v3 graph as base...")
    with GRAPH_V3.open("rb") as f:
        G = pickle.load(f)
    n_v3_phen = sum(1 for _,_,e in G.edges(data=True) if e.get("etype")=="HAS_PHENOTYPE")
    print(f"  v3 HAS_PHENOTYPE edges: {n_v3_phen:,}")

    # Track existing pairs to avoid duplicates
    existing = set()
    for u, v, edata in G.edges(data=True):
        if edata.get("etype") == "HAS_PHENOTYPE":
            existing.add((u, v))

    # Disease/phen frequency for IDF (new linkages)
    new_freq = Counter()
    new_edges_proto = []  # (disease, phen_cui, score, sources, weight_factor)
    for dcui, phens in features.items():
        for p in phens:
            norm = normalize(p["phenotype"])
            links = cache.get(norm, [])
            sources = p.get("sources", [])
            n_sources = p.get("n_sources", len(sources))
            score = p.get("score", 0.5)
            for pcui, link_score, ent_text in links:
                if pcui == dcui: continue
                new_edges_proto.append({
                    "disease": dcui, "phen": pcui,
                    "score": score * link_score,  # downweight by link confidence
                    "sources": sources, "n_sources": n_sources,
                    "ent_text": ent_text,
                })
                new_freq[pcui] += 1

    # IDF for new phens
    N = len(features)
    new_idf = {p: math.log(N / new_freq[p]) for p in new_freq}

    added = 0; dedup = 0
    for e in new_edges_proto:
        d, p = e["disease"], e["phen"]
        if (d, p) in existing:
            dedup += 1; continue
        if p not in G:
            G.add_node(p, ntype="Phenotype", name=e["ent_text"], source="v4_scispacy")
        source_agreement = 0.5 + 0.5 * min(e["n_sources"], 5) / 5
        w = math.log1p(e["score"] * 10) * source_agreement * new_idf.get(p, 1.0)
        G.add_edge(d, p, etype="HAS_PHENOTYPE", weight=w,
                   score=e["score"], sources=e["sources"])
        existing.add((d, p))
        added += 1
    print(f"  Added {added:,} new v4 edges (skipped {dedup:,} duplicates)")
    print(f"  Final: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\nSaving v4 graph to {GRAPH_OUT}...")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)
    print(f"  Saved")

    # Sample
    PN = "C0694504"
    if PN in G:
        phens = [(p, G.nodes[p].get("name","?"), edata.get("weight",0))
                 for _, p, edata in G.out_edges(PN, data=True)
                 if edata.get("etype") == "HAS_PHENOTYPE"]
        phens.sort(key=lambda x: -x[2])
        print(f"\nSample Pneumonia (C0694504) top 15:")
        for pcui, name, w in phens[:15]:
            print(f"  [{w:6.2f}] {name:<40s} {pcui}")


if __name__ == "__main__":
    main()
