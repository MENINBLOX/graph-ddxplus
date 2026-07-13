#!/usr/bin/env python3
"""only-KG v6 graph: lemma-based CUI consolidation.

Apply simple lemmatization (strip plural/inflection suffixes) before MRCONSO
lookup, so singular/plural variants ("pain"/"pains", "cough"/"coughing")
map to the same CUI. Also tries lowercased + lemmatized + original forms.

This addresses the user-flagged issue: nodes split by inflection get
fragmented and lose multi-disease connection signal.
"""
from __future__ import annotations
import sys, json, math, time, pickle, re
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

import networkx as nx

FEATURES = MEDKG_ROOT / "kg" / "disease_features_dual_v2_by_cui.json"
GRAPH_V4 = MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v6.pkl"
PHEN_LINKS = MEDKG_ROOT / "kg" / "phenotype_scispacy_links.json"

PREFERRED_SABS = ["HPO", "SNOMEDCT_US", "MSH", "MEDCIN", "NCI", "ICD10CM"]
SAB_PRIORITY = {s: i for i, s in enumerate(PREFERRED_SABS)}


def normalize(text):
    t = text.lower().strip()
    t = re.sub(r'[()\[\]{}]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def lemma_word(w):
    """Strip common inflection suffixes."""
    if len(w) <= 3: return w
    for suffix in ["'s", "ies", "sses", "ches", "ses", "es", "ed", "ing", "s"]:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            base = w[:-len(suffix)]
            # Re-add for -ies/-y conversion
            if suffix == "ies":
                return base + "y"
            return base
    return w


def lemmatize(text):
    return " ".join(lemma_word(w) for w in text.split())


def gen_variants(text):
    """Generate forms: original lowercased, lemmatized, both."""
    norm = normalize(text)
    variants = {norm}
    lemma = lemmatize(norm)
    variants.add(lemma)
    # Lowercase only (no lemma)
    return variants


def main():
    print("=" * 70)
    print("only-KG v6: lemma-based CUI consolidation")
    print("=" * 70)

    features = json.load(open(FEATURES))
    G = pickle.load(open(GRAPH_V4, "rb"))
    phen_links = json.load(open(PHEN_LINKS)) if PHEN_LINKS.exists() else {}

    # UNION mode: keep v4 edges, ADD only new
    n_existing = sum(1 for _,_,e in G.edges(data=True) if e.get("etype")=="HAS_PHENOTYPE")
    print(f"\n[1] Kept {n_existing:,} v4 HAS_PHENOTYPE edges (UNION mode)")

    # Build comprehensive string→CUI from MRCONSO with lemmatization
    print("\n[2] Building MRCONSO lemma+normalized index...")
    # Collect all phenotype text variants
    target_variants = set()
    text_to_variants = {}  # original_normalized → set of variants
    for dcui, phens in features.items():
        for p in phens:
            norm = normalize(p["phenotype"])
            variants = gen_variants(p["phenotype"])
            target_variants.update(variants)
            text_to_variants[norm] = variants
    print(f"   Total target variants: {len(target_variants):,}")

    # Index MRCONSO with both lowercased and lemmatized
    str2cui = {}
    t0 = time.time()
    n_total = 0
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            n_total += 1
            if n_total % 2000000 == 0:
                print(f"   {n_total:,} lines, {len(str2cui):,} matched ({time.time()-t0:.0f}s)")
            parts = line.split("|")
            if len(parts) < 15 or parts[1] != "ENG": continue
            text = parts[14]
            norm = normalize(text)
            lemma = lemmatize(norm)
            cui = parts[0]; sab = parts[11]
            prio = SAB_PRIORITY.get(sab, 99)
            for variant in (norm, lemma):
                if variant not in target_variants: continue
                if variant in str2cui:
                    old_cui, old_prio = str2cui[variant]
                    if prio < old_prio: str2cui[variant] = (cui, prio)
                else:
                    str2cui[variant] = (cui, prio)
    print(f"   Done: {len(str2cui):,} variants linked ({time.time()-t0:.0f}s)")

    # Compute coverage
    matched_originals = set()
    for orig, variants in text_to_variants.items():
        if any(v in str2cui for v in variants):
            matched_originals.add(orig)
    print(f"   Original phenotype string coverage: {len(matched_originals):,}/{len(text_to_variants):,} ({100*len(matched_originals)/len(text_to_variants):.1f}%)")

    # Build edges using all CUI sources: MRCONSO direct (with lemma), scispaCy
    print("\n[3] Building edges...")
    edges = []
    for dcui, phens in features.items():
        for p in phens:
            orig = normalize(p["phenotype"])
            variants = text_to_variants.get(orig, gen_variants(p["phenotype"]))
            phen_cuis = set()
            # MRCONSO via any variant
            for var in variants:
                if var in str2cui:
                    phen_cuis.add(str2cui[var][0])
            # scispaCy NER
            for cui, _, _ in phen_links.get(orig, []):
                phen_cuis.add(cui)
            if not phen_cuis: continue
            score = p.get("score", 0.5)
            n_sources = p.get("n_sources", 1)
            for pcui in phen_cuis:
                if pcui == dcui: continue
                edges.append({"disease": dcui, "phen": pcui,
                              "score": score, "n_sources": n_sources})

    phen_freq = Counter(e["phen"] for e in edges)
    N = len(features)
    idf = {p: math.log(N / max(c, 1)) for p, c in phen_freq.items()}

    # Aggregate edges
    edge_acc = {}
    for e in edges:
        key = (e["disease"], e["phen"])
        agreement = 0.5 + 0.5 * min(e["n_sources"], 5) / 5
        w = math.log1p(e["score"] * 10) * agreement * idf.get(e["phen"], 1.0)
        edge_acc[key] = edge_acc.get(key, 0) + w

    print(f"   Edges (proto): {len(edges):,}, unique (D,P) pairs: {len(edge_acc):,}")

    print("\n[4] Adding NEW edges (union with v4)...")
    existing_pairs = set()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            existing_pairs.add((u, v))
    added = 0; skipped = 0
    for (d, p), w in edge_acc.items():
        if (d, p) in existing_pairs:
            skipped += 1; continue
        if p not in G:
            G.add_node(p, ntype="Phenotype", name=p, source="v6_lemma")
        G.add_edge(d, p, etype="HAS_PHENOTYPE", weight=w)
        added += 1
    print(f"   Added {added:,} new edges, skipped {skipped:,} duplicates")
    print(f"   Final: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\n[5] Saving to {GRAPH_OUT}")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)


if __name__ == "__main__":
    main()
