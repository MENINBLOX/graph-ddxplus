#!/usr/bin/env python3
"""Audit benchmark coverage + CUI normalization issues.

1. For each of 49 DDXPlus diseases:
   - n_phens in KG
   - Q-coverage: |Q ∩ phens(D)|
2. Identify orphan phenotype nodes (out-degree=0 except 1 disease edge)
3. Identify potential singular/plural collisions: phenotype texts that
   differ only by inflection but map to different (or no) CUIs.
"""
from __future__ import annotations
import sys, json, pickle, re
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
FEATURES = MEDKG_ROOT / "kg" / "disease_features_dual_v2_by_cui.json"


def lemmatize_simple(text):
    """Quick lemma: strip trailing s/es/ies/ed/ing."""
    t = text.lower().strip()
    for suffix in ["'s", "ies", "ses", "es", "ed", "ing", "s"]:
        if t.endswith(suffix) and len(t) > len(suffix) + 2:
            return t[:-len(suffix)]
    return t


def normalize(text):
    t = text.lower().strip()
    t = re.sub(r'[()\[\]{}]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def main():
    G = pickle.load(open(GRAPH, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))
    features = json.load(open(FEATURES))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)
    print(f"|Q| = {len(Q)}")

    # 1. Per-disease coverage
    print("\n=== 49 DDXPlus disease coverage ===")
    print(f"{'Disease':<35s} {'IE-text':>7s} {'KG-edges':>9s} {'Q∩phens':>8s}")
    low_coverage = []
    for cui in dcs_list:
        n_ie = len(features.get(cui, []))
        if cui in G:
            phens = [p for _, p, e in G.out_edges(cui, data=True) if e.get('etype')=='HAS_PHENOTYPE']
            n_edges = len(phens)
            q_phens = len([p for p in phens if p in Q])
        else:
            n_edges = q_phens = 0
        name = cui2name.get(cui, cui)
        line = f"  {name:<35s} {n_ie:>7d} {n_edges:>9d} {q_phens:>8d}"
        if q_phens < 5:
            low_coverage.append((name, n_ie, n_edges, q_phens))
            print(line + "  <-- LOW Q coverage")

    print(f"\nLow Q-coverage diseases (Q∩phens<5): {len(low_coverage)}/49")
    for name, n_ie, n_edges, q_phens in low_coverage:
        print(f"  {name}: IE_text={n_ie} edges={n_edges} Q∩phens={q_phens}")

    # 2. Orphan / singleton phenotype detection
    print("\n=== Phenotype node connectivity ===")
    phen_indegree = Counter()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            phen_indegree[v] += 1
    indeg_dist = Counter(phen_indegree.values())
    print(f"Phenotype nodes by in-degree (# diseases pointing to them):")
    for k in sorted(indeg_dist.keys())[:10]:
        print(f"  indegree={k}: {indeg_dist[k]:,} nodes")
    n_singleton = indeg_dist.get(1, 0)
    print(f"Singleton phen nodes (indegree=1): {n_singleton:,}")

    # 3. Lemma-based collision check: how many phenotype texts share a lemma but
    #    got linked to different CUIs (or only some got linked)?
    print("\n=== Lemmatization-based normalization audit ===")
    all_texts = set()
    for dcui, phens in features.items():
        for p in phens:
            all_texts.add(normalize(p["phenotype"]))
    lemma_groups = defaultdict(set)
    for t in all_texts:
        lemma = " ".join(lemmatize_simple(w) for w in t.split())
        lemma_groups[lemma].add(t)
    multi_lemma = {k: v for k, v in lemma_groups.items() if len(v) > 1}
    print(f"Lemma groups with >1 surface variant: {len(multi_lemma):,}")
    print("Examples (3 samples):")
    for lemma, variants in list(multi_lemma.items())[:10]:
        print(f"  '{lemma}': {sorted(variants)[:5]}")

    # Total potential savings: # of text variants that would merge
    n_var = sum(len(v) for v in multi_lemma.values())
    n_groups = len(multi_lemma)
    print(f"  → {n_var:,} variant strings could collapse to {n_groups:,} lemma groups (saving {n_var-n_groups:,})")


if __name__ == "__main__":
    main()
