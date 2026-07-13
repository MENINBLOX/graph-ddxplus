#!/usr/bin/env python3
"""only-KG v7 graph: disease CUI alias remapping.

DDXPlus disease CUIs map to specific/narrow UMLS CUIs (e.g., HIV→C0001175
"Acquired Immunodeficiency Syndrome"). PubMed IE accumulated phenotypes on
broader CUIs (e.g., C0019693 "HIV Infections"). Aggregate phens from
broader CUIs into DDXPlus disease nodes.

Mapping curated based on:
1. UMLS alternate CUIs for same concept
2. Name-keyed feature lookup
"""
from __future__ import annotations
import sys, json, math, time, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

import networkx as nx

GRAPH_V4 = MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v7.pkl"
FEATURES_CUI = MEDKG_ROOT / "kg" / "disease_features_dual_v2_by_cui.json"
FEATURES_NAME = MEDKG_ROOT / "kg" / "disease_features_dual_v2.json"

# DDXPlus disease CUI → list of additional UMLS CUIs to aggregate phenotypes from
ALIAS_MAP = {
    "C0001175": ["C0019693", "C0019682"],            # HIV (initial) → HIV Infections, HIV
    "C1304447": ["C0151744", "C0027051"],            # NSTEMI → Acute MI, Myocardial Infarction
    "C0741421": ["C0024117"],                         # Acute COPD exacerbation → COPD
    "C0035525": ["C0016659"],                         # Spontaneous rib fracture → Rib Fractures
    "C0013609": ["C0013604"],                         # Localized edema → Edema (general)
    "C0153466": ["C0030297"],                         # Pancreatic neoplasm → Pancreatic neoplasm (alt)
    "C0023065": ["C0023068"],                         # Cluster headache alt
}

# Disease CUI → disease name key (for name-keyed features lookup; broader concepts)
NAME_LOOKUP = {
    "C0001175": ["HIV Infections", "Acute HIV infection", "AIDS related complex"],
    "C1304447": ["Myocardial Infarction", "Acute myocardial infarction"],
    "C0741421": ["Other chronic obstructive pulmonary disease"],
    "C0035525": ["Rib Fractures", "Spontaneous rib fracture"],
    "C0023068": ["Laryngospasm"],
    "C0013362": ["Drug-induced dystonia", "Neuroleptic-induced acute dystonia", "Dystonia Disorders", "Other extrapyramidal disease and abnormal movement disorders"],  # Acute dystonic reactions
    "C0153466": ["Pancreatic carcinoma"],              # Pancreatic neoplasm → carcinoma
    "C0030297": ["Pancreatic carcinoma"],              # alt
    "C0002792": ["Anaphylaxis", "anaphylaxis", "Anaphylactoid Reaction"],  # Anaphylaxis
    "C0023067": ["Acute laryngitis", "Laryngitis"],   # Acute laryngitis
}


def main():
    print("Loading v4 graph + features...")
    G = pickle.load(open(GRAPH_V4, "rb"))
    feats_cui = json.load(open(FEATURES_CUI))
    feats_name = json.load(open(FEATURES_NAME))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))

    # Existing (D, P) pairs in v4
    existing_pairs = set()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            existing_pairs.add((u, v))
    print(f"v4 existing pairs: {len(existing_pairs):,}")

    # Build MRCONSO string→CUI for phenotype text lookup
    import re
    def normalize(text):
        t = text.lower().strip()
        t = re.sub(r'[()\[\]{}]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    # For name-keyed features, we need phen text → CUI mapping
    # Reuse existing phenotype_scispacy_links cache for speed
    phen_links = json.load(open(MEDKG_ROOT / "kg" / "phenotype_scispacy_links.json"))

    # MRCONSO direct lookup (subset)
    from medkg_paths import UMLS_DIR
    PREFERRED_SABS = ["HPO", "SNOMEDCT_US", "MSH", "MEDCIN", "NCI", "ICD10CM"]
    SAB_PRIORITY = {s: i for i, s in enumerate(PREFERRED_SABS)}

    # Collect phen texts that we need to resolve (for diseases being aliased)
    need_resolve = set()
    for cui in ALIAS_MAP:
        for alt_cui in ALIAS_MAP[cui]:
            for p in feats_cui.get(alt_cui, []):
                need_resolve.add(normalize(p["phenotype"]))
        for name in NAME_LOOKUP.get(cui, []):
            for p in feats_name.get(name, []):
                need_resolve.add(normalize(p["phenotype"]))
    print(f"Phenotype strings needing CUI resolution: {len(need_resolve):,}")

    str2cui = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < 15 or parts[1] != "ENG": continue
            norm = normalize(parts[14])
            if norm not in need_resolve: continue
            cui = parts[0]; sab = parts[11]
            prio = SAB_PRIORITY.get(sab, 99)
            if norm in str2cui:
                old_cui, old_prio = str2cui[norm]
                if prio < old_prio: str2cui[norm] = (cui, prio)
            else:
                str2cui[norm] = (cui, prio)
    print(f"Resolved {len(str2cui):,} strings to CUI")

    # IDF for new edges - use v4 frequency
    phen_freq = Counter()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            phen_freq[v] += 1
    N_features = len(feats_cui)
    idf = {p: math.log(N_features / max(c, 1)) for p, c in phen_freq.items()}

    # Add edges from alias CUIs and name-keys
    added = 0; skipped = 0
    all_keys = set(ALIAS_MAP.keys()) | set(NAME_LOOKUP.keys())
    for ddx_cui in all_keys:
        all_alt_phens = []
        for alt_cui in ALIAS_MAP.get(ddx_cui, []):
            all_alt_phens.extend(feats_cui.get(alt_cui, []))
        for name in NAME_LOOKUP.get(ddx_cui, []):
            all_alt_phens.extend(feats_name.get(name, []))

        for p in all_alt_phens:
            text = normalize(p["phenotype"])
            # Resolve to CUI
            phen_cuis = set()
            if text in str2cui:
                phen_cuis.add(str2cui[text][0])
            for cui, _, _ in phen_links.get(text, []):
                phen_cuis.add(cui)

            score = p.get("score", 0.5)
            n_sources = p.get("n_sources", 1)
            agreement = 0.5 + 0.5 * min(n_sources, 5) / 5
            for pcui in phen_cuis:
                if pcui == ddx_cui: continue
                if (ddx_cui, pcui) in existing_pairs:
                    skipped += 1; continue
                if pcui not in G:
                    G.add_node(pcui, ntype="Phenotype", name=p["phenotype"], source="v7_alias")
                w = math.log1p(score * 10) * agreement * idf.get(pcui, 1.0)
                G.add_edge(ddx_cui, pcui, etype="HAS_PHENOTYPE", weight=w)
                existing_pairs.add((ddx_cui, pcui))
                added += 1

    print(f"Added {added:,} new alias-derived edges (skipped {skipped:,} dupes)")
    print(f"Final: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\nSaving to {GRAPH_OUT}")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)

    # Verify
    for ddx_cui in sorted(all_keys):
        if ddx_cui not in G: continue
        n_edges = sum(1 for _,_,e in G.out_edges(ddx_cui, data=True) if e.get("etype")=="HAS_PHENOTYPE")
        name = cui2name.get(ddx_cui, ddx_cui)
        print(f"  {name}: {n_edges} HAS_PHENOTYPE edges now")


if __name__ == "__main__":
    main()
