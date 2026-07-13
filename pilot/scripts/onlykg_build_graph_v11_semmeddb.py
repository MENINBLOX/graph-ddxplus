#!/usr/bin/env python3
"""v11: merge SemMedDB DDXPlus-related predications.

SemMedDB = PubMed-derived semantic predications (auto-extracted by SemRep).
Filter for disease-symptom relations (MANIFESTATION_OF, CAUSES, etc.) involving
DDXPlus 49 disease CUIs (+ aliases). Weight by # of supporting PMIDs.
"""
from __future__ import annotations
import sys, json, math, pickle, gzip, csv, time
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH_V11 = MEDKG_ROOT / "kg" / "onlykg_graph_v10.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v12.pkl"
SEMMEDDB = "/windows/data/semmeddb/semmedVER43_2024_R_PREDICATION.csv.gz"

# Predicates relevant for disease→symptom relations
DISEASE_SYM_PREDS = {
    "MANIFESTATION_OF": 1.0,   # symptom IS manifestation of disease (strongest)
    "PRODUCES": 0.9,            # disease produces symptom
    "CAUSES": 0.8,              # disease causes symptom
    "COEXISTS_WITH": 0.6,       # weaker association
    "COMPLICATES": 0.7,
    "PRECEDES": 0.5,
    "ASSOCIATED_WITH": 0.5,
    "AFFECTS": 0.4,             # weakest signal
}

# DDXPlus disease CUIs + manual aliases (broader concepts often used in PubMed)
ALIAS_TO_DDX = {
    "C0019693": "C0001175",  # HIV Infections → HIV (initial infection)
    "C0151744": "C1304447",  # Acute MI → NSTEMI/STEMI
    "C0024117": "C0741421",  # COPD → Acute COPD exacerbation
    "C0016659": "C0035525",  # Rib Fractures → Spontaneous rib fracture
    "C0013604": "C0013609",  # Edema → Localized edema
    "C0030297": "C0153466",  # Pancreatic neoplasm → Pancreatic neoplasm
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_pmid", type=int, default=2, help="min PMID support per edge")
    args = ap.parse_args()

    print("Loading v10...")
    G = pickle.load(open(GRAPH_V11, "rb"))
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    ddx_cuis = {info["cui"] for info in icd.values() if "cui" in info}
    target_cuis = ddx_cuis | set(ALIAS_TO_DDX.keys())
    print(f"Target DDXPlus CUIs (+aliases): {len(target_cuis)}")

    # Scan SemMedDB
    print(f"\nScanning SemMedDB...")
    edges = defaultdict(lambda: {"pmids": set(), "preds": Counter(), "name": ""})
    t0 = time.time()
    n_total = 0
    with gzip.open(SEMMEDDB, "rt", encoding="utf-8", errors="ignore") as f:
        for parts in csv.reader(f):
            n_total += 1
            if n_total % 20000000 == 0:
                print(f"  {n_total:,} lines scanned, {len(edges):,} pairs ({time.time()-t0:.0f}s)")
            if len(parts) < 13: continue
            pmid = parts[2]
            pred = parts[3]
            if pred not in DISEASE_SYM_PREDS: continue
            s_cui = parts[4]; s_name = parts[5]
            o_cui = parts[8]; o_name = parts[9]
            # disease in subject (disease→symptom)
            if s_cui in target_cuis and o_cui not in target_cuis:
                ddx = ALIAS_TO_DDX.get(s_cui, s_cui)
                k = (ddx, o_cui)
                edges[k]["pmids"].add(pmid)
                edges[k]["preds"][pred] += 1
                edges[k]["name"] = o_name
            # symptom in subject (symptom MANIFESTATION_OF disease)
            elif pred == "MANIFESTATION_OF" and o_cui in target_cuis and s_cui not in target_cuis:
                ddx = ALIAS_TO_DDX.get(o_cui, o_cui)
                k = (ddx, s_cui)
                edges[k]["pmids"].add(pmid)
                edges[k]["preds"][pred] += 1
                edges[k]["name"] = s_name
    print(f"  Done: {n_total:,} lines, {len(edges):,} unique (D,P) pairs ({time.time()-t0:.0f}s)")

    # Filter by min PMID support
    filtered = {k: v for k, v in edges.items() if len(v["pmids"]) >= args.min_pmid}
    print(f"  Filtered (≥{args.min_pmid} PMIDs): {len(filtered):,} pairs")

    # Existing pairs in v10
    existing_pairs = set()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            existing_pairs.add((u, v))
    print(f"  v10 existing pairs: {len(existing_pairs):,}")

    # IDF
    new_freq = Counter()
    for k in filtered:
        new_freq[k[1]] += 1
    full_freq = Counter()
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") == "HAS_PHENOTYPE":
            full_freq[v] += 1
    for p, c in new_freq.items():
        full_freq[p] += c
    N = 19000
    idf = {p: math.log(N / max(c, 1)) for p, c in full_freq.items()}

    # Add edges
    added = 0; skipped_dup = 0
    for (d, p), info in filtered.items():
        if (d, p) in existing_pairs:
            skipped_dup += 1; continue
        if p not in G:
            G.add_node(p, ntype="Phenotype", name=info["name"], source="v11_semmeddb")
        # Weight by predicate quality × PMID support × IDF
        pred_q = max(DISEASE_SYM_PREDS.get(pr, 0.3) for pr in info["preds"])
        pmid_q = math.log1p(len(info["pmids"]))
        w = pred_q * pmid_q * idf.get(p, 1.0)
        G.add_edge(d, p, etype="HAS_PHENOTYPE", weight=w, source="semmeddb")
        added += 1
    print(f"\nAdded {added:,} new SemMedDB edges (skipped {skipped_dup:,} dupes)")
    print(f"Final: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\nSaving to {GRAPH_OUT}")
    with GRAPH_OUT.open("wb") as f:
        pickle.dump(G, f)


if __name__ == "__main__":
    main()
