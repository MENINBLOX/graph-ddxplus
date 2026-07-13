#!/usr/bin/env python3
"""only-KG Stage 1 evaluation on RareBench (cross-benchmark).

RareBench: each patient has list of HPO terms (Phenotype) and target RareDisease
(OMIM/ORPHA IDs). Map HPO → CUI via hpo_umls_mapping.json; map disease →
UMLS CUI via disease_umls_mapping.json.
"""
from __future__ import annotations
import sys, json, math, time, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v2.pkl"


def build_extended_phenotypes(G, disease_cuis, hop2_decay=0.5):
    extended = {}
    for d in disease_cuis:
        if d not in G:
            extended[d] = {}; continue
        phen_w = {}
        for _, p, edata in G.out_edges(d, data=True):
            if edata.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + edata.get("weight", 0)
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, edata2 in G.out_edges(p_direct, data=True):
                if edata2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + hop2_decay * dw * edata2.get("weight", 0)
        extended[d] = phen_w
    return extended


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="HMS,LIRICAL,MME,RAMEDIS")
    ap.add_argument("--hop2_decay", type=float, default=0.5)
    args = ap.parse_args()

    print("Loading v2 graph + RareBench mappings...")
    with GRAPH.open("rb") as f:
        G = pickle.load(f)
    hpo_map = json.load(open("data/rarebench/hpo_umls_mapping.json"))["mapping"]
    dis_map = json.load(open("data/rarebench/disease_umls_mapping.json"))["mapping"]

    # Load all rarebench patients
    patients = []
    for ds in args.datasets.split(","):
        path = Path(f"data/rarebench/data/{ds}.jsonl")
        if not path.exists(): continue
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                hpos = r["Phenotype"]
                target = r["RareDisease"]
                target_cuis = []
                for t in target:
                    info = dis_map.get(t)
                    if info and info.get("umls_cui"):
                        target_cuis.append(info["umls_cui"])
                if not target_cuis: continue
                # Phenotype CUIs
                pcuis = set()
                for hp in hpos:
                    info = hpo_map.get(hp)
                    if isinstance(info, dict):
                        c = info.get("umls_cui")
                    elif isinstance(info, str):
                        c = info
                    else:
                        c = None
                    if c: pcuis.add(c)
                if pcuis:
                    patients.append((target_cuis, pcuis, ds))
    print(f"  Total patients: {len(patients)}")
    n_by_ds = Counter(p[2] for p in patients)
    for ds, c in n_by_ds.items():
        print(f"    {ds}: {c}")

    # Disease candidate set = union of target CUIs across all patients ∩ KG
    all_targets = set()
    for ts, _, _ in patients:
        all_targets.update(ts)
    candidates = sorted([c for c in all_targets if c in G])
    print(f"  Candidate diseases (target ∩ KG): {len(candidates)} of {len(all_targets)} unique targets")

    print("Pre-computing extended phenotypes...")
    extended = build_extended_phenotypes(G, candidates, args.hop2_decay)

    print(f"\nEvaluating only-KG on RareBench...")
    n = 0; c1=c3=c5=c10=0; rr_sum=0
    per_ds = {ds: [0, 0] for ds in n_by_ds}  # [@1, n]
    for target_cuis, pcuis, ds in patients:
        target_in_cand = [t for t in target_cuis if t in candidates]
        if not target_in_cand:
            n += 1
            per_ds[ds][1] += 1
            continue
        scores = {}
        for d in candidates:
            ext = extended.get(d, {})
            s = sum(ext.get(p, 0) for p in pcuis)
            norm = math.sqrt(len(ext)) if ext else 1
            scores[d] = s / max(norm, 1)
        ranked = sorted(candidates, key=lambda d: -scores.get(d, 0))
        # Best rank among target CUIs (some patients have multi-CUI targets)
        ranks = []
        for tc in target_in_cand:
            try: ranks.append(ranked.index(tc) + 1)
            except: ranks.append(len(ranked))
        rank = min(ranks)
        n += 1; per_ds[ds][1] += 1
        if rank == 1: c1 += 1; per_ds[ds][0] += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        rr_sum += 1.0 / rank

    print(f"\n=== only-KG Stage 1 on RareBench ({n} patients, {len(candidates)} candidate diseases) ===")
    print(f"  GTPA@1  = {100*c1/n:.2f}%")
    print(f"  GTPA@3  = {100*c3/n:.2f}%")
    print(f"  GTPA@5  = {100*c5/n:.2f}%")
    print(f"  GTPA@10 = {100*c10/n:.2f}%")
    print(f"  MRR     = {rr_sum/n:.4f}")
    print(f"\n  Per-dataset breakdown:")
    for ds, (c1d, nd) in per_ds.items():
        if nd > 0: print(f"    {ds}: {100*c1d/nd:.2f}% ({c1d}/{nd})")


if __name__ == "__main__":
    main()
