#!/usr/bin/env python3
"""v103b — TF-IDF style scoring for attribute-rich KG.

변경 사항 (v103 → v103b):
1. Frequency 의존 줄임: tf = log(1 + n_mentions) instead of raw frequency
2. Phenotype IDF: discriminative weighting across diseases
3. Disease self-reference filter (Epiglottitis → epiglottitis 등 제거)
4. Total score (not mean), evidence convergence 반영
"""
from __future__ import annotations
import sys, json, pickle, math, argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from v103_eval_one_case import name_sim, attr_align, patient_seed42_json


def compute_phen_idf(G):
    """Phenotype IDF: log((N+1) / (df+1)) + 1.
    df = # diseases having this phenotype."""
    n_diseases = sum(1 for n in G.nodes if G.nodes[n].get("ntype") == "disease")
    df = defaultdict(int)
    for u, v, _ in G.edges(data=True):
        if G.nodes[u].get("ntype") == "disease":
            df[v] += 1
    idf = {phen: math.log((n_diseases + 1) / (df_v + 1)) + 1
           for phen, df_v in df.items()}
    return idf, n_diseases


def is_self_reference(prof_phen, disease_name):
    """Check if phenotype name is just the disease name (LLM over-extraction)."""
    p = prof_phen.lower().strip()
    d = disease_name.lower().strip()
    if p == d: return True
    if p in d or d in p: return True  # substring
    # Common LLM artifacts
    if p in {"infection","symptoms","reaction","disease"}: return True
    return False


def score_v103b(patient, disease_cui, G, phen_idf, alpha=0.5):
    """v103b TF-IDF scoring.

    score(D) = Σ over patient evidence:
        max over disease's phenotypes:
            name_sim * (alpha + (1-alpha)*attr_alignment) * tf * idf
    """
    if disease_cui not in G: return 0, []
    d_name = G.nodes[disease_cui].get("name","")
    total = 0
    matched = []
    for pat_ev in patient:
        best = 0
        best_prof = None
        for _, prof_phen, edge_attrs in G.out_edges(disease_cui, data=True):
            if is_self_reference(prof_phen, d_name): continue
            sim = name_sim(pat_ev["name"], prof_phen)
            if sim < 0.3: continue
            attr_a = attr_align(pat_ev["attributes"], edge_attrs)
            tf = math.log(1 + edge_attrs["n_mentions"])
            idf = phen_idf.get(prof_phen, 1.0)
            combined = sim * (alpha + (1-alpha)*attr_a) * tf * idf
            if combined > best:
                best = combined
                best_prof = prof_phen
        if best > 0:
            total += best
            matched.append((pat_ev["name"], best_prof, round(best, 3)))
    return total, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="weight of base name match (1-alpha = attr weight)")
    args = ap.parse_args()

    G = pickle.load(open("pilot/data/onlykg_graph_v103.pkl", "rb"))
    phen_idf, n_dis = compute_phen_idf(G)
    print(f"=== v103b: {n_dis} diseases, IDF range [{min(phen_idf.values()):.2f}, {max(phen_idf.values()):.2f}]", flush=True)

    cui2name = {cui: G.nodes[cui].get("name","?")
                for cui in G.nodes if G.nodes[cui].get("ntype")=="disease"}

    patient = patient_seed42_json()
    truth_cui = "C0685898"
    print(f"\nseed=42 anaphylaxis patient, truth={cui2name.get(truth_cui)}\n", flush=True)

    scores = [(score_v103b(patient, d, G, phen_idf, args.alpha), d)
              for d in cui2name]
    scores = [(s, m, d) for (s, m), d in scores]
    scores.sort(reverse=True, key=lambda x: x[0])

    print("--- v103b Top 10 ---")
    rank_truth = None
    for i, (s, matched, d_cui) in enumerate(scores[:10]):
        mark = " ⭐ TRUTH" if d_cui == truth_cui else ""
        print(f"  [{i+1}] {cui2name[d_cui]:<40}: {s:.3f}{mark}")
        for pn, prof, sc in matched[:3]:
            print(f"      {pn:<15} ↔ {prof:<30}: {sc}")
        print()

    for i, (s, _, d) in enumerate(scores):
        if d == truth_cui:
            rank_truth = i+1; break
    print(f"\n=== Truth rank: {rank_truth} / {len(scores)} (alpha={args.alpha}) ===", flush=True)
    print(f"Baseline: v95_full=1, v101=5, v103=8, v103b=?", flush=True)


if __name__ == "__main__":
    main()
