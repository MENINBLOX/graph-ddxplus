#!/usr/bin/env python3
"""only-KG Stage 1: questionnaire-universe Bayesian scoring.

For each candidate disease D and patient evidence set E:
  score(D) = sum_{q ∈ Q ∩ phens(D), q ∈ E}  weight(D,q)      # matched positive
           - α × sum_{q ∈ Q ∩ phens(D), q ∉ E}  weight(D,q)   # mismatched negative
           + β × log P(D)                                       # disease prior

Q = DDXPlus questionnaire CUI universe (335 CUIs).

Rationale:
  - Original scoring only summed positive matches → no penalty for missing core symptoms
  - Restricting negative to Q ∩ phens(D) avoids unfair penalty for non-questionnaire CUIs
  - Disease prior P(D) is a standard clinical Bayesian factor (population statistic)
"""
from __future__ import annotations
import sys, json, csv, ast, math, time, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
DISEASE_PRIOR = MEDKG_ROOT / "kg" / "disease_prior_by_cui.json"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl"))
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--hop2_decay", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.5, help="negative evidence weight")
    ap.add_argument("--beta", type=float, default=1.0, help="prior weight")
    ap.add_argument("--use_prior", action="store_true", default=False)
    ap.add_argument("--use_neg", action="store_true", default=False)
    args = ap.parse_args()

    print(f"Loading graph {args.graph}...")
    with open(args.graph, "rb") as f:
        G = pickle.load(f)
    value_cuis = json.load(open(VALUE_CUIS))
    prior_by_name = json.load(open(DISEASE_PRIOR)) if DISEASE_PRIOR.exists() else {}

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))

    # Disease prior
    prior_cui = {}
    for fr_name, c in fr2cui.items():
        en_name = cui2name.get(c, "")
        prior_cui[c] = prior_by_name.get(en_name, 1.0/len(dcs_list))
    tot = sum(prior_cui.values()) or 1
    for c in prior_cui: prior_cui[c] /= tot

    # Questionnaire CUI universe (Q)
    Q = set()
    for ev_name, mapping in value_cuis.items():
        if not isinstance(mapping, dict): continue
        for vkey, vcuis in mapping.items():
            if isinstance(vcuis, list):
                Q.update(vcuis)
    print(f"Questionnaire CUI universe |Q| = {len(Q)}")

    # Per-disease extended phenotype weights (with hierarchy)
    print(f"Pre-computing extended phenotypes (hop2_decay={args.hop2_decay})...")
    ext = {}
    for d in dcs_list:
        if d not in G:
            ext[d] = {}; continue
        phen_w = {}
        for _, p, edata in G.out_edges(d, data=True):
            if edata.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + edata.get("weight", 0)
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, edata2 in G.out_edges(p_direct, data=True):
                if edata2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + args.hop2_decay * dw * edata2.get("weight", 0)
        ext[d] = phen_w

    # Per disease: precompute Q ∩ phens(D) and its total weight (for normalization)
    d_q_phens = {}  # d → {q: weight}
    for d in dcs_list:
        d_q_phens[d] = {p: w for p, w in ext[d].items() if p in Q}

    def get_patient_cuis(evs):
        cuis = set()
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                m = value_cuis.get(base, {})
                cuis.update(m.get("_question", []))
                cuis.update(m.get(val, []))
            else:
                m = value_cuis.get(ev, {})
                cuis.update(m.get("_question", []))
        return cuis

    print(f"\nEvaluating (alpha={args.alpha}, beta={args.beta}, use_prior={args.use_prior}, use_neg={args.use_neg})...")
    t0 = time.time()
    n = 0; c1=c3=c5=c10=0; rr_sum=0
    fail_per_d = Counter(); total_per_d = Counter()
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis = get_patient_cuis(evs)
            scores = {}
            for d in dcs_list:
                qp = d_q_phens.get(d, {})
                if not qp:
                    scores[d] = -1e6
                    continue
                pos = sum(w for q, w in qp.items() if q in pcuis)
                neg = sum(w for q, w in qp.items() if q not in pcuis) if args.use_neg else 0
                # Normalize by sqrt of total to reduce bias
                norm = math.sqrt(sum(qp.values())) or 1
                s = (pos - args.alpha * neg) / norm
                if args.use_prior:
                    s += args.beta * math.log(max(prior_cui.get(d, 1e-6), 1e-6))
                scores[d] = s
            ranked = sorted(dcs_list, key=lambda d: -scores.get(d, -1e9))
            true_name = cui2name.get(true_cui, "?")
            n += 1; total_per_d[true_name] += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            else: fail_per_d[true_name] += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"\n=== only-KG Q-aware scoring ===")
    print(f"  graph={args.graph.split('/')[-1]}")
    print(f"  alpha={args.alpha}, beta={args.beta}, use_prior={args.use_prior}, use_neg={args.use_neg}")
    print(f"  GTPA@1  = {100*c1/n:.2f}%")
    print(f"  GTPA@3  = {100*c3/n:.2f}%")
    print(f"  GTPA@5  = {100*c5/n:.2f}%")
    print(f"  GTPA@10 = {100*c10/n:.2f}%")
    print(f"  MRR     = {rr_sum/n:.4f}")

    if c1 / n < 0.5:
        print("\nTop failures (>5%):")
        for d, fc_ in fail_per_d.most_common(10):
            tot = total_per_d[d]
            print(f"  {d:35s}  {fc_}/{tot} ({100*fc_/tot:.1f}%)")


if __name__ == "__main__":
    main()
