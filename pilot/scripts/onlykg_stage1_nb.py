#!/usr/bin/env python3
"""only-KG Stage 1 with Naive Bayes scoring + negative evidence + disease prior.

Scoring:
  score(D | E) = log P(D) + sum_{p in phens(D), p in E} log P(p|D)
                          + α × sum_{p in phens(D), p not in E} log(1 - P(p|D))

Where:
  - P(D) from DDXPlus train (population prior — universal clinical Bayesian)
  - P(p|D) ∝ edge_weight(D, p), normalized so sum_p = 1
  - α controls negative evidence strength (default 0.3)

Patient evidence CUIs come from value-aware extraction (clean).
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
    ap.add_argument("--alpha", type=float, default=0.3, help="negative evidence weight")
    ap.add_argument("--use_prior", action="store_true", default=False)
    ap.add_argument("--use_neg", action="store_true", default=False)
    args = ap.parse_args()

    print(f"Loading graph {args.graph}...")
    with open(args.graph, "rb") as f:
        G = pickle.load(f)

    print(f"Loading value-aware evidence CUIs...")
    value_cuis = json.load(open(VALUE_CUIS))
    prior_by_name = json.load(open(DISEASE_PRIOR)) if DISEASE_PRIOR.exists() else {}

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))

    # Prior by CUI
    prior_cui = {}
    for fr_name, c in fr2cui.items():
        en_name = cui2name.get(c, "")
        prior_cui[c] = prior_by_name.get(en_name, 1.0/len(dcs_list))
    tot = sum(prior_cui.values()) or 1
    for c in prior_cui: prior_cui[c] /= tot

    # Pre-compute extended phenotypes (with hierarchy) and normalize to P(p|D)
    print(f"Pre-computing per-disease P(p|D) (hop2_decay={args.hop2_decay})...")
    p_p_given_d = {}  # {d: {p: prob}}
    for d in dcs_list:
        if d not in G:
            p_p_given_d[d] = {}; continue
        phen_w = {}
        # 1-hop
        for _, p, edata in G.out_edges(d, data=True):
            if edata.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + edata.get("weight", 0)
        # 2-hop via HIERARCHY
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, edata2 in G.out_edges(p_direct, data=True):
                if edata2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + args.hop2_decay * dw * edata2.get("weight", 0)
        # Normalize to P(p|D)
        total = sum(phen_w.values()) or 1
        p_p_given_d[d] = {p: w/total for p, w in phen_w.items()}

    def get_patient_cuis(evs):
        cuis = set()
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                ev_map = value_cuis.get(base, {})
                cuis.update(ev_map.get("_question", []))
                cuis.update(ev_map.get(val, []))
            else:
                ev_map = value_cuis.get(ev, {})
                cuis.update(ev_map.get("_question", []))
        return cuis

    print(f"\nEvaluating NB scoring (alpha={args.alpha}, use_prior={args.use_prior}, use_neg={args.use_neg})...")
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
                phens = p_p_given_d.get(d, {})
                if not phens:
                    scores[d] = -1e6 + (math.log(prior_cui.get(d, 1e-6)) if args.use_prior else 0)
                    continue
                pos = 0.0; neg = 0.0
                for p, prob in phens.items():
                    if p in pcuis:
                        pos += math.log(max(prob, 1e-6))
                    elif args.use_neg:
                        neg += math.log(max(1 - prob, 1e-6))
                s = pos + args.alpha * neg
                if args.use_prior:
                    s += math.log(max(prior_cui.get(d, 1e-6), 1e-6))
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
            if n % 5000 == 0:
                print(f"  {n}/{args.n} @1={100*c1/n:.2f}% @5={100*c5/n:.2f}% MRR={rr_sum/n:.3f} ({time.time()-t0:.0f}s)")

    print(f"\n=== only-KG NB scoring ({args.graph.split('/')[-1]}, alpha={args.alpha}) ===")
    print(f"  use_prior={args.use_prior}, use_neg={args.use_neg}")
    print(f"  GTPA@1  = {100*c1/n:.2f}%")
    print(f"  GTPA@3  = {100*c3/n:.2f}%")
    print(f"  GTPA@5  = {100*c5/n:.2f}%")
    print(f"  GTPA@10 = {100*c10/n:.2f}%")
    print(f"  MRR     = {rr_sum/n:.4f}")

    print("\nTop failures:")
    for d, fc_ in fail_per_d.most_common(10):
        tot = total_per_d[d]
        print(f"  {d:35s}  {fc_}/{tot} ({100*fc_/tot:.1f}%)")


if __name__ == "__main__":
    main()
