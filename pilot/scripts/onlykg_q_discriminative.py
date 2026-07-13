#!/usr/bin/env python3
"""Discriminative scoring: log P(p|D) - log P(p|not D).

For each evidence p that the patient has, reward D by how much MORE D
has p than the average disease. This naturally suppresses common
phenotypes (Coughing — present in many) and boosts specific ones.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--hop2_decay", type=float, default=0.5)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--norm", default="sqrt")
    args = ap.parse_args()

    G = pickle.load(open(MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl", "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if not isinstance(mapping, dict): continue
        for vkey, vcuis in mapping.items():
            if isinstance(vcuis, list): Q.update(vcuis)

    # d_q[d][p] = weight (with hierarchy expansion)
    d_q = {}
    for d in dcs_list:
        if d not in G: d_q[d] = {}; continue
        phen_w = {}
        for _, p, edata in G.out_edges(d, data=True):
            if edata.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + edata.get("weight", 0)
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, edata2 in G.out_edges(p_direct, data=True):
                if edata2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + args.hop2_decay * dw * edata2.get("weight", 0)
        d_q[d] = {p: w for p, w in phen_w.items() if p in Q}

    # Normalize per-disease weights to probabilities P(p|D)
    p_p_d = {}
    for d, qp in d_q.items():
        total = sum(qp.values()) or 1
        p_p_d[d] = {p: w/total for p, w in qp.items()}

    # Average probability across diseases for each p
    p_avg = defaultdict(list)
    for d in dcs_list:
        for p in Q:
            p_avg[p].append(p_p_d[d].get(p, 0.0))
    p_avg_mean = {p: sum(probs) / len(probs) for p, probs in p_avg.items()}

    # Discriminative score per (d, p): log P(p|D) - log P(p|avg)
    # Equivalent: log (P(p|D) / P(p|avg))
    eps = args.smoothing / len(dcs_list)
    log_disc = {}
    for d in dcs_list:
        log_disc[d] = {}
        for p, pr in p_p_d[d].items():
            avg = p_avg_mean.get(p, eps)
            ratio = (pr + eps) / (avg + eps)
            log_disc[d][p] = math.log(ratio)

    def get_pcuis(evs):
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

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    fail_per_d = Counter(); total_per_d = Counter()
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis = get_pcuis(evs)
            scores = {}
            for d in dcs_list:
                ld = log_disc.get(d, {})
                if not ld: scores[d] = -1e6; continue
                s = sum(ld.get(p, 0) for p in pcuis if p in ld)
                if args.norm == "sqrt":
                    denom = math.sqrt(len(d_q.get(d, {})) or 1)
                elif args.norm == "none":
                    denom = 1
                elif args.norm == "matched_count":
                    matched = sum(1 for p in pcuis if p in ld)
                    denom = math.sqrt(max(matched, 1))
                else: denom = 1
                scores[d] = s / max(denom, 1e-6)
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

    print(f"smoothing={args.smoothing} norm={args.norm}:")
    print(f"  @1={100*c1/n:.2f}%  @3={100*c3/n:.2f}%  @5={100*c5/n:.2f}%  @10={100*c10/n:.2f}%  MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
