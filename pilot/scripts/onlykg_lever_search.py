#!/usr/bin/env python3
"""Aggressive lever search: try many variants quickly to find +%p."""
from __future__ import annotations
import sys, json, csv, ast, math, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--p1", type=float, default=1.0)
    ap.add_argument("--p2", type=float, default=1.0)
    ap.add_argument("--p3", type=float, default=1.0)
    args = ap.parse_args()

    G = pickle.load(open(GRAPH, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))
    dcs_set = set(dcs_list)

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)

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
                    phen_w[p2] = phen_w.get(p2, 0) + 0.5 * dw * edata2.get("weight", 0)
        d_q[d] = {p: w for p, w in phen_w.items() if p in Q}

    # IDF on 49 disease universe
    phen_freq = Counter()
    for d, qp in d_q.items():
        for p in qp: phen_freq[p] += 1
    N_D = len(dcs_list)
    idf = {p: math.log(N_D / max(c, 1)) ** 0.5 for p, c in phen_freq.items()}
    d_q_idf = {d: {p: w * idf.get(p, 1.0) for p, w in qp.items()} for d, qp in d_q.items()}

    # Top-K phens per disease
    d_topk = {}
    for d, qp in d_q_idf.items():
        d_topk[d] = sorted(qp.keys(), key=lambda p: -qp[p])[:int(args.p2)] if qp else []

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
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis = get_pcuis(evs)
            identity_diseases = pcuis & dcs_set

            scores = {}
            for d in dcs_list:
                qp = d_q_idf.get(d, {})
                if not qp: scores[d] = -1e6; continue
                pos = sum(w for q, w in qp.items() if q in pcuis)
                total = sum(qp.values()) or 1

                if args.mode == "base":
                    s = pos / math.sqrt(total)
                elif args.mode == "topk_identity":
                    # Symptom identity: top-K phen matched bonus
                    topk = set(d_topk[d][:5])
                    topk_match = sum(1 for p in topk if p in pcuis)
                    s = pos / math.sqrt(total) + args.p1 * topk_match
                elif args.mode == "product":
                    # Multiplicative: product of (1 + matched weight ratio)
                    matched_phens = [p for p in pcuis if p in qp]
                    if not matched_phens: s = -1e6
                    else:
                        s = sum(math.log(1 + qp[p]) for p in matched_phens)
                        s = s / math.sqrt(total)
                elif args.mode == "count_norm":
                    n_matched = sum(1 for p in pcuis if p in qp)
                    n_total = len(qp)
                    s = (pos / math.sqrt(total)) * (n_matched / (n_total ** args.p1))
                elif args.mode == "coverage":
                    # Score by fraction of Q-phens covered
                    n_matched = sum(1 for p in qp if p in pcuis)
                    coverage = n_matched / max(len(qp), 1)
                    s = (pos / math.sqrt(total)) + args.p1 * coverage
                elif args.mode == "max_weight":
                    # Reward by max single weight match (most discriminative)
                    max_w = max((qp[p] for p in pcuis if p in qp), default=0)
                    s = (pos / math.sqrt(total)) + args.p1 * max_w
                elif args.mode == "log_match":
                    # log(1 + sum w) instead of sum w / sqrt
                    s = math.log(1 + pos) - 0.5 * math.log(total)

                if d in identity_diseases:
                    s += 1.5
                scores[d] = s

            ranked = sorted(dcs_list, key=lambda d: -scores.get(d, -1e9))
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"mode={args.mode} p1={args.p1} p2={args.p2}: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
