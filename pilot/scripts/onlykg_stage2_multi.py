#!/usr/bin/env python3
"""Stage 2 multi-signal reranking on top-K.

Signals:
  - Stage 1 score (Q-aware IDF + neg_core + identity)
  - Patient-coverage: |patient ∩ phens(D)| / |patient|
  - Disease-coverage: |patient ∩ phens(D)| / |phens(D)|
  - Match-weight: sum of edge weights for matches
  - Pathognomonic: # of matched phens that are unique to D in top-K
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v11.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--top_k", type=int, default=25)
    ap.add_argument("--w_cov", type=float, default=0.4)
    ap.add_argument("--w_disc", type=float, default=0.2)
    ap.add_argument("--w_path", type=float, default=0.0)
    ap.add_argument("--core_k", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--identity_boost", type=float, default=1.0)
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

    d_q = {}; d_all = {}
    for d in dcs_list:
        if d not in G: d_q[d] = {}; d_all[d] = set(); continue
        phen_w = {}
        for _, p, edata in G.out_edges(d, data=True):
            if edata.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + edata.get("weight", 0)
        all_phens = set(phen_w.keys())
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, edata2 in G.out_edges(p_direct, data=True):
                if edata2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + 0.5 * dw * edata2.get("weight", 0)
                    all_phens.add(p2)
        d_q[d] = {p: w for p, w in phen_w.items() if p in Q}
        d_all[d] = all_phens

    phen_freq = Counter()
    for d, qp in d_q.items():
        for p in qp: phen_freq[p] += 1
    N_D = len(dcs_list)
    idf = {p: math.log(N_D / max(c, 1)) ** 0.5 for p, c in phen_freq.items()}
    d_q_idf = {d: {p: w * idf.get(p, 1.0) for p, w in qp.items()} for d, qp in d_q.items()}
    d_core = {d: set(sorted(qp.keys(), key=lambda p: -qp[p])[:args.core_k]) for d, qp in d_q_idf.items()}

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

    def normalize_dict(d, keys):
        if not d: return {k: 0 for k in keys}
        max_v = max(d.get(k, 0) for k in keys) or 1
        return {k: d.get(k, 0) / abs(max_v) for k in keys}

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis = get_pcuis(evs)
            identity = pcuis & dcs_set

            s1_scores = {}
            for d in dcs_list:
                qp = d_q_idf.get(d, {})
                if not qp: s1_scores[d] = -1e6; continue
                pos = sum(w for q, w in qp.items() if q in pcuis)
                core = d_core[d]
                neg = sum(qp.get(c, 0) for c in core if c not in pcuis)
                total = sum(qp.values()) or 1
                s = (pos - args.alpha * neg) / math.sqrt(total)
                if d in identity: s += args.identity_boost
                s1_scores[d] = s
            ranked1 = sorted(dcs_list, key=lambda d: -s1_scores.get(d, -1e9))
            top_k = ranked1[:args.top_k]

            # Compute Stage 2 signals
            cov_scores = {}    # patient coverage
            disc_scores = {}   # disease coverage
            path_scores = {}   # pathognomonic
            top_k_all_phens = {d: d_all.get(d, set()) for d in top_k}
            # Compute "unique phen" per d (phens in d but not in other top_k diseases)
            all_top_k_phens = Counter()
            for d in top_k:
                for p in top_k_all_phens[d]: all_top_k_phens[p] += 1
            unique_to_d = {d: {p for p in top_k_all_phens[d] if all_top_k_phens[p] == 1} for d in top_k}

            for d in top_k:
                phens = top_k_all_phens[d]
                match = pcuis & phens
                cov_scores[d] = len(match) / max(len(pcuis), 1)  # patient coverage
                disc_scores[d] = len(match) / max(len(phens), 1)  # disease coverage
                path_scores[d] = len(pcuis & unique_to_d[d])

            s1_n = normalize_dict(s1_scores, top_k)
            cov_n = normalize_dict(cov_scores, top_k)
            disc_n = normalize_dict(disc_scores, top_k)
            path_n = normalize_dict(path_scores, top_k)

            w_s1 = 1 - args.w_cov - args.w_disc - args.w_path
            combined = {}
            for d in top_k:
                combined[d] = w_s1 * s1_n[d] + args.w_cov * cov_n[d] + args.w_disc * disc_n[d] + args.w_path * path_n[d]

            top_k_resorted = sorted(top_k, key=lambda d: -combined.get(d, -1e9))
            final = top_k_resorted + [d for d in ranked1 if d not in top_k]
            n += 1
            try: rank = final.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"cov={args.w_cov} disc={args.w_disc} path={args.w_path} ck={args.core_k} α={args.alpha} ib={args.identity_boost}: @1={100*c1/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
