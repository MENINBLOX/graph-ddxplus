#!/usr/bin/env python3
"""Stage 2 reranker on v11 (rich graph with @10=89% recall).

Stage 1: Q-aware sqrt + IDF + identity + neg_core (best @1 ~42.5%)
Stage 2 (on top-K): pairwise discriminative match using FULL phens (not just Q).
  Score(D | patient, K) = sum_{p ∈ phens(D), p ∈ patient} w(D,p) × idf_full(p)

Final = α × stage1 + β × stage2 (top-K only)
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, time
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
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--stage2_weight", type=float, default=0.5)
    ap.add_argument("--idf_pow", type=float, default=0.5)
    ap.add_argument("--core_k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.25)
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

    # Stage 1: Q-restricted (with hierarchy)
    d_q = {}
    # Stage 2: ALL phens (no Q restriction)
    d_all = {}
    for d in dcs_list:
        if d not in G:
            d_q[d] = {}; d_all[d] = {}; continue
        phen_w = {}
        for _, p, edata in G.out_edges(d, data=True):
            if edata.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + edata.get("weight", 0)
        all_phens = dict(phen_w)
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, edata2 in G.out_edges(p_direct, data=True):
                if edata2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + 0.5 * dw * edata2.get("weight", 0)
        d_q[d] = {p: w for p, w in phen_w.items() if p in Q}
        d_all[d] = all_phens  # without hierarchy expansion for stage 2

    # IDF for Q
    phen_freq_q = Counter()
    for d, qp in d_q.items():
        for p in qp: phen_freq_q[p] += 1
    N_D = len(dcs_list)
    idf_q = {p: math.log(N_D / max(c, 1)) ** args.idf_pow for p, c in phen_freq_q.items()}
    d_q_idf = {d: {p: w * idf_q.get(p, 1.0) for p, w in qp.items()} for d, qp in d_q.items()}

    # IDF for ALL phens (stage 2)
    phen_freq_a = Counter()
    for d, ap in d_all.items():
        for p in ap: phen_freq_a[p] += 1
    idf_a = {p: math.log(N_D / max(c, 1)) for p, c in phen_freq_a.items()}

    d_core = {}
    for d, qp in d_q_idf.items():
        d_core[d] = set(sorted(qp.keys(), key=lambda p: -qp[p])[:args.core_k])

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
    s1_c1 = 0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis = get_pcuis(evs)
            identity = pcuis & dcs_set

            # Stage 1 score
            s1_scores = {}
            for d in dcs_list:
                qp = d_q_idf.get(d, {})
                if not qp: s1_scores[d] = -1e6; continue
                pos = sum(w for q, w in qp.items() if q in pcuis)
                core = d_core.get(d, set())
                neg = sum(qp.get(c, 0) for c in core if c not in pcuis)
                total = sum(qp.values()) or 1
                s = (pos - args.alpha * neg) / math.sqrt(total)
                if d in identity: s += args.identity_boost
                s1_scores[d] = s
            ranked1 = sorted(dcs_list, key=lambda d: -s1_scores.get(d, -1e9))
            if ranked1[0] == true_cui: s1_c1 += 1

            # Stage 2: rerank top_k by full-phen IDF-weighted match
            top_k = ranked1[:args.top_k]
            s2_scores = {}
            for d in top_k:
                ap = d_all.get(d, {})
                if not ap: s2_scores[d] = 0; continue
                pos2 = sum(w * idf_a.get(p, 1.0) for p, w in ap.items() if p in pcuis)
                total2 = sum(w * idf_a.get(p, 1.0) for p, w in ap.items()) or 1
                s2_scores[d] = pos2 / math.sqrt(total2)

            # Combined: normalize both to similar scale then add
            # Scale stage 1 / stage 2 by their respective max for top_k
            max1 = max(s1_scores[d] for d in top_k) or 1
            max2 = max(s2_scores[d] for d in top_k) or 1
            combined = {}
            for d in top_k:
                s1 = s1_scores[d] / abs(max1) if max1 != 0 else 0
                s2 = s2_scores[d] / abs(max2) if max2 != 0 else 0
                combined[d] = (1 - args.stage2_weight) * s1 + args.stage2_weight * s2

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

    print(f"top_k={args.top_k} s2_weight={args.stage2_weight} α={args.alpha} ib={args.identity_boost}:")
    print(f"  S1@1={100*s1_c1/n:.2f}% S2@1={100*c1/n:.2f} @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
