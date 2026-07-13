#!/usr/bin/env python3
"""v47: BM25 retrieval architecture.

Current scoring uses ad-hoc weighted sum with sqrt normalization.
BM25 is the IR standard for document-query matching:
  score(D, Q) = Σ IDF(t) * (f(t,D) * (k+1)) / (f(t,D) + k*(1 - b + b * |D|/avgD))

Treat each disease as a document (multiset of phens with weights = TF),
each patient as a query (set of pcuis).

Different from current model in:
- Length normalization via avgD (vs sqrt(total))
- Saturation: f(t,D)*(k+1)/(f(t,D)+k) — diminishing returns for high f
- Length penalty b: longer disease phen list penalized
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--hop2_decay", type=float, default=0.7)
    ap.add_argument("--idf_pow", type=float, default=0.5)
    ap.add_argument("--bm25_k", type=float, default=1.5)
    ap.add_argument("--bm25_b", type=float, default=0.75)
    ap.add_argument("--identity_boost", type=float, default=2.0)
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    dcs_set = set(dcs_list)

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)

    # d_q (Q-restricted) with hop2
    d_q = {}
    for d in dcs_list:
        if d not in G: d_q[d] = {}; continue
        phen_w = {}
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + ed.get("weight", 0)
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, ed2 in G.out_edges(p_direct, data=True):
                if ed2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + args.hop2_decay * dw * ed2.get("weight", 0)
        d_q[d] = {p: w for p, w in phen_w.items() if p in Q}

    # IDF
    phen_freq = Counter()
    for d, qp in d_q.items():
        for p in qp: phen_freq[p] += 1
    N_D = len(dcs_list)
    idf = {p: math.log(N_D / max(c, 1)) ** args.idf_pow for p, c in phen_freq.items()}

    # Document length (= total weight) per disease
    d_lengths = {d: sum(qp.values()) for d, qp in d_q.items()}
    avg_len = sum(d_lengths.values()) / max(len(d_lengths), 1)

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

    k1 = args.bm25_k
    b = args.bm25_b

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
                qp = d_q.get(d, {})
                length = d_lengths.get(d, 0)
                length_norm = (1 - b + b * length / avg_len) if avg_len > 0 else 1
                bm25_score = 0
                for p in pcuis:
                    if p not in qp: continue
                    f_p = qp[p]
                    bm25_score += idf.get(p, 0) * f_p * (k1 + 1) / (f_p + k1 * length_norm)
                if d in identity_diseases:
                    bm25_score += args.identity_boost
                scores[d] = bm25_score

            ranked = sorted(dcs_list, key=lambda d: -scores.get(d, -1e9))
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"v47 BM25 [k={args.bm25_k},b={args.bm25_b},ib={args.identity_boost}]: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
