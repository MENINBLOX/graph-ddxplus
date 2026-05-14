#!/usr/bin/env python3
"""v28 + Stage 2 pairwise discrimination for top-K candidates.

After Stage 1, if score gap between top-1 and top-2 is small,
run pairwise discrimination using phens UNIQUE to each disease.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--hop2_decay", type=float, default=0.7)
    ap.add_argument("--idf_pow", type=float, default=0.5)
    ap.add_argument("--core_k", type=int, default=35)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--identity_boost", type=float, default=1.5)
    ap.add_argument("--sig_k", type=int, default=10)
    ap.add_argument("--sig_w", type=float, default=9.0)
    ap.add_argument("--pairwise_top_k", type=int, default=3, help="apply pairwise to top-K candidates")
    ap.add_argument("--pairwise_gap_threshold", type=float, default=0.5, help="trigger pairwise if gap < threshold")
    ap.add_argument("--pairwise_w", type=float, default=2.0, help="weight of pairwise score")
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    cui2name = {icd[dn]["cui"]: dn for dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    dcs_set = set(dcs_list)

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)

    # Build d_q_idf and d_sig
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

    phen_freq = Counter()
    for d, qp in d_q.items():
        for p in qp: phen_freq[p] += 1
    N_D = len(dcs_list)
    idf = {p: math.log(N_D / max(c, 1)) ** args.idf_pow for p, c in phen_freq.items()}
    d_q_idf = {d: {p: w * idf.get(p, 1.0) for p, w in qp.items()} for d, qp in d_q.items()}
    d_sig = {d: set(sorted(qp.keys(), key=lambda p: -qp[p])[:args.sig_k]) for d, qp in d_q_idf.items()}
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

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    n_pairwise = 0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis = get_pcuis(evs)
            identity_diseases = pcuis & dcs_set

            # Stage 1: standard v28 scoring
            scores = {}
            for d in dcs_list:
                qp = d_q_idf.get(d, {})
                pos = sum(w for q, w in qp.items() if q in pcuis)
                core = d_core.get(d, set())
                neg = sum(qp.get(c, 0) for c in core if c not in pcuis)
                s1 = pos - args.alpha * neg
                total = sum(qp.values()) if qp else 1
                s1 = s1 / (math.sqrt(total) or 1)
                sig = d_sig.get(d, set())
                if sig:
                    s1 += args.sig_w * (sum(1 for p in sig if p in pcuis) / len(sig))
                if d in identity_diseases:
                    s1 += args.identity_boost
                scores[d] = s1

            ranked = sorted(dcs_list, key=lambda d: -scores.get(d, -1e9))

            # Pairwise discrimination among top-K
            top_k = ranked[:args.pairwise_top_k]
            top1_score = scores[top_k[0]]
            top2_score = scores[top_k[1]] if len(top_k) > 1 else -1e9
            gap = top1_score - top2_score

            if gap < args.pairwise_gap_threshold and len(top_k) >= 2:
                n_pairwise += 1
                # Compute pairwise score for each pair in top-K
                pairwise_adj = {d: 0.0 for d in top_k}
                for i in range(len(top_k)):
                    for j in range(i+1, len(top_k)):
                        d1, d2 = top_k[i], top_k[j]
                        qp1 = d_q_idf.get(d1, {})
                        qp2 = d_q_idf.get(d2, {})
                        # Unique phens for d1 (in d1 but not d2)
                        unique_d1 = set(qp1.keys()) - set(qp2.keys())
                        unique_d2 = set(qp2.keys()) - set(qp1.keys())
                        match_d1 = sum(qp1[p] for p in unique_d1 if p in pcuis)
                        match_d2 = sum(qp2[p] for p in unique_d2 if p in pcuis)
                        if match_d1 > match_d2:
                            pairwise_adj[d1] += args.pairwise_w * (match_d1 - match_d2)
                        else:
                            pairwise_adj[d2] += args.pairwise_w * (match_d2 - match_d1)

                # Apply pairwise adj
                top_k_scores = {d: scores[d] + pairwise_adj.get(d, 0) for d in top_k}
                # Re-rank top-K
                reranked = sorted(top_k, key=lambda d: -top_k_scores[d])
                # Replace top-K in ranked list
                ranked = reranked + ranked[args.pairwise_top_k:]

            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"\nResults ({n} patients):")
    print(f"  @1={100*c1/n:.2f}%  @3={100*c3/n:.2f}%  @5={100*c5/n:.2f}%  @10={100*c10/n:.2f}%  MRR={rr_sum/n:.4f}")
    print(f"  Pairwise triggered: {n_pairwise:,} ({100*n_pairwise/n:.1f}%)")
    print(f"  Hp: ck={args.core_k} α={args.alpha} ib={args.identity_boost} sig_k={args.sig_k} sig_w={args.sig_w} pw_K={args.pairwise_top_k} pw_w={args.pairwise_w} pw_gap={args.pairwise_gap_threshold}")


if __name__ == "__main__":
    main()
