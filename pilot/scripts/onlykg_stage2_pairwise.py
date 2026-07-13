#!/usr/bin/env python3
"""only-KG Stage 2: pairwise discriminative reranking on top-K candidates.

Stage 1 produces top-K candidates. Stage 2 reranks by pairwise tournament:
  For each pair (Di, Dj) in top-K:
    disc_phens_i = phenotypes where w(Di,p) > w(Dj,p) + margin
    disc_phens_j = phenotypes where w(Dj,p) > w(Di,p) + margin
    Patient supports Di if sum_{p in disc_phens_i ∩ patient} w(Di,p) - w(Dj,p) > 0
  Winner of pair gets a vote.
  Final ranking by pair wins, breaking ties with Stage 1 score.

This explicitly tackles confounder cluster confusion (Bronchitis vs Bronchiolitis etc).
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, time
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--hop2_decay", type=float, default=0.5)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--vote_weight", type=float, default=1.0)
    args = ap.parse_args()

    print("Loading v4 graph + value-aware CUIs...")
    G = pickle.load(open(GRAPH, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)

    # Compute Q-phens with hierarchy
    print("Pre-computing per-disease Q-phens (with hierarchy)...")
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

    def stage1_score(pcuis):
        scores = {}
        for d in dcs_list:
            qp = d_q.get(d, {})
            if not qp: scores[d] = -1e6; continue
            pos = sum(w for q, w in qp.items() if q in pcuis)
            total = sum(qp.values())
            scores[d] = pos / (math.sqrt(total) or 1)
        return scores

    def stage2_rerank(top_k_d, pcuis, stage1):
        """Pairwise tournament among top_k_d."""
        wins = Counter()
        for i in range(len(top_k_d)):
            for j in range(i+1, len(top_k_d)):
                di, dj = top_k_d[i], top_k_d[j]
                qi, qj = d_q.get(di, {}), d_q.get(dj, {})
                # All phens that appear in either
                all_phens = set(qi.keys()) | set(qj.keys())
                # Discriminative for Di vs Dj
                score_i = 0; score_j = 0
                for p in all_phens & pcuis:
                    wi = qi.get(p, 0); wj = qj.get(p, 0)
                    diff = wi - wj
                    if diff > args.margin: score_i += diff
                    elif diff < -args.margin: score_j -= diff
                if score_i > score_j: wins[di] += 1
                elif score_j > score_i: wins[dj] += 1
        # Final: stage1 + vote_weight * wins
        out = {d: stage1.get(d, -1e6) + args.vote_weight * wins.get(d, 0)
               for d in top_k_d}
        return out

    print(f"Evaluating Stage 2 (top_k={args.top_k}, margin={args.margin}, vote_weight={args.vote_weight})...")
    t0 = time.time()
    n = 0; c1=c3=c5=c10=0; rr_sum=0
    s1_c1 = 0  # Stage 1 baseline @1
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis = get_pcuis(evs)
            s1 = stage1_score(pcuis)
            ranked1 = sorted(dcs_list, key=lambda d: -s1.get(d, -1e9))
            top_k = ranked1[:args.top_k]
            # Stage 1 @1 tracking
            if ranked1[0] == true_cui: s1_c1 += 1
            # Stage 2 rerank
            s2 = stage2_rerank(top_k, pcuis, s1)
            top_k_resorted = sorted(top_k, key=lambda d: -s2.get(d, -1e9))
            # Final ranked: stage2 results then remaining from stage1
            final = top_k_resorted + [d for d in ranked1 if d not in top_k]
            n += 1
            try: rank = final.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank
            if n % 5000 == 0:
                print(f"  {n}/{args.n} S1@1={100*s1_c1/n:.2f}% S2@1={100*c1/n:.2f}% MRR={rr_sum/n:.3f} ({time.time()-t0:.0f}s)")

    print(f"\n=== Stage 2 pairwise rerank (top_k={args.top_k}, margin={args.margin}, vote_w={args.vote_weight}) ===")
    print(f"  Stage 1 @1 (baseline): {100*s1_c1/n:.2f}%")
    print(f"  Stage 2 @1:            {100*c1/n:.2f}%")
    print(f"  Stage 2 @3:            {100*c3/n:.2f}%")
    print(f"  Stage 2 @5:            {100*c5/n:.2f}%")
    print(f"  Stage 2 @10:           {100*c10/n:.2f}%")
    print(f"  Stage 2 MRR:           {rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
