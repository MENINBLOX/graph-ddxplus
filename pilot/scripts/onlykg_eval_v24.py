#!/usr/bin/env python3
"""Canonical only-KG DDXPlus evaluator with v23-SOTA scoring stack.

Scoring stack (verified at 53.43% on v23 SOTA):
1. Q-aware: restrict to questionnaire CUI universe
2. hop2_decay hierarchy propagation (0.8)
3. IDF reweighting (pow=0.5)
4. core_k negative evidence (k=28, alpha=0.25)
5. identity_boost (medical-history→disease, 2.0)
6. signature_match boost (sig_k=10, sig_w=0.5)
7. Stage 2 coverage rerank (cw=0.56)

Usage:
  python onlykg_eval_v24.py --graph /mnt/medkg/kg/onlykg_graph_v24_bfs.pkl
"""
from __future__ import annotations
import os, sys, json, csv, ast, math, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=30000)
    # Stage 1 hyperparams (v23 SOTA defaults)
    ap.add_argument("--hop2_decay", type=float, default=0.8)
    ap.add_argument("--idf_pow", type=float, default=0.5)
    ap.add_argument("--core_k", type=int, default=28)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--identity_boost", type=float, default=2.0)
    ap.add_argument("--sig_k", type=int, default=10)
    ap.add_argument("--sig_w", type=float, default=0.5)
    ap.add_argument("--cw", type=float, default=0.56, help="Stage 2 coverage weight")
    ap.add_argument("--use_3channel", action="store_true", help="use 3-channel Stage 2 (0.44*s1 + 0.30*cov + 0.26*PR-cov)")
    ap.add_argument("--w_s1", type=float, default=0.44)
    ap.add_argument("--w_cov", type=float, default=0.30)
    ap.add_argument("--w_prcov", type=float, default=0.26)
    ap.add_argument("--pr_universe", default="pilot/data/pr_universe.json")
    ap.add_argument("--q_universe", default="", help="custom Q universe (else use questionnaire value CUIs)")
    ap.add_argument("--expand_patient_cuis", action="store_true", help="expand patient CUIs via UMLS")
    ap.add_argument("--q_to_phen", default="pilot/data/q_to_phen_umls.json")
    args = ap.parse_args()

    print(f"Loading graph {args.graph}...", flush=True)
    G = pickle.load(open(args.graph, "rb"))
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges", flush=True)
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))
    dcs_set = set(dcs_list)
    print(f"  49 DDXPlus diseases", flush=True)

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)
    if args.q_universe and Path(args.q_universe).exists():
        Q = set(json.load(open(args.q_universe)))
        print(f"  Q (custom from {args.q_universe}): {len(Q):,} CUIs", flush=True)
    else:
        print(f"  Q (questionnaire universe): {len(Q):,} CUIs", flush=True)

    # PR universe for PR-cov channel
    PR = set()
    if args.use_3channel and Path(args.pr_universe).exists():
        PR = set(json.load(open(args.pr_universe)))
        print(f"  PR universe: {len(PR):,} CUIs", flush=True)

    # Per-disease Q-restricted phen weights with hop2 propagation
    d_q = {}
    for d in dcs_list:
        if d not in G: d_q[d] = {}; continue
        phen_w = {}
        for _, p, edata in G.out_edges(d, data=True):
            if edata.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + edata.get("weight", 0)
        # 2-hop hierarchy propagation
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, edata2 in G.out_edges(p_direct, data=True):
                if edata2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + args.hop2_decay * dw * edata2.get("weight", 0)
        d_q[d] = {p: w for p, w in phen_w.items() if p in Q}

    # IDF on Q-restricted phens
    phen_freq = Counter()
    for d, qp in d_q.items():
        for p in qp: phen_freq[p] += 1
    N_D = len(dcs_list)
    idf = {p: math.log(N_D / max(c, 1)) ** args.idf_pow for p, c in phen_freq.items()}
    d_q_idf = {d: {p: w * idf.get(p, 1.0) for p, w in qp.items()} for d, qp in d_q.items()}

    # Core-K for negative evidence
    d_core = {}
    for d, qp in d_q_idf.items():
        d_core[d] = set(sorted(qp.keys(), key=lambda p: -qp[p])[:args.core_k])

    # Signature: top-sig_k by IDF*weight
    d_sig = {}
    for d, qp in d_q_idf.items():
        ranked = sorted(qp.items(), key=lambda kv: -kv[1])
        d_sig[d] = set(p for p, _ in ranked[:args.sig_k])

    q_to_phen = {}
    if args.expand_patient_cuis and Path(args.q_to_phen).exists():
        raw = json.load(open(args.q_to_phen))
        q_to_phen = {k: set(v) for k, v in raw.items()}
        print(f"  q→phen UMLS expansion: {len(q_to_phen):,}", flush=True)

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
        # Expand via UMLS if requested
        if q_to_phen:
            expanded = set(cuis)
            for c in cuis:
                expanded.update(q_to_phen.get(c, set()))
            return expanded
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
            cov_scores = {}
            for d in dcs_list:
                qp = d_q_idf.get(d, {})
                # Stage 1: pos - alpha * neg(core_k missing) + sig boost + identity boost
                pos = sum(w for q, w in qp.items() if q in pcuis)
                core = d_core.get(d, set())
                neg = sum(qp.get(c, 0) for c in core if c not in pcuis)
                s1 = pos - args.alpha * neg
                total = sum(qp.values()) if qp else 1
                s1 = s1 / (math.sqrt(total) or 1)

                # Signature match boost (fraction of top-sig_k matched)
                sig = d_sig.get(d, set())
                if sig:
                    sig_match = sum(1 for p in sig if p in pcuis) / len(sig)
                    s1 += args.sig_w * sig_match

                # Identity boost (medical history → disease)
                if d in identity_diseases:
                    s1 += args.identity_boost

                # Stage 2 coverage: fraction of pcuis covered by disease phens
                if pcuis and qp:
                    cov = sum(1 for p in pcuis if p in qp) / len(pcuis)
                else:
                    cov = 0
                cov_scores[d] = cov

                # PR-cov: same as cov but restricted to PR universe intersected
                if PR and pcuis and qp:
                    pr_pcuis = pcuis & PR
                    pr_qp = {p: w for p, w in qp.items() if p in PR}
                    if pr_pcuis and pr_qp:
                        prcov = sum(1 for p in pr_pcuis if p in pr_qp) / max(len(pr_pcuis), 1)
                    else:
                        prcov = 0
                else:
                    prcov = 0

                scores[d] = (s1, cov, prcov)

            # Normalize each channel (z-score-like) then combine
            if args.use_3channel:
                # 3-channel convex combination
                final = {}
                for d in dcs_list:
                    s1, cov, prcov = scores[d]
                    final[d] = args.w_s1 * s1 + args.w_cov * cov + args.w_prcov * prcov
            else:
                final = {d: scores[d][0] + args.cw * scores[d][1] for d in dcs_list}
            ranked = sorted(dcs_list, key=lambda d: -final.get(d, -1e9))
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
    print(f"  Hyperparams: hop2={args.hop2_decay} idf_pow={args.idf_pow} ck={args.core_k} α={args.alpha} ib={args.identity_boost} sig_k={args.sig_k} sig_w={args.sig_w} cw={args.cw}")


if __name__ == "__main__":
    main()
