#!/usr/bin/env python3
"""v103 KG evaluated with the proven v71 self-aware-negative algorithm.

Purpose: fair comparison. Same scoring algorithm as the universal SOTA
(onlykg_eval_v71_selfaware.py, ~62% on v95_full), swapping ONLY the KG content
(v103 source-grounded property graph) so any delta reflects content quality,
not the scoring method.

v103 KG edge schema: HAS_PHENOTYPE with attrs
  {n_mentions, frequency_in_abstracts, location_dist, severity_dist,
   onset_dist, character_dist}
We map edge weight = frequency_in_abstracts (empirical P(E|D) proxy, grounded).

Optionally (--attr) multiply the matched-phenotype contribution by attribute
alignment between patient evidence attributes and the disease edge's
attribute distributions — the v103 hypothesis that attributes discriminate.
"""
from __future__ import annotations
import sys, json, math, pickle, argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from onlykg_eval_v71_selfaware import (
    compute_idf, reweight, precompute_signal_v71,
    load_ddxplus_full, score, VALUE_CUIS,
)


def build_profile_v103(G, dcs_list, kappa=20.0, top_k=80, weight_key="frequency"):
    """profile[d][pcui] = w/(w+kappa), w aggregated from v103 HAS_PHENOTYPE edges."""
    profile = {}
    all_evs = set()
    for d in dcs_list:
        if d not in G:
            profile[d] = {}
            continue
        ed_w = defaultdict(float)
        for _, pcui, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE":
                continue
            w = ed.get(weight_key, 0.0)
            if weight_key == "n_mentions":
                w = ed.get("n_mentions", 0.0)
            ed_w[pcui] += w
        prof = {p: w / (w + kappa) for p, w in ed_w.items() if w > 0}
        if top_k and len(prof) > top_k:
            prof = dict(sorted(prof.items(), key=lambda x: -x[1])[:top_k])
        profile[d] = prof
        all_evs.update(prof.keys())
    return profile, all_evs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--top_k", type=int, default=80)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--weight_key", default="frequency",
                    choices=["frequency", "n_mentions"])
    ap.add_argument("--tau_sweep", default="2.0,2.5,3.0,3.5")
    ap.add_argument("--sharp_sweep", default="0.5")
    ap.add_argument("--lam_sweep", default="0.1,0.2,0.3")
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    dcs_list, patients, binary_evs = load_ddxplus_full(args.n)
    # pool restricted to diseases present in this (possibly partial) KG
    present = [d for d in dcs_list if d in G and G.out_degree(d) > 0]
    base, all_evs = build_profile_v103(G, present, kappa=args.kappa,
                                       top_k=args.top_k, weight_key=args.weight_key)
    base = {d: p for d, p in base.items() if p}
    idf = compute_idf(base, 0.12)
    profile = reweight(base, idf, 1.0, args.beta)
    value_cuis = json.load(open(VALUE_CUIS))

    n_pool = len(profile)
    n_truth_in = sum(1 for tc, _, _ in patients if tc in profile)
    print(f"=== v103 KG × v71 algo — N={args.n} ===", flush=True)
    print(f"  pool diseases in KG: {n_pool}/{len(dcs_list)} | "
          f"patients truth-in-pool: {n_truth_in}/{len(patients)}", flush=True)
    if not idf:
        print("  EMPTY profile — no overlap"); return
    print(f"  IDF: min={min(idf.values()):.2f} max={max(idf.values()):.2f} "
          f"| evs={len(all_evs)}", flush=True)

    # restrict patients to those whose truth is in this pool (fair @1 on present)
    pats = [(tc, pos, neg) for tc, pos, neg in patients if tc in profile]
    for tau in [float(x) for x in args.tau_sweep.split(",")]:
        for sharp in [float(x) for x in args.sharp_sweep.split(",")]:
            sig = precompute_signal_v71(profile, value_cuis, binary_evs, idf, tau, sharp)
            for lam in [float(x) for x in args.lam_sweep.split(",")]:
                n = c1 = c3 = c5 = c10 = 0; rr = 0.0
                for tc, pos_raw, neg in pats:
                    pos = pos_raw & all_evs
                    if not pos: continue
                    s = score(pos, neg, profile, idf, args.beta, sig, lam)
                    ranked = sorted(profile.keys(), key=lambda d: -s[d])
                    n += 1
                    rank = ranked.index(tc) + 1 if tc in s else n_pool
                    if rank == 1: c1 += 1
                    if rank <= 3: c3 += 1
                    if rank <= 5: c5 += 1
                    if rank <= 10: c10 += 1
                    rr += 1.0 / rank
                if n:
                    print(f"  tau={tau:.1f} sharp={sharp:.1f} lam={lam:.2f}: "
                          f"@1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% "
                          f"@10={100*c10/n:.2f}% MRR={rr/n:.4f} (N={n})", flush=True)


if __name__ == "__main__":
    main()
