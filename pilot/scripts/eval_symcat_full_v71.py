#!/usr/bin/env python3
"""SymCat 801 disease full evaluation with v71 algorithm.

Reuses v71 config: cosine + IDF (df_thr=0.12, alpha=1.0, beta=0.75) +
self-aware negative penalty. NO LLM at inference.

SymCat doesn't have binary evidence default values (DDXPlus-specific),
so no_binary penalty term is skipped. Only positive cosine + IDF.

Patient generation: per disease, Bernoulli sampling from symptom_prob.
"""
from __future__ import annotations
import sys, json, math, pickle, argparse, random
from pathlib import Path
from collections import defaultdict

PR_UNIVERSE = "pilot/data/pr_universe.json"


def build_profile(G, dcs, kappa, pr):
    """Profile in lay mode (patient_reportable + history + demographic)."""
    profile = {}; all_evs = set()
    allowed = {"patient_reportable", "history", "demographic"}
    for d in dcs:
        if d not in G: profile[d] = {}; continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            cat = ed.get("category")
            if cat is None:
                if p not in pr: continue
            elif cat not in allowed: continue
            ed_w[p] += ed.get("weight", 0.0)
        prof = {p: w/(w+kappa) for p, w in ed_w.items() if w > 0}
        profile[d] = prof
        all_evs.update(prof.keys())
    return profile, all_evs


def compute_idf(profile, df_threshold):
    N = len(profile)
    df = defaultdict(int)
    for prof in profile.values():
        for e, p in prof.items():
            if p >= df_threshold: df[e] += 1
    return {e: math.log((N+1)/(df_e+1))+1.0 for e, df_e in df.items()}


def reweight(profile, idf, alpha, beta):
    return {d: {e: (p**alpha)*(idf.get(e,1.0)**beta) for e,p in prof.items()}
            for d, prof in profile.items()}


def score_cosine(pcuis, profile, idf, beta):
    scores = {}
    pat_vec = {e: idf.get(e, 1.0) ** beta for e in pcuis}
    p_norm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
    for d, prof in profile.items():
        if not prof: scores[d] = -1e9; continue
        dot = sum(pat_vec[e] * prof[e] for e in pcuis if e in prof)
        d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
        scores[d] = dot / (p_norm * d_norm)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--parsed", default="data/symcat/symcat_parsed_full.json")
    ap.add_argument("--sym_map", default="data/symcat/symptom_umls_mapping.json")
    ap.add_argument("--dis_map", default="data/symcat/disease_umls_mapping.json")
    ap.add_argument("--n_patients_per_d", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beta", type=float, default=0.75)
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))
    parsed = json.load(open(args.parsed))
    sym_map = json.load(open(args.sym_map))["mapping"]
    dis_map = json.load(open(args.dis_map))["mapping"]

    pairs = parsed["disease_symptom_pairs"]
    sym2cui = {n: v["umls_cui"] for n, v in sym_map.items() if v.get("umls_cui")}
    dis2cui = {n: v["umls_cui"] for n, v in dis_map.items() if v.get("umls_cui")}

    # FAIR EVAL: include ALL mapped diseases as candidates, regardless of KG presence
    cand = []
    for dname in pairs:
        cui = dis2cui.get(dname)
        if not cui: continue
        cand.append((dname, cui))
    dcs_list = sorted({c for _, c in cand})
    in_kg = sum(1 for c in dcs_list if c in G)
    print(f"SymCat: {len(pairs)} parsed, all mapped: {len(cand)}", flush=True)
    print(f"Unique disease CUIs: {len(dcs_list)} (in KG: {in_kg}, missing: {len(dcs_list)-in_kg})", flush=True)

    # Build profile + IDF + reweight
    base, all_evs = build_profile(G, dcs_list, 20.0, pr)
    idf = compute_idf(base, 0.12)
    profile = reweight(base, idf, 1.0, args.beta)
    n_with_profile = sum(1 for d in dcs_list if profile[d])
    print(f"Disease with non-empty profile: {n_with_profile}/{len(dcs_list)}", flush=True)
    print(f"|all_evs|={len(all_evs)}", flush=True)

    # Generate patients
    random.seed(args.seed)
    patients = []
    for dname, true_cui in cand:
        sym_prob = {sym2cui.get(s): p/100.0
                    for s, p in pairs[dname] if sym2cui.get(s)}
        sym_prob = {c: p for c, p in sym_prob.items() if c is not None}
        for _ in range(args.n_patients_per_d):
            pcuis = {c for c, p in sym_prob.items() if random.random() < p}
            if pcuis:
                patients.append((true_cui, pcuis))
    print(f"Generated {len(patients)} patients", flush=True)

    # FAIR Evaluate: ALL patients (no skip), full disease pool
    n = c1 = c3 = c5 = c10 = c20 = 0; rr = 0.0
    n_empty = 0  # patients with no KG-matchable evidence
    print_every = 5000
    for true_cui, raw in patients:
        pcuis = raw & all_evs
        if not pcuis:
            # No evidence matches KG vocabulary → cannot rank → always fail (rank=last)
            n_empty += 1
            rank = len(dcs_list)
        else:
            scores = score_cosine(pcuis, profile, idf, args.beta)
            # rank against FULL dcs_list (incl. diseases not in KG → score 0 → low rank)
            ranked = sorted(dcs_list, key=lambda d: -scores.get(d, -1e9))
            try: rank = ranked.index(true_cui)+1
            except ValueError: rank = len(dcs_list)
        n += 1
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        if rank <= 20: c20 += 1
        rr += 1.0/rank
        if n % print_every == 0:
            print(f"  [{n}] @1={100*c1/n:.2f}% empty={n_empty}", flush=True)

    print(f"\n=== SymCat 801 FAIR full eval (v71) ===")
    print(f"  N={n} patients over {len(dcs_list)} diseases (KG: {in_kg}, missing: {len(dcs_list)-in_kg})")
    print(f"  Empty-KG-match patients (forced fail): {n_empty} ({100*n_empty/n:.1f}%)")
    print(f"  Random baseline @1: {100/len(dcs_list):.3f}%")
    print(f"  GTPA: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% "
          f"@10={100*c10/n:.2f}% @20={100*c20/n:.2f}%")
    print(f"  MRR: {rr/n:.4f}")


if __name__ == "__main__":
    main()
