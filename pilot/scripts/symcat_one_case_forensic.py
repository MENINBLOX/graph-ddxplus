#!/usr/bin/env python3
"""SymCat 한 실패 케이스 깊이 분석.

한 patient를 추적:
1. 어떤 disease (true)?
2. Patient가 보고한 SymCat symptoms (raw + CUI)?
3. KG profile에서 true disease가 가진 CUI (top by weight)?
4. Patient CUI ∩ true_profile (얼마나 매칭됐나)?
5. 모델이 top-1로 예측한 disease는 무엇? 왜 그게 선택됐나?
6. True disease의 score vs 예측 disease의 score?
"""
from __future__ import annotations
import json, math, pickle, argparse, random, sys
from pathlib import Path
from collections import defaultdict

PR_UNIVERSE = "pilot/data/pr_universe.json"


def build_profile(G, dcs, kappa, pr):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="pilot/data/onlykg_graph_v93_s3.pkl")
    ap.add_argument("--target_disease", default="Acute bronchitis")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))
    parsed = json.load(open("data/symcat/symcat_parsed_full.json"))
    sym_map = json.load(open("data/symcat/symptom_umls_mapping.json"))["mapping"]
    dis_map = json.load(open("data/symcat/disease_umls_mapping.json"))["mapping"]
    sym2cui = {n: v["umls_cui"] for n, v in sym_map.items() if v.get("umls_cui")}
    cui2sym = {v: n for n, v in sym2cui.items()}
    dis2cui = {n: v["umls_cui"] for n, v in dis_map.items() if v.get("umls_cui")}
    cui2dis = {v: n for n, v in dis2cui.items()}

    cand = [(dn, dis2cui[dn]) for dn in parsed["disease_symptom_pairs"] if dis2cui.get(dn)]
    dcs_list = sorted({c for _, c in cand})
    base, all_evs = build_profile(G, dcs_list, 20.0, pr)
    idf = compute_idf(base, 0.12)
    profile = reweight(base, idf, 1.0, 0.75)

    if args.target_disease not in dis2cui:
        print(f"Disease '{args.target_disease}' not in SymCat mapping")
        return
    true_cui = dis2cui[args.target_disease]
    sym_prob = {sym2cui[s]: p/100.0
                for s, p in parsed["disease_symptom_pairs"][args.target_disease]
                if sym2cui.get(s)}

    print(f"=== Forensic: '{args.target_disease}' (CUI={true_cui}) ===\n")
    print(f"--- SymCat truth: symptoms & weights ---")
    for s, p in sorted(parsed["disease_symptom_pairs"][args.target_disease], key=lambda x: -x[1]):
        cui = sym2cui.get(s, '?')
        in_profile = "  IN" if cui in profile.get(true_cui, {}) else "OUT"
        print(f"  [{in_profile}] {s} (CUI={cui}, weight={p}%)")

    # Generate a Bernoulli patient (sample from sym_prob)
    random.seed(args.seed)
    pcuis = {c for c, p in sym_prob.items() if random.random() < p}
    print(f"\n--- Sampled patient: {len(pcuis)} symptoms (seed={args.seed}) ---")
    for c in pcuis:
        print(f"  - {cui2sym.get(c, c)} ({c})")
    pcuis_kg = pcuis & all_evs
    print(f"  Of these in KG vocab: {len(pcuis_kg)}/{len(pcuis)}")

    if not pcuis_kg:
        print("\nEMPTY patient — cannot rank")
        return

    # Score all 761 candidates
    def score(pcuis, prof):
        pat_vec = {e: idf.get(e, 1.0)**0.75 for e in pcuis}
        p_norm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
        if not prof: return -1e9
        dot = sum(pat_vec[e] * prof[e] for e in pcuis if e in prof)
        d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
        return dot / (p_norm * d_norm)

    scores = {d: score(pcuis_kg, profile[d]) for d in dcs_list}
    ranked = sorted(dcs_list, key=lambda d: -scores[d])

    true_rank = ranked.index(true_cui) + 1
    print(f"\n--- True disease rank: {true_rank} / {len(dcs_list)} ---")
    print(f"True disease score: {scores[true_cui]:.4f}")

    print(f"\n--- TOP 10 ranked candidates ---")
    for i, dcui in enumerate(ranked[:10]):
        mark = " *** TRUE" if dcui == true_cui else ""
        dn = cui2dis.get(dcui, dcui)
        print(f"  [{i+1}] {dn} (CUI={dcui}): score={scores[dcui]:.4f}{mark}")

    # Why? Show which CUIs the top-1 has that matched
    pred_cui = ranked[0]
    pred_name = cui2dis.get(pred_cui, pred_cui)
    print(f"\n--- Why #1 = '{pred_name}' was picked ---")
    pred_prof = profile.get(pred_cui, {})
    matched = sorted([(c, pred_prof.get(c, 0)*idf.get(c,1.0)**0.75)
                      for c in pcuis_kg if c in pred_prof], key=lambda x:-x[1])
    print(f"  Patient CUI ∩ pred profile: {len(matched)} matches")
    for cui, contrib in matched[:10]:
        print(f"    {cui2sym.get(cui, cui)} ({cui}): contrib={contrib:.4f}, "
              f"idf={idf.get(cui,1):.2f}")

    print(f"\n--- True disease matches ---")
    true_prof = profile.get(true_cui, {})
    matched_true = sorted([(c, true_prof.get(c, 0)*idf.get(c,1.0)**0.75)
                           for c in pcuis_kg if c in true_prof], key=lambda x:-x[1])
    print(f"  Patient CUI ∩ true profile: {len(matched_true)} matches")
    for cui, contrib in matched_true[:10]:
        print(f"    {cui2sym.get(cui, cui)} ({cui}): contrib={contrib:.4f}, "
              f"idf={idf.get(cui,1):.2f}")

    print(f"\n--- True disease profile top-20 CUIs (KG-derived) ---")
    top_true = sorted(true_prof.items(), key=lambda x: -x[1])[:20]
    for cui, w in top_true:
        sym_name = cui2sym.get(cui, '(not in SymCat vocab)')
        mark = " *<<< patient has" if cui in pcuis_kg else ""
        print(f"  {sym_name} ({cui}): prof_w={w:.4f}, idf={idf.get(cui,1):.2f}{mark}")


if __name__ == "__main__":
    main()
