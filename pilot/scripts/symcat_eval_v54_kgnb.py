#!/usr/bin/env python3
"""SymCat KG-NB cross-benchmark eval.

Uses SAME v54 KG-NB architecture as DDXPlus, only swapping data loaders.
Demonstrates benchmark-agnostic transfer: single KG, single algorithm.

Disease set: SymCat 801 conditions → UMLS CUI via disease_umls_mapping.
Patient simulation: binomial sampling from SymCat symptom probabilities,
                    then map symptom name → CUI.
Profile: P(E|D) = w/(w+kappa) from KG edges (lay mode = PR universe only).
"""
from __future__ import annotations
import sys, json, math, pickle, argparse, random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))


PR_UNIVERSE = "pilot/data/pr_universe.json"


def load_symcat():
    """Returns (disease_profile, sym2cui).
    disease_profile: disease_name -> {symptom_name: prob (0-1)}
    sym2cui: symptom_name -> UMLS CUI
    """
    parsed = json.load(open("data/symcat/symcat_parsed.json"))
    sym_map = json.load(open("data/symcat/symptom_umls_mapping.json"))["mapping"]
    dis_map = json.load(open("data/symcat/disease_umls_mapping.json"))["mapping"]

    sym2cui = {name: info["umls_cui"] for name, info in sym_map.items()}
    dis2cui = {name: info["umls_cui"] for name, info in dis_map.items()}

    profile = {}  # disease_name -> {symptom_name: prob}
    for dname, sym_list in parsed["disease_symptom_pairs"].items():
        profile[dname] = {s[0]: s[1] / 100.0 for s in sym_list}

    return profile, sym2cui, dis2cui


def build_kg_profile(G, disease_cuis, allowed_cuis, kappa):
    """P(E|D) ∈ (0,1) for each disease D and evidence E."""
    profile = {}
    all_evs = set()
    for d in disease_cuis:
        if d not in G:
            profile[d] = {}
            continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE":
                continue
            if allowed_cuis is not None and p not in allowed_cuis:
                continue
            ed_w[p] += ed.get("weight", 0.0)
        prof_d = {p: w / (w + kappa) for p, w in ed_w.items() if w > 0}
        profile[d] = prof_d
        all_evs.update(prof_d.keys())
    return profile, all_evs


def nb_score(patient_cuis, profile, all_evs, log_prior, p_baseline, smooth=1e-3):
    scores = {}
    for d, prof_d in profile.items():
        log_p = log_prior
        for e in all_evs:
            p = prof_d.get(e, p_baseline)
            p = max(smooth, min(1 - smooth, p))
            if e in patient_cuis:
                log_p += math.log(p)
            else:
                log_p += math.log(1 - p)
        scores[d] = log_p
    return scores


def simulate_patient(disease_name, profile_sym, sym2cui):
    """Sample symptoms from disease distribution, convert to CUIs."""
    sym_prob = profile_sym[disease_name]
    patient_cuis = set()
    for sym_name, p in sym_prob.items():
        if random.random() < p:
            cui = sym2cui.get(sym_name)
            if cui:
                patient_cuis.add(cui)
    return patient_cuis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n_patients", type=int, default=100,
                    help="patients per disease to simulate")
    ap.add_argument("--n_diseases", type=int, default=0,
                    help="restrict to first N diseases (0 = all 50 in parsed file)")
    ap.add_argument("--evidence_categories", choices=["lay", "clinical", "comprehensive"],
                    default="lay")
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--p_baseline", type=float, default=0.01)
    ap.add_argument("--smooth", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    G = pickle.load(open(args.graph, "rb"))

    profile_sym, sym2cui, dis2cui = load_symcat()
    # Restrict to diseases that are both in parsed file AND have CUI mapping AND in KG
    candidate_diseases = []
    for dname in profile_sym:
        cui = dis2cui.get(dname)
        if cui and cui in G:
            candidate_diseases.append((dname, cui))
    if args.n_diseases > 0:
        candidate_diseases = candidate_diseases[: args.n_diseases]

    disease_cuis = [c for _, c in candidate_diseases]
    name_to_cui = {n: c for n, c in candidate_diseases}

    # Allowed CUI set per mode
    if args.evidence_categories == "lay":
        allowed = set(json.load(open(PR_UNIVERSE)))
    else:
        allowed = None

    # Build profile
    kg_profile, all_evs = build_kg_profile(G, disease_cuis, allowed, args.kappa)
    log_prior = math.log(1.0 / len(disease_cuis))

    print(f"=== v54 KG-NB on SymCat ===", flush=True)
    print(f"  mode={args.evidence_categories}, kappa={args.kappa}, "
          f"p_baseline={args.p_baseline}", flush=True)
    print(f"  diseases (mapped+in-KG): {len(disease_cuis)} / {len(profile_sym)} parsed",
          flush=True)
    sizes = [len(p) for p in kg_profile.values()]
    if sizes:
        print(f"  evidences per disease: avg={sum(sizes)/len(sizes):.0f}, "
              f"min={min(sizes)}, max={max(sizes)}, |all_evs|={len(all_evs)}",
              flush=True)

    n = 0; c1 = c3 = c5 = c10 = 0; rr_sum = 0.0
    for dname, true_cui in candidate_diseases:
        for _ in range(args.n_patients):
            patient_cuis = simulate_patient(dname, profile_sym, sym2cui) & all_evs
            if not patient_cuis:
                continue
            scores = nb_score(patient_cuis, kg_profile, all_evs, log_prior,
                              args.p_baseline, args.smooth)
            ranked = sorted(scores.keys(), key=lambda d: -scores[d])
            n += 1
            try:
                rank = ranked.index(true_cui) + 1
            except ValueError:
                rank = len(disease_cuis)
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0 / rank

    print(f"v54 KG-NB SymCat mode={args.evidence_categories} kappa={args.kappa}: "
          f"@1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% "
          f"@10={100*c10/n:.2f}% MRR={rr_sum/n:.4f} N={n}", flush=True)


if __name__ == "__main__":
    main()
