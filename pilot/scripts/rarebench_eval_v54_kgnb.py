#!/usr/bin/env python3
"""RareBench KG-NB cross-benchmark eval.

Same v54 KG-NB architecture as DDXPlus / SymCat. Demonstrates 3-benchmark
transfer with single KG + single algorithm.

RareBench input:
  {"Phenotype": ["HP:xxx", ...], "RareDisease": ["ORPHA:n", "OMIM:n", ...]}

Mapping:
  HPO -> CUI via hpo_umls_mapping.json
  ORPHA/OMIM -> CUI via disease_umls_mapping.json

Mode: 'clinical' by default (HPO is clinical phenotype vocab, not lay).
"""
from __future__ import annotations
import sys, json, math, pickle, argparse
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))

PR_UNIVERSE = "pilot/data/pr_universe.json"


def load_rarebench(jsonl_path):
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_kg_profile(G, disease_cuis, allowed_cuis, kappa):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--dataset", default="RAMEDIS",
                    choices=["RAMEDIS", "HMS", "MME", "LIRICAL"])
    ap.add_argument("--evidence_categories", choices=["lay", "clinical", "comprehensive"],
                    default="clinical")
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--p_baseline", type=float, default=0.01)
    ap.add_argument("--smooth", type=float, default=1e-3)
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    hpo2cui_raw = json.load(open("data/rarebench/hpo_umls_mapping.json"))["mapping"]
    dis2cui_raw = json.load(open("data/rarebench/disease_umls_mapping.json"))["mapping"]
    hpo2cui = {k: v["umls_cui"] for k, v in hpo2cui_raw.items()}
    dis2cui = {k: v["umls_cui"] for k, v in dis2cui_raw.items()}

    test_rows = load_rarebench(f"data/rarebench/data/{args.dataset}.jsonl")

    # Build candidate disease pool: union of all true diseases in test set
    # that have CUI mappings AND are in KG
    candidate_set = set()
    n_no_map = 0
    for row in test_rows:
        any_mapped = False
        for did in row.get("RareDisease", []):
            cui = dis2cui.get(did)
            if cui and cui in G:
                candidate_set.add(cui)
                any_mapped = True
        if not any_mapped:
            n_no_map += 1

    disease_cuis = sorted(candidate_set)

    if args.evidence_categories == "lay":
        allowed = set(json.load(open(PR_UNIVERSE)))
    else:
        allowed = None

    kg_profile, all_evs = build_kg_profile(G, disease_cuis, allowed, args.kappa)
    log_prior = math.log(1.0 / len(disease_cuis))

    print(f"=== v54 KG-NB on RareBench/{args.dataset} ===", flush=True)
    print(f"  mode={args.evidence_categories}, kappa={args.kappa}, "
          f"p_baseline={args.p_baseline}", flush=True)
    print(f"  test rows: {len(test_rows)}, no-mapping rows: {n_no_map}", flush=True)
    print(f"  candidate diseases (in KG): {len(disease_cuis)}", flush=True)
    sizes = [len(p) for p in kg_profile.values() if p]
    if sizes:
        print(f"  evidences per disease: avg={sum(sizes)/len(sizes):.0f}, "
              f"min={min(sizes)}, max={max(sizes)}, |all_evs|={len(all_evs)}",
              flush=True)

    n = 0; c1 = c3 = c5 = c10 = 0; rr_sum = 0.0
    for row in test_rows:
        # Patient: phenotypes -> CUIs
        patient_cuis = set()
        for hp in row.get("Phenotype", []):
            cui = hpo2cui.get(hp)
            if cui:
                patient_cuis.add(cui)
        patient_cuis &= all_evs
        if not patient_cuis:
            continue

        # True: any RareDisease that maps to a candidate
        true_cuis = set()
        for did in row.get("RareDisease", []):
            cui = dis2cui.get(did)
            if cui and cui in candidate_set:
                true_cuis.add(cui)
        if not true_cuis:
            continue

        scores = nb_score(patient_cuis, kg_profile, all_evs, log_prior,
                          args.p_baseline, args.smooth)
        ranked = sorted(scores.keys(), key=lambda d: -scores[d])
        n += 1
        # rank = min rank of any true_cui
        ranks = [ranked.index(t) + 1 for t in true_cuis if t in ranked]
        rank = min(ranks) if ranks else len(disease_cuis)
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        rr_sum += 1.0 / rank

    print(f"v54 KG-NB RareBench/{args.dataset} mode={args.evidence_categories} "
          f"kappa={args.kappa}: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% "
          f"@5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f} N={n}",
          flush=True)


if __name__ == "__main__":
    main()
