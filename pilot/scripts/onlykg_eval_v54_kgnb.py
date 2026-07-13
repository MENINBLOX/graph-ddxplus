#!/usr/bin/env python3
"""v54 KG-NB: benchmark-agnostic Naive Bayes from KG edges.

Key difference vs v53:
- v53 NB: P(E|D) from DDXPlus train labels (supervised, benchmark-specific)
- v54 KG-NB: P(E|D) from KG edge weights (zero-shot, benchmark-agnostic)

Eval-time category filter via --evidence_categories:
  lay         = patient_reportable only (DDXPlus, SymCat)
  clinical    = all CUIs (RareBench HPO, ER-Reason w/ labs)
  comprehensive = lay + extended (default for unknown benchmarks)

Hill-function mapping w(D,E) -> P(E|D):
  P(E|D) = w / (w + kappa)
  Single tunable param kappa controls saturation point.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"


def build_kg_profile(G, dcs_list, allowed_cuis, kappa):
    """P(E|D) ∈ (0,1) for each disease D and evidence E.

    Uses Hill function: P = w / (w + kappa). Edges restricted to allowed_cuis.
    Returns dict[D -> dict[CUI -> P(E|D)]] and full set of evidence CUIs across all D.
    """
    profile = {}
    all_evs = set()
    for d in dcs_list:
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
    """log P(D|patient) for each disease D under independent-evidence NB.

    For each E in all_evs:
      if E in patient: log P(E_present|D)
      else:            log P(E_absent|D) = log(1 - P(E_present|D))

    p_baseline: prob for evidences not in disease's profile (low, e.g. 0.02)
    """
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


def parse_features_ddxplus(evs, value_cuis):
    pcuis = set()
    for ev in evs:
        if "_@_" in ev:
            base, val = ev.split("_@_", 1)
            m = value_cuis.get(base, {})
            for k in ("_question", val):
                v = m.get(k, [])
                if isinstance(v, list):
                    pcuis.update(v)
        else:
            m = value_cuis.get(ev, {})
            pcuis.update(m.get("_question", []))
    return pcuis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=200000, help="max patients to eval")
    ap.add_argument("--evidence_categories", choices=["lay", "clinical", "comprehensive"],
                    default="lay", help="lay=PR only, clinical=all, comprehensive=lay+ext")
    ap.add_argument("--kappa", type=float, default=5.0,
                    help="Hill function saturation: P = w/(w+kappa)")
    ap.add_argument("--p_baseline", type=float, default=0.02,
                    help="Background prob for evidence not in disease profile")
    ap.add_argument("--smooth", type=float, default=1e-3,
                    help="Clamp range to avoid log(0)")
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f:
        icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f:
        cond = json.load(f)
    fr2cui = {info.get("cond-name-fr", ""): icd[dn]["cui"]
              for dn, info in cond.items() if dn in icd}
    dcs_list = sorted(set(fr2cui.values()))

    # Allowed CUI set per mode
    if args.evidence_categories == "lay":
        allowed = set(json.load(open(PR_UNIVERSE)))
    elif args.evidence_categories == "clinical":
        allowed = None  # all CUIs
    else:  # comprehensive
        allowed = None

    # Build profile
    profile, all_evs = build_kg_profile(G, dcs_list, allowed, args.kappa)
    log_prior = math.log(1.0 / len(dcs_list))

    print(f"=== v54 KG-NB ===", flush=True)
    print(f"  mode={args.evidence_categories}, kappa={args.kappa}, p_baseline={args.p_baseline}", flush=True)
    print(f"  diseases={len(dcs_list)}, total evidence CUIs in profile={len(all_evs)}", flush=True)
    profile_sizes = [len(p) for p in profile.values()]
    print(f"  evidences per disease: avg={sum(profile_sizes)/len(profile_sizes):.0f}, "
          f"min={min(profile_sizes)}, max={max(profile_sizes)}", flush=True)

    n = 0; c1 = c3 = c5 = c10 = 0; rr_sum = 0.0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n:
                break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list:
                continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis = parse_features_ddxplus(evs, value_cuis)
            # Filter patient CUIs to evidence universe
            patient_cuis = pcuis & all_evs

            scores = nb_score(patient_cuis, profile, all_evs, log_prior,
                              args.p_baseline, args.smooth)
            ranked = sorted(scores.keys(), key=lambda d: -scores[d])
            n += 1
            try:
                rank = ranked.index(true_cui) + 1
            except ValueError:
                rank = len(dcs_list)
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0 / rank

            if n % 10000 == 0:
                print(f"  [{n} patients] @1={100*c1/n:.2f}%", flush=True)

    print(f"v54 KG-NB mode={args.evidence_categories} kappa={args.kappa}: "
          f"@1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% "
          f"@10={100*c10/n:.2f}% MRR={rr_sum/n:.4f} N={n}", flush=True)


if __name__ == "__main__":
    main()
