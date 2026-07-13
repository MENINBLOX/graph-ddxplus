#!/usr/bin/env python3
"""v65 — Per-disease PMI weighting on top of top-K + IDF.

Forensic v64 finding: Bronchitis loses to Bronchiolitis even after top-K
+IDF because shared CUIs (Fever, Dyspnea, Wheezing, Virus Diseases) all
have higher weight in Bronchiolitis profile. The CUI is in BOTH profiles,
so IDF doesn't penalize it sufficiently.

Pointwise Mutual Information measures how much MORE likely E is in
disease D compared to baseline:

    P(E) = (1/N) * Σ_D P(E|D)
    PMI(E, D) = log( P(E|D) / (P(E) + ε) )

When PMI(Fever, Bronchiolitis) > PMI(Fever, Bronchitis), Bronchiolitis is
correctly favored when patient has Fever — but only by the actual
disease-specific signal, not by raw weight magnitude.

Final weight:
    w(E, D) = P(E|D)^alpha * IDF(E)^beta * max(0, PMI(E,D))^gamma
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse, random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"

MODE_CATEGORIES = {
    "lay": {"patient_reportable", "history", "demographic"},
    "clinical": {"clinical_sign", "lab_finding", "imaging_finding", "history", "demographic"},
}


def build_profile(G, dcs, mode, kappa, pr, top_k=None):
    allowed = MODE_CATEGORIES.get(mode)
    profile = {}; all_evs = set()
    for d in dcs:
        if d not in G: profile[d] = {}; continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            cat = ed.get("category")
            if cat is None:
                if mode == "lay" and p not in pr: continue
            else:
                if allowed is not None and cat not in allowed: continue
            ed_w[p] += ed.get("weight", 0.0)
        prof = {p: w/(w+kappa) for p, w in ed_w.items() if w > 0}
        if top_k and len(prof) > top_k:
            prof = dict(sorted(prof.items(), key=lambda x: -x[1])[:top_k])
        profile[d] = prof
        all_evs.update(prof.keys())
    return profile, all_evs


def compute_idf(profile, df_threshold):
    N = len(profile)
    df = defaultdict(int)
    for prof in profile.values():
        for e, p in prof.items():
            if p >= df_threshold:
                df[e] += 1
    return {e: math.log((N+1)/(df_e+1))+1.0 for e, df_e in df.items()}


def compute_pmi(profile, eps=0.005):
    """PMI(E, D) = log( (P(E|D)+eps) / (mean_D' P(E|D')+eps) )"""
    N = len(profile)
    # mean P(E|D) over disease universe
    total = defaultdict(float)
    for prof in profile.values():
        for e, p in prof.items():
            total[e] += p
    mean_p = {e: total[e]/N for e in total}
    pmi = {}
    for d, prof in profile.items():
        pmi[d] = {}
        for e, p in prof.items():
            pmi[d][e] = math.log((p + eps) / (mean_p[e] + eps))
    return pmi, mean_p


def reweight(profile, idf, pmi, alpha, beta, gamma):
    new = {}
    for d, prof in profile.items():
        new[d] = {}
        for e, p in prof.items():
            base = (p ** alpha) * (idf.get(e, 1.0) ** beta)
            pmi_val = max(0.0, pmi.get(d, {}).get(e, 0.0))
            new[d][e] = base * ((1.0 + pmi_val) ** gamma)
    return new


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


def load_ddxplus(n_max):
    value_cuis = json.load(open(VALUE_CUIS))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    patients = []; n = 0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= n_max: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis = set()
            for ev in evs:
                if "_@_" in ev:
                    base, val = ev.split("_@_", 1)
                    m = value_cuis.get(base, {})
                    for k in ("_question", val):
                        v = m.get(k, [])
                        if isinstance(v, list): pcuis.update(v)
                else:
                    m = value_cuis.get(ev, {})
                    pcuis.update(m.get("_question", []))
            patients.append((true_cui, pcuis)); n += 1
    return dcs_list, patients


def evaluate(profile, idf, beta, patients, all_evs, dcs_list):
    n = c1 = c3 = c5 = c10 = 0; rr = 0.0
    for true_cui, raw in patients:
        pcuis = set(raw) & all_evs
        if not pcuis: continue
        scores = score_cosine(pcuis, profile, idf, beta)
        ranked = sorted(profile.keys(), key=lambda d: -scores[d])
        n += 1
        try: rank = ranked.index(true_cui)+1
        except: rank = len(dcs_list)
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        rr += 1.0/rank
    return {"n": n, "at1": 100*c1/n, "at3": 100*c3/n, "at5": 100*c5/n,
            "at10": 100*c10/n, "mrr": rr/n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--mode", default="lay")
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--df_threshold", type=float, default=0.12)
    ap.add_argument("--top_k", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--gamma_sweep", type=str, default="0.0,0.25,0.5,0.75,1.0,1.5,2.0")
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))
    dcs_list, patients = load_ddxplus(args.n)

    base, all_evs = build_profile(G, dcs_list, args.mode, args.kappa, pr, top_k=args.top_k)
    idf = compute_idf(base, args.df_threshold)
    pmi, mean_p = compute_pmi(base)

    print(f"=== v65 PMI sweep — N={args.n} top_k={args.top_k} a={args.alpha} b={args.beta} df_t={args.df_threshold} ===")
    print(f"  diseases={len(dcs_list)}, all_evs={len(all_evs)}")
    # PMI stats
    pmi_vals = [v for d in pmi.values() for v in d.values()]
    if pmi_vals:
        pmi_pos = [v for v in pmi_vals if v > 0]
        print(f"  PMI range: min={min(pmi_vals):.2f} max={max(pmi_vals):.2f} "
              f"%positive={100*len(pmi_pos)/len(pmi_vals):.1f}%")

    for g in args.gamma_sweep.split(","):
        g = float(g)
        prof = reweight(base, idf, pmi, args.alpha, args.beta, g)
        r = evaluate(prof, idf, args.beta, patients, all_evs, dcs_list)
        print(f"  gamma={g:.2f}: @1={r['at1']:.2f}% @3={r['at3']:.2f}% "
              f"@5={r['at5']:.2f}% @10={r['at10']:.2f}% MRR={r['mrr']:.4f}", flush=True)


if __name__ == "__main__":
    main()
