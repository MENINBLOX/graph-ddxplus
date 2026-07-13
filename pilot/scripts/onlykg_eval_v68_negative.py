#!/usr/bin/env python3
"""v68 — Integrate negative evidence (default=0 binary answers).

CRITICAL FINDING: DDXPlus has 208 binary evidences, ALL with default=0.
The CSV's EVIDENCES column only lists non-default values. Therefore,
any binary evidence NOT in the patient's EVIDENCES list = explicit 'NO'.

Patient profile decomposition:
  pos_pcuis = mapped CUIs from EVIDENCES (yes answers + value choices)
  neg_pcuis = mapped CUIs from BINARY evidences NOT in EVIDENCES
              (i.e., the patient said NO to that symptom)

Scoring:
  pos_score = cosine(pos_pat_vec, profile_D)
  neg_score = sum over E in neg_pcuis ∩ profile_D of profile_D[E]
              (high if disease D expects E but patient denied E)
              normalized by sqrt(|profile_D|) so larger profiles aren't punished

  final(D | patient) = pos_score - lambda * (neg_score / norm)

Single algorithm (cosine + IDF + top-K + negative penalty).
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
EV_META = "data/ddxplus/release_evidences.json"

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
            if p >= df_threshold: df[e] += 1
    return {e: math.log((N+1)/(df_e+1))+1.0 for e, df_e in df.items()}


def reweight(profile, idf, alpha, beta):
    return {d: {e: (p**alpha)*(idf.get(e,1.0)**beta) for e,p in prof.items()}
            for d, prof in profile.items()}


def load_ddxplus_with_negative(n_max):
    """Returns patients as (true_cui, pos_pcuis, neg_pcuis)."""
    value_cuis = json.load(open(VALUE_CUIS))
    ev_meta = json.load(open(EV_META))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    dcs_list = sorted(set(fr2cui.values()))

    # All binary evidence IDs with default=0
    binary_evs = {ev_id for ev_id, m in ev_meta.items()
                  if m.get("data_type") == "B" and m.get("default_value") == 0}

    patients = []; n = 0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= n_max: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pos_pcuis = set()
            answered_bases = set()
            for ev in evs:
                if "_@_" in ev:
                    base, val = ev.split("_@_", 1)
                    answered_bases.add(base)
                    m = value_cuis.get(base, {})
                    for k in ("_question", val):
                        v = m.get(k, [])
                        if isinstance(v, list): pos_pcuis.update(v)
                else:
                    answered_bases.add(ev)
                    m = value_cuis.get(ev, {})
                    pos_pcuis.update(m.get("_question", []))

            # Negative: binary evidences NOT answered → default 'no'
            unanswered = binary_evs - answered_bases
            neg_pcuis = set()
            for ev_id in unanswered:
                m = value_cuis.get(ev_id, {})
                neg_pcuis.update(m.get("_question", []))
            # Remove overlap with pos (sometimes _question CUIs are shared)
            neg_pcuis -= pos_pcuis

            patients.append((true_cui, pos_pcuis, neg_pcuis)); n += 1
    return dcs_list, patients


def score_v68(pos_pcuis, neg_pcuis, profile, idf, beta, lam, normalize_neg='sqrt'):
    scores = {}
    pat_vec = {e: idf.get(e, 1.0) ** beta for e in pos_pcuis}
    p_norm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
    for d, prof in profile.items():
        if not prof: scores[d] = -1e9; continue
        dot = sum(pat_vec[e] * prof[e] for e in pos_pcuis if e in prof)
        d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
        pos_score = dot / (p_norm * d_norm)
        # Negative penalty
        neg_dot = sum((idf.get(e,1.0)**beta) * prof[e] for e in neg_pcuis if e in prof)
        if normalize_neg == 'sqrt':
            neg_norm = math.sqrt(len(neg_pcuis)) or 1e-9
            neg_score = neg_dot / (neg_norm * d_norm)
        elif normalize_neg == 'd_norm':
            neg_score = neg_dot / d_norm
        else:
            neg_score = neg_dot
        scores[d] = pos_score - lam * neg_score
    return scores


def evaluate(profile, idf, beta, patients, all_evs, dcs_list, lam, normalize_neg='sqrt'):
    n = c1 = c3 = c5 = c10 = 0; rr = 0.0
    for true_cui, pos_raw, neg_raw in patients:
        pos = pos_raw & all_evs
        neg = neg_raw & all_evs
        if not pos: continue
        scores = score_v68(pos, neg, profile, idf, beta, lam, normalize_neg)
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
    ap.add_argument("--lam_sweep", type=str, default="0.0,0.05,0.1,0.2,0.3,0.5,1.0,2.0")
    ap.add_argument("--normalize_neg", default="sqrt", choices=["sqrt", "d_norm", "none"])
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))

    dcs_list, patients = load_ddxplus_with_negative(args.n)
    base, all_evs = build_profile(G, dcs_list, args.mode, args.kappa, pr, top_k=args.top_k)
    idf = compute_idf(base, args.df_threshold)
    profile = reweight(base, idf, args.alpha, args.beta)

    avg_pos = sum(len(p[1]) for p in patients)/len(patients)
    avg_neg = sum(len(p[2]) for p in patients)/len(patients)
    print(f"=== v68 negative evidence — N={args.n} ===")
    print(f"  avg pos CUIs: {avg_pos:.1f}, avg neg CUIs: {avg_neg:.1f}, norm={args.normalize_neg}")

    for lam in args.lam_sweep.split(","):
        lam = float(lam)
        r = evaluate(profile, idf, args.beta, patients, all_evs, dcs_list, lam, args.normalize_neg)
        print(f"  lambda={lam:.2f}: @1={r['at1']:.2f}% @3={r['at3']:.2f}% "
              f"@5={r['at5']:.2f}% @10={r['at10']:.2f}% MRR={r['mrr']:.4f}", flush=True)


if __name__ == "__main__":
    main()
