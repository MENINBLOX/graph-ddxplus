#!/usr/bin/env python3
"""v67 — Add AGE-based CUI to patient evidence.

Forensic v64 finding: Bronchitis vs Bronchiolitis confusion driven by
patient AGE (Bronchitis adult vs Bronchiolitis <2yr). KG has demographic
CUIs in 45/49 diseases. Map patient AGE to standard demographic CUIs:

  AGE < 2     → C0027361 Infant
  AGE 2-17    → C0008059 Child
  AGE 18-64   → C0241889 Adult
  AGE 65+     → C0001779 Aged

Single algorithm preserved (cosine + IDF + top-K), only the patient CUI
set is augmented with the age token.
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


def age_to_cuis(age):
    cuis = set()
    if age < 2: cuis.add("C0027361")  # Infant
    if age < 18: cuis.add("C0008059")  # Child
    if 18 <= age <= 64: cuis.add("C0241889")  # Adult
    if age >= 65: cuis.add("C0001779")  # Aged
    return cuis


def sex_to_cui(sex):
    if sex == "F": return "C0086287"
    if sex == "M": return "C0086582"
    return None


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


def load_ddxplus_with_age(n_max, use_age=True, use_sex=False, age_boost=1.0):
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
            age_cuis = set(); sex_cui = None
            if use_age:
                age_cuis = age_to_cuis(int(row["AGE"]))
                pcuis |= age_cuis
            if use_sex:
                sex_cui = sex_to_cui(row["SEX"])
                if sex_cui: pcuis.add(sex_cui)
            patients.append((true_cui, pcuis, age_cuis, sex_cui)); n += 1
    return dcs_list, patients


def evaluate(profile, idf, beta, patients, all_evs, dcs_list, age_boost=1.0):
    n = c1 = c3 = c5 = c10 = 0; rr = 0.0
    age_set = {"C0027361", "C0008059", "C0241889", "C0001779"}
    for true_cui, raw, age_cuis, sex_cui in patients:
        pcuis = set(raw) & all_evs
        if not pcuis: continue
        # Optionally amplify age CUI weight in patient vector
        scores = {}
        pat_vec = {e: idf.get(e, 1.0) ** beta for e in pcuis}
        if age_boost != 1.0:
            for e in pcuis & age_set:
                pat_vec[e] *= age_boost
        p_norm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
        for d, prof in profile.items():
            if not prof: scores[d] = -1e9; continue
            dot = sum(pat_vec[e] * prof[e] for e in pcuis if e in prof)
            d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
            scores[d] = dot / (p_norm * d_norm)
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
    ap.add_argument("--variants", type=str, default="baseline,age,age_sex,age_boost2,age_boost4")
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))

    base, all_evs = build_profile(G, [], args.mode, args.kappa, pr, top_k=args.top_k)
    # Reload with proper dcs
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    base, all_evs = build_profile(G, dcs_list, args.mode, args.kappa, pr, top_k=args.top_k)
    idf = compute_idf(base, args.df_threshold)
    profile = reweight(base, idf, args.alpha, args.beta)

    print(f"=== v67 AGE channel — N={args.n} top_k={args.top_k} ===")

    for variant in args.variants.split(","):
        use_age = variant != "baseline"
        use_sex = "sex" in variant
        age_boost = 1.0
        if "boost2" in variant: age_boost = 2.0
        elif "boost4" in variant: age_boost = 4.0
        elif "boost8" in variant: age_boost = 8.0

        dcs_list, patients = load_ddxplus_with_age(args.n, use_age, use_sex)
        r = evaluate(profile, idf, args.beta, patients, all_evs, dcs_list, age_boost)
        print(f"  {variant:<15}: @1={r['at1']:.2f}% @3={r['at3']:.2f}% "
              f"@5={r['at5']:.2f}% @10={r['at10']:.2f}% MRR={r['mrr']:.4f}", flush=True)


if __name__ == "__main__":
    main()
