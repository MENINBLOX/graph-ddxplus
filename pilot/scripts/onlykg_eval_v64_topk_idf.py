#!/usr/bin/env python3
"""v64 — v63 IDF + per-disease top-K profile pruning.

Forensic v63 finding: Bronchitis lost 5/5 to Bronchiolitis. Both are
respiratory; their KG profiles share too many low-confidence edges.
Hypothesis: pruning each disease to its top-K strongest edges removes
noise that lets confusable diseases steal probability mass.
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
        # Hill -> per-disease top-K
        prof = {p: w/(w+kappa) for p, w in ed_w.items() if w > 0}
        if top_k and len(prof) > top_k:
            top = sorted(prof.items(), key=lambda x: -x[1])[:top_k]
            prof = dict(top)
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
    return {e: math.log((N+1)/(df_e+1)) + 1.0 for e, df_e in df.items()}


def reweight(profile, idf, alpha, beta):
    return {d: {e: (p**alpha) * (idf.get(e, 1.0)**beta) for e, p in prof.items()}
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


def load_symcat(n_patients_per_d, seed):
    import random
    parsed = json.load(open("data/symcat/symcat_parsed.json"))
    sym_map = json.load(open("data/symcat/symptom_umls_mapping.json"))["mapping"]
    dis_map = json.load(open("data/symcat/disease_umls_mapping.json"))["mapping"]
    sym2cui = {n: v["umls_cui"] for n, v in sym_map.items()}
    dis2cui = {n: v["umls_cui"] for n, v in dis_map.items()}
    cand = [(n, dis2cui[n]) for n in parsed["disease_symptom_pairs"] if dis2cui.get(n)]
    dcs_list = sorted({c for _, c in cand})
    random.seed(seed)
    patients = []
    for dname, true_cui in cand:
        sym_prob = {sym2cui.get(s[0]): s[1]/100.0
                    for s in parsed["disease_symptom_pairs"][dname]
                    if sym2cui.get(s[0])}
        for _ in range(n_patients_per_d):
            pcuis = {c for c, p in sym_prob.items() if random.random() < p}
            if pcuis: patients.append((true_cui, pcuis))
    return dcs_list, patients


def load_rarebench(dataset):
    hpo2cui = {k: v["umls_cui"] for k, v in
               json.load(open("data/rarebench/hpo_umls_mapping.json"))["mapping"].items()}
    dis2cui = {k: v["umls_cui"] for k, v in
               json.load(open("data/rarebench/disease_umls_mapping.json"))["mapping"].items()}
    rows = [json.loads(l) for l in open(f"data/rarebench/data/{dataset}.jsonl")]
    cand = set()
    for r in rows:
        for did in r.get("RareDisease", []):
            c = dis2cui.get(did)
            if c: cand.add(c)
    dcs_list = sorted(cand)
    patients = []
    for r in rows:
        pcuis = {hpo2cui.get(hp) for hp in r.get("Phenotype", [])}
        pcuis = {c for c in pcuis if c}
        true_set = {dis2cui.get(did) for did in r.get("RareDisease", [])}
        true_set = {c for c in true_set if c and c in cand}
        if pcuis and true_set:
            patients.append((true_set, pcuis))
    return dcs_list, patients


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
        if isinstance(true_cui, set):
            ranks = [ranked.index(t)+1 for t in true_cui if t in ranked]
            rank = min(ranks) if ranks else len(dcs_list)
        else:
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
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--top_k_sweep", type=str, default="20,30,50,75,100,150,200,None")
    ap.add_argument("--benchmark", default="ddxplus", choices=["ddxplus","symcat","rarebench"])
    ap.add_argument("--rb_dataset", default="RAMEDIS")
    ap.add_argument("--n_patients_per_d", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))
    if args.benchmark == "ddxplus":
        dcs_list, patients = load_ddxplus(args.n)
    elif args.benchmark == "symcat":
        dcs_list, patients = load_symcat(args.n_patients_per_d, args.seed)
    else:
        dcs_list, patients = load_rarebench(args.rb_dataset)

    print(f"=== v64 top-K pruning + IDF ===", flush=True)
    print(f"  N={args.n}, alpha={args.alpha}, beta={args.beta}, df_thr={args.df_threshold}",
          flush=True)
    for tk in args.top_k_sweep.split(","):
        tk = None if tk == "None" else int(tk)
        base, all_evs = build_profile(G, dcs_list, args.mode, args.kappa, pr, top_k=tk)
        # IDF recomputed AFTER pruning so universal CUIs are still detected
        idf = compute_idf(base, args.df_threshold)
        rew = reweight(base, idf, args.alpha, args.beta)
        r = evaluate(rew, idf, args.beta, patients, all_evs, dcs_list)
        # profile size stats
        sizes = [len(p) for p in rew.values()]
        avg_size = sum(sizes)/len(sizes)
        print(f"  top_k={str(tk):>4}: @1={r['at1']:.2f}% @3={r['at3']:.2f}% "
              f"@5={r['at5']:.2f}% @10={r['at10']:.2f}% MRR={r['mrr']:.4f} "
              f"avg_profile={avg_size:.0f}", flush=True)


if __name__ == "__main__":
    main()
