#!/usr/bin/env python3
"""v63 — Discriminative IDF reweighting on cosine.

Forensic finding (v49): failure mode is disease-cluster confusion. The true
disease KG contains every patient CUI but loses to a clinically adjacent
disease whose KG has stronger weight on shared, non-discriminative CUIs
(Pain, Chest, Anatomic Site, Erythema, ...).

Fix: reweight each CUI by IDF over the disease universe:
    idf(E) = log(N / df(E))
    df(E)  = number of diseases whose profile contains E with P >= threshold
    final_w(E,D) = P(E|D) ^ alpha  *  idf(E) ^ beta

alpha controls profile-weight contribution, beta controls IDF strength.
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
    "comprehensive": {"patient_reportable", "clinical_sign", "lab_finding",
                      "imaging_finding", "history", "demographic"},
    "all": None,
}


def build_profile(G, disease_cuis, mode, kappa, pr_fallback=None):
    allowed = MODE_CATEGORIES.get(mode)
    profile = {}; all_evs = set()
    for d in disease_cuis:
        if d not in G: profile[d] = {}; continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            cat = ed.get("category")
            if cat is None:
                if mode == "lay" and pr_fallback is not None and p not in pr_fallback:
                    continue
            else:
                if allowed is not None and cat not in allowed: continue
            ed_w[p] += ed.get("weight", 0.0)
        profile[d] = {p: w/(w+kappa) for p, w in ed_w.items() if w > 0}
        all_evs.update(profile[d].keys())
    return profile, all_evs


def compute_idf(profile, df_threshold):
    """idf(E) = log((N+1) / (df(E)+1)) + 1 — smoothed."""
    N = len(profile)
    df = defaultdict(int)
    for d, prof in profile.items():
        for e, p in prof.items():
            if p >= df_threshold:
                df[e] += 1
    idf = {e: math.log((N + 1) / (df_e + 1)) + 1.0 for e, df_e in df.items()}
    return idf, df


def reweight_profile(profile, idf, alpha, beta):
    """final(E,D) = P(E|D)^alpha * idf(E)^beta."""
    new_profile = {}
    for d, prof in profile.items():
        new_profile[d] = {
            e: (p ** alpha) * (idf.get(e, 1.0) ** beta)
            for e, p in prof.items()
        }
    return new_profile


def score_cosine(pcuis, profile, idf, beta):
    """Cosine where the patient indicator is also IDF-weighted."""
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
    patients = []
    n = 0
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
            patients.append((true_cui, pcuis))
            n += 1
    return dcs_list, patients


def load_symcat(n_patients_per_d, seed):
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


def evaluate(profile, idf, beta, patients, all_evs, dcs_list):
    n = c1 = c3 = c5 = c10 = 0; rr = 0.0
    for true_cui, raw_pcuis in patients:
        pcuis = set(raw_pcuis) & all_evs
        if not pcuis: continue
        scores = score_cosine(pcuis, profile, idf, beta)
        ranked = sorted(profile.keys(), key=lambda d: -scores[d])
        n += 1
        if isinstance(true_cui, set):
            ranks = [ranked.index(t)+1 for t in true_cui if t in ranked]
            rank = min(ranks) if ranks else len(dcs_list)
        else:
            try: rank = ranked.index(true_cui)+1
            except ValueError: rank = len(dcs_list)
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        rr += 1.0/rank
    if n == 0: return None
    return {"n": n, "at1": 100*c1/n, "at3": 100*c3/n, "at5": 100*c5/n,
            "at10": 100*c10/n, "mrr": rr/n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--benchmark", required=True, choices=["ddxplus","symcat","rarebench"])
    ap.add_argument("--rb_dataset", default="RAMEDIS")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--n_patients_per_d", type=int, default=100)
    ap.add_argument("--mode", choices=list(MODE_CATEGORIES), default="lay")
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--df_threshold", type=float, default=0.01,
                    help="P(E|D) >= threshold counts as 'disease has E'")
    ap.add_argument("--sweep_alpha", type=str, default="1.0",
                    help="comma-separated alpha values")
    ap.add_argument("--sweep_beta", type=str, default="0.0,0.5,1.0,1.5,2.0",
                    help="comma-separated beta (IDF strength) values")
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))

    if args.benchmark == "ddxplus":
        dcs_list, patients = load_ddxplus(args.n)
    elif args.benchmark == "symcat":
        dcs_list, patients = load_symcat(args.n_patients_per_d, args.seed)
    else:
        dcs_list, patients = load_rarebench(args.rb_dataset)

    base_profile, all_evs = build_profile(G, dcs_list, args.mode, args.kappa, pr)
    idf, df = compute_idf(base_profile, args.df_threshold)

    print(f"=== v63 IDF sweep — {args.benchmark}"
          f"{'/'+args.rb_dataset if args.benchmark=='rarebench' else ''} mode={args.mode} ===",
          flush=True)
    print(f"  diseases={len(dcs_list)}, all_evs={len(all_evs)}, patients={len(patients)}",
          flush=True)
    # IDF stats
    universal = [e for e, d in df.items() if d >= len(dcs_list) * 0.8]
    rare = [e for e, d in df.items() if d == 1]
    print(f"  CUI df dist: universal(>=80%)={len(universal)} rare(df=1)={len(rare)}",
          flush=True)
    if universal:
        idf_min = min(idf[e] for e in universal)
        idf_max_rare = max(idf[e] for e in rare) if rare else 0
        print(f"  IDF range: universal_min={idf_min:.3f} rare_max={idf_max_rare:.3f}",
              flush=True)

    alphas = [float(x) for x in args.sweep_alpha.split(",")]
    betas = [float(x) for x in args.sweep_beta.split(",")]
    best = None
    for alpha in alphas:
        for beta in betas:
            profile = reweight_profile(base_profile, idf, alpha, beta)
            r = evaluate(profile, idf, beta, patients, all_evs, dcs_list)
            if r is None:
                print(f"  alpha={alpha} beta={beta}: NO PATIENTS", flush=True)
                continue
            print(f"  alpha={alpha:.2f} beta={beta:.2f}: "
                  f"@1={r['at1']:.2f}% @3={r['at3']:.2f}% @5={r['at5']:.2f}% "
                  f"@10={r['at10']:.2f}% MRR={r['mrr']:.4f} N={r['n']}", flush=True)
            if best is None or r["at1"] > best[2]["at1"]:
                best = (alpha, beta, r)
    if best:
        a, b, r = best
        print(f"\nBEST: alpha={a} beta={b} @1={r['at1']:.2f}% MRR={r['mrr']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
