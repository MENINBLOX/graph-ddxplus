#!/usr/bin/env python3
"""v57 KG-NB with IDF weighting on evidence CUIs.

Problem: KG profile contains "Pain" in 39/49 diseases → no discrimination.
Fix: down-weight (or filter) CUIs that appear in many disease profiles.

IDF(E) = log(N_D / df(E)) where df(E) = # diseases with E in profile

Two strategies:
  --idf_mode skip: Skip CUIs with df > skip_threshold (e.g., > N_D/2)
  --idf_mode weight: Multiply log P(E|D) and log(1-P(E|D)) by IDF(E)
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, gzip, argparse, random
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
CUI_EXPANSION = "pilot/data/cui_expansion_lookup.json.gz"

MODE_CATEGORIES = {
    "lay": {"patient_reportable", "history", "demographic"},
    "clinical": {"clinical_sign", "lab_finding", "imaging_finding", "history", "demographic"},
    "comprehensive": {"patient_reportable", "clinical_sign", "lab_finding",
                      "imaging_finding", "history", "demographic"},
    "all": None,
}


def load_expansion():
    with gzip.open(CUI_EXPANSION, "rt") as f:
        return json.load(f)


def expand_cuis(cuis, exp, depth_par=1, use_syn=True):
    out = set(cuis); frontier = set(cuis)
    for _ in range(depth_par):
        nf = set()
        for c in frontier: nf.update(exp["parents"].get(c, []))
        nf -= out; out.update(nf)
        if use_syn:
            for c in list(nf) + list(frontier):
                out.update(exp["synonyms"].get(c, []))
        frontier = nf
        if not frontier: break
    return out


def build_kg_profile(G, disease_cuis, mode, kappa, pr_fallback=None):
    allowed_cats = MODE_CATEGORIES.get(mode)
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
                if allowed_cats is not None and cat not in allowed_cats: continue
            ed_w[p] += ed.get("weight", 0.0)
        profile[d] = {p: w / (w + kappa) for p, w in ed_w.items() if w > 0}
        all_evs.update(profile[d].keys())
    return profile, all_evs


def compute_idf(profile, n_diseases):
    """For each evidence CUI, IDF = log(N / df)."""
    df = Counter()
    for d, prof_d in profile.items():
        for e in prof_d:
            df[e] += 1
    idf = {e: math.log(n_diseases / count) for e, count in df.items()}
    return idf, df


def nb_score_idf(patient_cuis, profile, all_evs, log_prior, p_baseline,
                 idf=None, idf_mode="weight", idf_skip_df=25,
                 idf_weight_floor=0.0, smooth=1e-3):
    scores = {}
    for d, prof_d in profile.items():
        log_p = log_prior
        for e in all_evs:
            if idf is not None and idf_mode == "skip":
                df_val = math.exp(-idf.get(e, 0)) * len(profile)
                if df_val >= idf_skip_df: continue
            p = prof_d.get(e, p_baseline)
            p = max(smooth, min(1 - smooth, p))
            term_pres = math.log(p)
            term_abs = math.log(1 - p)
            if idf is not None and idf_mode == "weight":
                w = max(idf_weight_floor, idf.get(e, 0))
                term_pres *= w; term_abs *= w
            if e in patient_cuis:
                log_p += term_pres
            else:
                log_p += term_abs
        scores[d] = log_p
    return scores


# === Benchmark loaders (same as v56) ===
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
    candidate = [(n, dis2cui[n]) for n in parsed["disease_symptom_pairs"] if dis2cui.get(n)]
    dcs_list = sorted({c for _, c in candidate})
    random.seed(seed)
    patients = []
    for dname, true_cui in candidate:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--benchmark", choices=["ddxplus", "symcat", "rarebench"], required=True)
    ap.add_argument("--rb_dataset", default="RAMEDIS")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--n_patients_per_d", type=int, default=100)
    ap.add_argument("--mode", choices=list(MODE_CATEGORIES), default="lay")
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--p_baseline", type=float, default=0.01)
    ap.add_argument("--smooth", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--expand_depth", type=int, default=0)
    ap.add_argument("--idf_mode", choices=["none", "weight", "skip"], default="weight")
    ap.add_argument("--idf_skip_df", type=int, default=25,
                    help="for idf_mode=skip: skip CUIs appearing in >= this many diseases")
    ap.add_argument("--idf_weight_floor", type=float, default=0.1,
                    help="for idf_mode=weight: minimum IDF weight (0 = no penalty for ubiquitous)")
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    pr_fallback = set(json.load(open(PR_UNIVERSE)))
    exp = load_expansion() if args.expand_depth > 0 else None

    if args.benchmark == "ddxplus":
        dcs_list, patients = load_ddxplus(args.n)
    elif args.benchmark == "symcat":
        dcs_list, patients = load_symcat(args.n_patients_per_d, args.seed)
    else:
        dcs_list, patients = load_rarebench(args.rb_dataset)

    kg_profile, all_evs = build_kg_profile(G, dcs_list, args.mode, args.kappa, pr_fallback)
    log_prior = math.log(1.0 / len(dcs_list))

    # Compute IDF
    idf = None
    if args.idf_mode != "none":
        idf, df = compute_idf(kg_profile, len(dcs_list))
        # Show top noise (high df)
        top_noise = sorted(df.items(), key=lambda x: -x[1])[:5]
        print(f"  Top-5 noise CUIs (high df): {top_noise}", flush=True)

    print(f"=== v57 KG-NB+IDF on {args.benchmark}"
          f"{('/'+args.rb_dataset) if args.benchmark=='rarebench' else ''} ===",
          flush=True)
    print(f"  mode={args.mode}, kappa={args.kappa}, expand={args.expand_depth}, "
          f"idf_mode={args.idf_mode}", flush=True)
    print(f"  diseases={len(dcs_list)}, |all_evs|={len(all_evs)}, patients={len(patients)}",
          flush=True)

    n = 0; c1=c3=c5=c10=0; rr_sum=0.0
    for true_cui, raw_pcuis in patients:
        if args.expand_depth > 0:
            patient_cuis = expand_cuis(raw_pcuis, exp, args.expand_depth) & all_evs
        else:
            patient_cuis = set(raw_pcuis) & all_evs
        if not patient_cuis: continue
        scores = nb_score_idf(patient_cuis, kg_profile, all_evs, log_prior,
                              args.p_baseline, idf, args.idf_mode,
                              args.idf_skip_df, args.idf_weight_floor, args.smooth)
        ranked = sorted(scores.keys(), key=lambda d: -scores[d])
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
        rr_sum += 1.0/rank
        if n % 10000 == 0:
            print(f"  [{n}] @1={100*c1/n:.2f}%", flush=True)

    print(f"v57 KG-NB+IDF {args.benchmark}"
          f"{('/'+args.rb_dataset) if args.benchmark=='rarebench' else ''} "
          f"mode={args.mode} idf={args.idf_mode}: "
          f"@1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% "
          f"@10={100*c10/n:.2f}% MRR={rr_sum/n:.4f} N={n}", flush=True)


if __name__ == "__main__":
    main()
