#!/usr/bin/env python3
"""v56 KG-NB with edge-level category filter (from IE v3 KG merge).

Uses the `category` attribute on HAS_PHENOTYPE edges (added by medkg_merge_v3_kg.py)
to filter evidences at eval time, instead of using a CUI-universe filter.

Modes:
  lay         = {patient_reportable, history, demographic}
  clinical    = {clinical_sign, lab_finding, imaging_finding, history, demographic}
  comprehensive = all 6 categories

Edges without category attribute (legacy from v39) are treated as 'unknown' and
INCLUDED in all modes (backward compatibility).

Same NB scoring as v54/v55. Cross-benchmark via --benchmark.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, gzip, argparse, random
from pathlib import Path
from collections import defaultdict
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
    "all": None,  # no filter
}


def load_expansion():
    with gzip.open(CUI_EXPANSION, "rt") as f:
        return json.load(f)


def expand_cuis(cuis, exp, depth_par=1, use_syn=True):
    out = set(cuis); frontier = set(cuis)
    parents = exp["parents"]; synonyms = exp["synonyms"]
    for _ in range(depth_par):
        nf = set()
        for c in frontier: nf.update(parents.get(c, []))
        nf -= out; out.update(nf)
        if use_syn:
            for c in list(nf) + list(frontier):
                out.update(synonyms.get(c, []))
        frontier = nf
        if not frontier: break
    return out


def build_kg_profile_categorized(G, disease_cuis, mode, kappa, pr_fallback=None):
    """Build P(E|D) using category filter on edges.

    For edges with `category` attribute: include if category ∈ MODE_CATEGORIES[mode].
    For legacy edges without `category`: fall back to pr_fallback (CUI set) for lay mode,
                                          include for clinical/comprehensive modes.
    """
    allowed_cats = MODE_CATEGORIES.get(mode)  # None means all
    profile = {}
    all_evs = set()
    for d in disease_cuis:
        if d not in G:
            profile[d] = {}; continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            cat = ed.get("category")
            if cat is None:
                # Legacy edge — apply CUI-based fallback for lay; allow else
                if mode == "lay" and pr_fallback is not None and p not in pr_fallback:
                    continue
            else:
                if allowed_cats is not None and cat not in allowed_cats:
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
    candidate = []
    for dname in parsed["disease_symptom_pairs"]:
        cui = dis2cui.get(dname)
        if cui: candidate.append((dname, cui))
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

    kg_profile, all_evs = build_kg_profile_categorized(
        G, dcs_list, args.mode, args.kappa, pr_fallback)
    log_prior = math.log(1.0 / len(dcs_list))

    print(f"=== v56 KG-NB+Cat on {args.benchmark}"
          f"{('/'+args.rb_dataset) if args.benchmark=='rarebench' else ''} ===",
          flush=True)
    print(f"  mode={args.mode}, kappa={args.kappa}, expand={args.expand_depth}", flush=True)
    print(f"  diseases={len(dcs_list)}, |all_evs|={len(all_evs)}, patients={len(patients)}",
          flush=True)

    n = 0; c1=c3=c5=c10=0; rr_sum=0.0
    for true_cui, raw_pcuis in patients:
        if args.expand_depth > 0:
            patient_cuis = expand_cuis(raw_pcuis, exp, args.expand_depth) & all_evs
        else:
            patient_cuis = set(raw_pcuis) & all_evs
        if not patient_cuis: continue
        scores = nb_score(patient_cuis, kg_profile, all_evs, log_prior,
                          args.p_baseline, args.smooth)
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

    print(f"v56 KG-NB+Cat {args.benchmark}"
          f"{('/'+args.rb_dataset) if args.benchmark=='rarebench' else ''} "
          f"mode={args.mode} kappa={args.kappa} expand={args.expand_depth}: "
          f"@1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% "
          f"@10={100*c10/n:.2f}% MRR={rr_sum/n:.4f} N={n}", flush=True)


if __name__ == "__main__":
    main()
