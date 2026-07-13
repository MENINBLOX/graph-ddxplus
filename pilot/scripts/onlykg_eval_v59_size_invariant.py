#!/usr/bin/env python3
"""v59 KG-NB size-invariant variants.

Forensic finding: large KG profiles get diluted because NB penalizes absent
evidence per CUI in profile. Disease with 148 CUIs gets more `log(1-P(E|D))`
penalty than disease with 82 CUIs even when patient overlap is identical.

Three variants to compare:
  --score patient_only   : Σ_{E∈patient} log P(E|D)
                            (no absent penalty, only score patient's evidence)
  --score per_evidence   : (1/|profile|) * Σ_{E∈profile} NB term
                            (normalize NB by profile size)
  --score cosine         : cosine similarity between patient and profile vectors
  --score nb_baseline    : original v56 NB for comparison
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


def score_patient_only(pcuis, profile, log_prior, p_baseline, smooth=1e-3):
    scores = {}
    for d, prof in profile.items():
        s = log_prior
        for e in pcuis:
            p = prof.get(e, p_baseline)
            p = max(smooth, min(1-smooth, p))
            s += math.log(p)
        scores[d] = s
    return scores


def score_per_evidence(pcuis, profile, all_evs, log_prior, p_baseline, smooth=1e-3):
    """NB but each disease's score normalized by its profile size."""
    scores = {}
    for d, prof in profile.items():
        if not prof:
            scores[d] = log_prior; continue
        s = 0.0
        for e in all_evs:
            p = prof.get(e, p_baseline)
            p = max(smooth, min(1-smooth, p))
            term = math.log(p) if e in pcuis else math.log(1-p)
            s += term
        scores[d] = log_prior + s / max(len(prof), 1)
    return scores


def score_cosine(pcuis, profile, log_prior):
    """Cosine similarity between patient (indicator vector) and profile (weighted)."""
    scores = {}
    p_norm = math.sqrt(len(pcuis))  # ||patient|| = sqrt of count of present CUIs
    for d, prof in profile.items():
        if not prof:
            scores[d] = -1e9; continue
        dot = sum(prof[e] for e in pcuis if e in prof)
        d_norm = math.sqrt(sum(v*v for v in prof.values()))
        scores[d] = dot / (p_norm * d_norm + 1e-9)
    return scores


def score_nb_baseline(pcuis, profile, all_evs, log_prior, p_baseline, smooth=1e-3):
    """v56 baseline: NB over all_evs."""
    scores = {}
    for d, prof in profile.items():
        s = log_prior
        for e in all_evs:
            p = prof.get(e, p_baseline)
            p = max(smooth, min(1-smooth, p))
            s += math.log(p) if e in pcuis else math.log(1-p)
        scores[d] = s
    return scores


# Loaders
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--benchmark", required=True, choices=["ddxplus","symcat","rarebench"])
    ap.add_argument("--rb_dataset", default="RAMEDIS")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--n_patients_per_d", type=int, default=100)
    ap.add_argument("--mode", choices=list(MODE_CATEGORIES), default="lay")
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--p_baseline", type=float, default=0.01)
    ap.add_argument("--smooth", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score", choices=["nb_baseline","patient_only","per_evidence","cosine"],
                    required=True)
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))

    if args.benchmark == "ddxplus":
        dcs_list, patients = load_ddxplus(args.n)
    elif args.benchmark == "symcat":
        dcs_list, patients = load_symcat(args.n_patients_per_d, args.seed)
    else:
        dcs_list, patients = load_rarebench(args.rb_dataset)

    profile, all_evs = build_profile(G, dcs_list, args.mode, args.kappa, pr)
    log_prior = math.log(1.0/len(dcs_list))
    print(f"=== v59 score={args.score} bench={args.benchmark}"
          f"{'/'+args.rb_dataset if args.benchmark=='rarebench' else ''} mode={args.mode} ===",
          flush=True)
    print(f"  diseases={len(dcs_list)}, all_evs={len(all_evs)}, patients={len(patients)}",
          flush=True)

    n = c1 = c3 = c5 = c10 = 0; rr = 0.0
    for true_cui, raw_pcuis in patients:
        pcuis = set(raw_pcuis) & all_evs
        if not pcuis: continue
        if args.score == "patient_only":
            scores = score_patient_only(pcuis, profile, log_prior, args.p_baseline, args.smooth)
        elif args.score == "per_evidence":
            scores = score_per_evidence(pcuis, profile, all_evs, log_prior, args.p_baseline, args.smooth)
        elif args.score == "cosine":
            scores = score_cosine(pcuis, profile, log_prior)
        else:
            scores = score_nb_baseline(pcuis, profile, all_evs, log_prior, args.p_baseline, args.smooth)
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
        if n % 5000 == 0:
            print(f"  [{n}] @1={100*c1/n:.2f}%", flush=True)

    print(f"v59 {args.score} {args.benchmark}: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% "
          f"@5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr/n:.4f} N={n}", flush=True)


if __name__ == "__main__":
    main()
