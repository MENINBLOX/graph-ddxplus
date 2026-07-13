#!/usr/bin/env python3
"""v55 KG-NB with patient CUI hierarchical expansion.

Identical to v54 KG-NB except: at eval time, each patient CUI is expanded
via UMLS PAR (parents) + SY (synonyms) up to depth N, before NB scoring.

Goal: resolve vocabulary granularity mismatch between benchmark CUIs (specific)
and KG CUIs (often more general).

Example: patient has C0238995 (Sharp chest pain) but KG profile only has
         C0008031 (Chest pain). With expansion, patient set adds C0008031
         via PAR relation → NB scoring matches.

Usage:
  python onlykg_eval_v55_kgnb_expand.py --benchmark ddxplus
  python onlykg_eval_v55_kgnb_expand.py --benchmark symcat
  python onlykg_eval_v55_kgnb_expand.py --benchmark rarebench --rb_dataset RAMEDIS
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


def load_expansion():
    with gzip.open(CUI_EXPANSION, "rt") as f:
        d = json.load(f)
    return d  # {parents, synonyms, narrower}


def expand_cuis(cuis, exp, depth_par=2, use_syn=True, use_children=False):
    """Expand a CUI set with ancestors and synonyms."""
    out = set(cuis)
    frontier = set(cuis)
    parents = exp["parents"]
    synonyms = exp["synonyms"]
    narrower = exp["narrower"]
    for _ in range(depth_par):
        new_frontier = set()
        for c in frontier:
            new_frontier.update(parents.get(c, []))
        new_frontier -= out
        out.update(new_frontier)
        if use_syn:
            for c in list(new_frontier) + list(frontier):
                out.update(synonyms.get(c, []))
        if use_children:
            for c in frontier:
                out.update(narrower.get(c, []))
        frontier = new_frontier
        if not frontier: break
    return out


def build_kg_profile(G, disease_cuis, allowed_cuis, kappa):
    profile = {}
    all_evs = set()
    for d in disease_cuis:
        if d not in G:
            profile[d] = {}; continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            if allowed_cuis is not None and p not in allowed_cuis: continue
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


# === Benchmark-specific loaders ===
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
        sym_prob = {sym2cui.get(s[0]): s[1]/100.0 for s in parsed["disease_symptom_pairs"][dname]
                    if sym2cui.get(s[0])}
        for _ in range(n_patients_per_d):
            pcuis = {c for c, p in sym_prob.items() if random.random() < p}
            if pcuis:
                patients.append((true_cui, pcuis))
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
    ap.add_argument("--evidence_categories", choices=["lay", "clinical", "comprehensive"],
                    default="lay")
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--p_baseline", type=float, default=0.01)
    ap.add_argument("--smooth", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--expand_depth", type=int, default=2,
                    help="UMLS parent expansion depth")
    ap.add_argument("--use_syn", type=int, default=1, help="include SY synonyms")
    ap.add_argument("--use_children", type=int, default=0,
                    help="include CHD children too (broader expansion)")
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    exp = load_expansion()

    if args.benchmark == "ddxplus":
        dcs_list, patients = load_ddxplus(args.n)
    elif args.benchmark == "symcat":
        dcs_list, patients = load_symcat(args.n_patients_per_d, args.seed)
    else:
        dcs_list, patients = load_rarebench(args.rb_dataset)

    if args.evidence_categories == "lay":
        allowed = set(json.load(open(PR_UNIVERSE)))
    else:
        allowed = None

    kg_profile, all_evs = build_kg_profile(G, dcs_list, allowed, args.kappa)
    log_prior = math.log(1.0 / len(dcs_list))

    print(f"=== v55 KG-NB+Expand on {args.benchmark}"
          f"{('/'+args.rb_dataset) if args.benchmark=='rarebench' else ''} ===", flush=True)
    print(f"  mode={args.evidence_categories}, kappa={args.kappa}, "
          f"expand_depth={args.expand_depth}, use_syn={args.use_syn}, "
          f"use_children={args.use_children}", flush=True)
    print(f"  diseases={len(dcs_list)}, |all_evs|={len(all_evs)}, patients={len(patients)}",
          flush=True)

    n = 0; c1=c3=c5=c10=0; rr_sum=0.0
    n_expansion_helps = 0
    for true_cui, raw_pcuis in patients:
        # Expand patient CUIs
        if args.expand_depth > 0:
            expanded = expand_cuis(raw_pcuis, exp,
                                   depth_par=args.expand_depth,
                                   use_syn=bool(args.use_syn),
                                   use_children=bool(args.use_children))
        else:
            expanded = set(raw_pcuis)
        patient_cuis = expanded & all_evs
        if not patient_cuis: continue

        if patient_cuis - (raw_pcuis & all_evs):
            n_expansion_helps += 1

        scores = nb_score(patient_cuis, kg_profile, all_evs, log_prior,
                          args.p_baseline, args.smooth)
        ranked = sorted(scores.keys(), key=lambda d: -scores[d])
        n += 1
        # Handle multi-truth (rarebench)
        if isinstance(true_cui, set):
            ranks = [ranked.index(t)+1 for t in true_cui if t in ranked]
            rank = min(ranks) if ranks else len(dcs_list)
        else:
            try: rank = ranked.index(true_cui) + 1
            except ValueError: rank = len(dcs_list)
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        rr_sum += 1.0/rank
        if n % 10000 == 0:
            print(f"  [{n} patients] @1={100*c1/n:.2f}%", flush=True)

    print(f"v55 KG-NB+Expand {args.benchmark}"
          f"{('/'+args.rb_dataset) if args.benchmark=='rarebench' else ''} "
          f"mode={args.evidence_categories} kappa={args.kappa} "
          f"expand={args.expand_depth}: "
          f"@1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% "
          f"@10={100*c10/n:.2f}% MRR={rr_sum/n:.4f} N={n} "
          f"(expansion_added={n_expansion_helps}/{n})", flush=True)


if __name__ == "__main__":
    main()
