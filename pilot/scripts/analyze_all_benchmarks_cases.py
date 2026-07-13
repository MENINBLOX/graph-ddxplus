#!/usr/bin/env python3
"""Forensic analysis: success/failure cases of v59 cosine on all 6 benchmarks.

For each benchmark, sample 10 success (rank=1) + 10 failure (rank>=5) and analyze.
Outputs separate markdown per benchmark.
"""
from __future__ import annotations
import json, csv, ast, math, pickle, random, sys, argparse
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = "/mnt/medkg/kg/onlykg_graph_v42_full_universal.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
MRCONSO = "/windows/data/umls_subset/MRCONSO.RRF"

KAPPA = 20.0
P_BASELINE = 0.01

MODE_CATEGORIES = {
    "lay": {"patient_reportable", "history", "demographic"},
    "clinical": {"clinical_sign", "lab_finding", "imaging_finding", "history", "demographic"},
}


def load_cui_names(target):
    names = {c: '' for c in target}
    with open(MRCONSO) as f:
        for line in f:
            parts = line.split('|')
            if len(parts) < 15: continue
            c, lang = parts[0], parts[1]
            if lang != 'ENG' or c not in names: continue
            if not names[c]:
                names[c] = parts[14]
    return names


def build_profile(G, dcs, mode, kappa, pr):
    allowed = MODE_CATEGORIES.get(mode)
    profile = {}; all_evs = set()
    for d in dcs:
        if d not in G: profile[d] = {}; continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            cat = ed.get("category")
            if cat is None:
                if mode == "lay" and pr and p not in pr: continue
            else:
                if allowed and cat not in allowed: continue
            ed_w[p] += ed.get("weight", 0.0)
        profile[d] = {p: w/(w+kappa) for p, w in ed_w.items() if w > 0}
        all_evs.update(profile[d].keys())
    return profile, all_evs


def score_cosine(pcuis, profile):
    scores = {}
    p_norm = math.sqrt(len(pcuis))
    for d, prof in profile.items():
        if not prof:
            scores[d] = -1e9; continue
        dot = sum(prof[e] for e in pcuis if e in prof)
        d_norm = math.sqrt(sum(v*v for v in prof.values()))
        scores[d] = dot / (p_norm * d_norm + 1e-9)
    return scores


def render_case_md(c, names, profile, dcs_list, label, cui_name_map):
    md = []
    true_name = cui_name_map.get(c['true_cui'], c.get('true_name', c['true_cui']))
    md.append(f"### {label} #{c['patient_id']} — True: {true_name} (`{c['true_cui']}`)\n")
    md.append(f"- Rank of true: **{c['rank']}** / {len(dcs_list)}, Score: {c['true_score']:.4f}\n")
    md.append(f"\n**Patient evidence ({len(c['patient_cuis_in_evs'])} CUIs in profile universe):**\n")
    for cui in c['patient_cuis_in_evs'][:15]:
        md.append(f"- `{cui}` {names.get(cui, '?')}")
    if len(c['patient_cuis_in_evs']) > 15:
        md.append(f"- ... +{len(c['patient_cuis_in_evs'])-15} more")
    md.append("")

    md.append(f"\n**Top-5 predictions (cosine score):**\n")
    md.append(f"| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |")
    md.append(f"|---|---|---|---|---|")
    for i, (d, sc) in enumerate(c["top5"], 1):
        prof = profile.get(d, {})
        overlap = len(set(c['patient_cuis_in_evs']) & set(prof.keys()))
        dname = cui_name_map.get(d, d)
        tag = " ← **TRUE**" if d == c['true_cui'] else ""
        md.append(f"| {i} | {sc:.4f} | {dname[:40]} (`{d}`){tag} | {overlap} | {len(prof)} |")

    if c['rank'] != 1:
        top1_cui = c['top5'][0][0]
        top1_prof = profile.get(top1_cui, {})
        true_prof = profile.get(c['true_cui'], {})
        pset = set(c['patient_cuis_in_evs'])
        only_top1 = (pset & set(top1_prof.keys())) - set(true_prof.keys())
        only_true = (pset & set(true_prof.keys())) - set(top1_prof.keys())
        both = (pset & set(top1_prof.keys())) & set(true_prof.keys())
        md.append(f"\n**Top-1 vs True comparison:**\n")
        md.append(f"- Profile size: top-1={len(top1_prof)}, true={len(true_prof)}")
        md.append(f"- Patient CUIs in **both**: {len(both)}")
        md.append(f"- **Only top-1**: {len(only_top1)}")
        for cui in sorted(only_top1)[:5]:
            md.append(f"  - `{cui}` {names.get(cui, '?')} (P={top1_prof.get(cui, 0):.2f})")
        md.append(f"- **Only true**: {len(only_true)}")
        for cui in sorted(only_true)[:5]:
            md.append(f"  - `{cui}` {names.get(cui, '?')} (P={true_prof.get(cui, 0):.2f})")
    md.append("\n---\n")
    return "\n".join(md)


def run_forensic(benchmark, dcs_list, patients, profile, all_evs, mode,
                 cui_name_map, out_path, target_n=10):
    print(f"  collecting cases for {benchmark}...", flush=True)
    successes, failures = [], []
    n = 0
    for true_cui, raw_pcuis in patients:
        pcuis = set(raw_pcuis) & all_evs
        if not pcuis: continue
        scores = score_cosine(pcuis, profile)
        ranked = sorted(profile.keys(), key=lambda d: -scores[d])
        if isinstance(true_cui, set):
            ranks = [ranked.index(t)+1 for t in true_cui if t in ranked]
            rank = min(ranks) if ranks else len(dcs_list)
            true_for_render = list(true_cui)[0] if true_cui else None
        else:
            try: rank = ranked.index(true_cui)+1
            except ValueError: rank = len(dcs_list)
            true_for_render = true_cui
        n += 1
        case = {
            "patient_id": n, "true_cui": true_for_render, "rank": rank,
            "patient_cuis_in_evs": list(pcuis),
            "top5": [(d, scores[d]) for d in ranked[:5]],
            "true_score": scores.get(true_for_render, 0),
        }
        if rank == 1 and len(successes) < target_n: successes.append(case)
        elif rank >= 5 and len(failures) < target_n: failures.append(case)
        if len(successes) >= target_n and len(failures) >= target_n: break
        if n > 5000: break
    print(f"    collected {len(successes)} success, {len(failures)} failure", flush=True)

    # Names
    print(f"  loading names...", flush=True)
    target = set()
    for c in successes + failures:
        target.update(c['patient_cuis_in_evs'])
        for d, _ in c['top5']:
            target.update(profile.get(d, {}).keys())
        if c['true_cui']: target.add(c['true_cui'])
    names = load_cui_names(target)

    # Write
    print(f"  writing {out_path}...", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# {benchmark} Forensic Analysis — v59 cosine (v42 KG, {mode} mode)\n\n")
        f.write(f"- KG: {GRAPH}\n- Mode: {mode}\n- |dcs|={len(dcs_list)}, |all_evs|={len(all_evs)}\n\n")
        f.write(f"## Success cases (rank=1, {len(successes)} samples)\n\n")
        for c in successes:
            f.write(render_case_md(c, names, profile, dcs_list, "Success", cui_name_map))
        f.write(f"\n## Failure cases (rank≥5, {len(failures)} samples)\n\n")
        for c in failures:
            f.write(render_case_md(c, names, profile, dcs_list, "Failure", cui_name_map))
        if successes and failures:
            import statistics
            s_n = [len(c['patient_cuis_in_evs']) for c in successes]
            f_n = [len(c['patient_cuis_in_evs']) for c in failures]
            s_sc = [c['true_score'] for c in successes]
            f_sc = [c['true_score'] for c in failures]
            f_gap = [c['top5'][0][1] - c['true_score'] for c in failures]
            f.write(f"\n## Aggregate\n\n")
            f.write(f"- Success: avg pcuis={statistics.mean(s_n):.1f}, true_score={statistics.mean(s_sc):.4f}\n")
            f.write(f"- Failure: avg pcuis={statistics.mean(f_n):.1f}, true_score={statistics.mean(f_sc):.4f}\n")
            f.write(f"- Failure score gap (top1 - true): {statistics.mean(f_gap):.4f}\n")


# ===== Loaders =====
def load_ddxplus():
    value_cuis = json.load(open(VALUE_CUIS))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    cui2name = {icd[dn]["cui"]: dn for dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= 5000: break
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
    return dcs_list, patients, cui2name


def load_symcat():
    parsed = json.load(open("data/symcat/symcat_parsed.json"))
    sym_map = json.load(open("data/symcat/symptom_umls_mapping.json"))["mapping"]
    dis_map = json.load(open("data/symcat/disease_umls_mapping.json"))["mapping"]
    sym2cui = {n: v["umls_cui"] for n, v in sym_map.items()}
    dis2cui = {n: v["umls_cui"] for n, v in dis_map.items()}
    cui2name = {v["umls_cui"]: n for n, v in dis_map.items()}
    cand = [(n, dis2cui[n]) for n in parsed["disease_symptom_pairs"] if dis2cui.get(n)]
    dcs_list = sorted({c for _, c in cand})
    random.seed(42)
    patients = []
    for dname, true_cui in cand:
        sym_prob = {sym2cui.get(s[0]): s[1]/100.0
                    for s in parsed["disease_symptom_pairs"][dname]
                    if sym2cui.get(s[0])}
        for _ in range(50):
            pcuis = {c for c, p in sym_prob.items() if random.random() < p}
            if pcuis: patients.append((true_cui, pcuis))
    return dcs_list, patients, cui2name


def load_rarebench(dataset):
    hpo2cui = {k: v["umls_cui"] for k, v in
               json.load(open("data/rarebench/hpo_umls_mapping.json"))["mapping"].items()}
    dis_map = json.load(open("data/rarebench/disease_umls_mapping.json"))["mapping"]
    dis2cui = {k: v["umls_cui"] for k, v in dis_map.items()}
    cui2name = {}
    for k, v in dis_map.items():
        if v.get("umls_cui"):
            cui2name[v["umls_cui"]] = v.get("disease_name", k)
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
    return dcs_list, patients, cui2name


def main():
    print(f"Loading KG...", flush=True)
    G = pickle.load(open(GRAPH, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))

    bench_specs = [
        ("symcat", "lay", load_symcat),
        ("rarebench_RAMEDIS", "clinical", lambda: load_rarebench("RAMEDIS")),
        ("rarebench_HMS", "clinical", lambda: load_rarebench("HMS")),
        ("rarebench_MME", "clinical", lambda: load_rarebench("MME")),
        ("rarebench_LIRICAL", "clinical", lambda: load_rarebench("LIRICAL")),
    ]
    for name, mode, loader in bench_specs:
        print(f"\n=== {name} ({mode}) ===", flush=True)
        dcs_list, patients, cui2name = loader()
        print(f"  diseases={len(dcs_list)}, patients={len(patients)}", flush=True)
        profile, all_evs = build_profile(G, dcs_list, mode, KAPPA, pr if mode=="lay" else None)
        print(f"  all_evs={len(all_evs)}", flush=True)
        out_path = f"docs/forensic_{name}_cases.md"
        run_forensic(name, dcs_list, patients, profile, all_evs, mode, cui2name, out_path)
        print(f"  done.", flush=True)


if __name__ == "__main__":
    main()
