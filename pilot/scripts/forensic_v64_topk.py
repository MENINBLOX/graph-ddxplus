#!/usr/bin/env python3
"""v64 forensic — top-K pruned cosine+IDF residual failures."""
from __future__ import annotations
import json, csv, ast, math, pickle, random, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = "pilot/data/onlykg_graph_v49_v5_full.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
MRCONSO = "/windows/data/umls_subset/MRCONSO.RRF"
OUT = "docs/forensic_ddxplus_v64_topk.md"

KAPPA = 20.0
DF_THRESHOLD = 0.12
ALPHA = 1.0
BETA = 0.75
TOP_K = 80


def load_cui_names(target):
    names = {c: '' for c in target}
    with open(MRCONSO) as f:
        for line in f:
            parts = line.split('|')
            if len(parts) < 15: continue
            c, lang = parts[0], parts[1]
            if lang != 'ENG': continue
            if c in names and not names[c]:
                names[c] = parts[14]
    return names


def main():
    G = pickle.load(open(GRAPH, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f: ev_meta = json.load(f)

    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    cui2en = {icd[dn]["cui"]: dn for dn in icd}
    dcs_list = sorted(set(fr2cui.values()))

    pr_set = set(json.load(open(PR_UNIVERSE)))
    base_profile = {}
    all_evs = set()
    for d in dcs_list:
        if d not in G: base_profile[d] = {}; continue
        ed_w = defaultdict(float)
        for _, p, edge in G.out_edges(d, data=True):
            if edge.get("etype") != "HAS_PHENOTYPE": continue
            cat = edge.get("category")
            if cat is None:
                if p not in pr_set: continue
            else:
                if cat not in {"patient_reportable", "history", "demographic"}: continue
            ed_w[p] += edge.get("weight", 0.0)
        prof = {p: w/(w+KAPPA) for p, w in ed_w.items() if w > 0}
        if len(prof) > TOP_K:
            prof = dict(sorted(prof.items(), key=lambda x: -x[1])[:TOP_K])
        base_profile[d] = prof
        all_evs.update(prof.keys())

    N = len(base_profile)
    df = defaultdict(int)
    for prof in base_profile.values():
        for e, p in prof.items():
            if p >= DF_THRESHOLD: df[e] += 1
    idf = {e: math.log((N+1)/(df_e+1)) + 1.0 for e, df_e in df.items()}
    profile = {d: {e: (p**ALPHA)*(idf.get(e,1.0)**BETA) for e,p in prof.items()}
               for d, prof in base_profile.items()}

    print(f"diseases={len(dcs_list)}, all_evs={len(all_evs)}", flush=True)

    def parse_evs(evs):
        pcuis = set(); ev_text = []
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                m = value_cuis.get(base, {})
                ev_text.append(f"{ev_meta.get(base,{}).get('question_en', base)} → {val}")
                for k in ("_question", val):
                    v = m.get(k, [])
                    if isinstance(v, list): pcuis.update(v)
            else:
                m = value_cuis.get(ev, {})
                ev_text.append(f"{ev_meta.get(ev,{}).get('question_en', ev)} (yes)")
                pcuis.update(m.get("_question", []))
        return pcuis, ev_text

    def score(pcuis):
        scores = {}
        pat_vec = {e: idf.get(e, 1.0)**BETA for e in pcuis}
        p_norm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
        for d, prof in profile.items():
            if not prof: scores[d] = -1e9; continue
            dot = sum(pat_vec[e]*prof[e] for e in pcuis if e in prof)
            d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
            scores[d] = dot / (p_norm * d_norm)
        return scores

    random.seed(99)
    successes = []; failures = []
    n = 0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis, ev_text = parse_evs(evs)
            pcuis_in_evs = pcuis & all_evs
            if not pcuis_in_evs: continue
            scores = score(pcuis_in_evs)
            ranked = sorted(dcs_list, key=lambda d: -scores[d])
            try: rank = ranked.index(true_cui) + 1
            except: rank = len(dcs_list)
            n += 1
            case = {"patient_id": n, "true_cui": true_cui,
                    "true_name_en": cui2en.get(true_cui, "?"),
                    "rank": rank, "ev_text": ev_text,
                    "patient_cuis_in_evs": list(pcuis_in_evs),
                    "top5": [(d, cui2en.get(d, "?"), scores[d]) for d in ranked[:5]],
                    "true_score": scores[true_cui]}
            if rank == 1 and len(successes) < 10 and random.random() < 0.04:
                successes.append(case)
            elif rank >= 3 and len(failures) < 15:
                failures.append(case)
            if len(successes) >= 10 and len(failures) >= 15: break
            if n > 20000: break

    print(f"Collected: {len(successes)} success, {len(failures)} failure", flush=True)
    all_target = set()
    for c in successes + failures:
        all_target.update(c["patient_cuis_in_evs"])
        for d, _, _ in c["top5"]:
            all_target.update(profile.get(d, {}).keys())
    names = load_cui_names(all_target)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(f"# v64 Forensic — top_K={TOP_K}, beta={BETA}\n\n")
        f.write(f"- Config: cosine + IDF(df_thr={DF_THRESHOLD}, alpha={ALPHA}, beta={BETA}) + top-K={TOP_K}\n")
        f.write(f"- |diseases|={len(dcs_list)}, |all_evs|={len(all_evs)}\n\n")

        def render(c, label):
            f.write(f"### {label} #{c['patient_id']} — True: {c['true_name_en']} (`{c['true_cui']}`)\n")
            f.write(f"- Rank: **{c['rank']}**, Score: {c['true_score']:.4f}\n\n")
            f.write(f"**Evidence ({len(c['ev_text'])}):** ")
            f.write("; ".join(c["ev_text"][:6]))
            if len(c["ev_text"]) > 6:
                f.write(f" ... +{len(c['ev_text'])-6} more")
            f.write("\n\n")
            f.write(f"**Top-5:**\n| Rank | Score | Disease | Profile | Overlap |\n|---|---|---|---|---|\n")
            for i, (d, dn, sc) in enumerate(c["top5"], 1):
                prof = profile.get(d, {})
                ov = set(c["patient_cuis_in_evs"]) & set(prof.keys())
                tag = " ← TRUE" if d == c['true_cui'] else ""
                f.write(f"| {i} | {sc:.4f} | {dn[:35]} (`{d}`){tag} | {len(prof)} | {len(ov)} |\n")
            if c['rank'] != 1:
                top1_cui = c["top5"][0][0]
                top1_prof = profile.get(top1_cui, {})
                true_prof = profile.get(c["true_cui"], {})
                pset = set(c["patient_cuis_in_evs"])
                in_t1 = (pset & set(top1_prof.keys())) - set(true_prof.keys())
                in_tr = (pset & set(true_prof.keys())) - set(top1_prof.keys())
                f.write(f"\n**Diff:**\n- Only-in-top1 ({len(in_t1)}):\n")
                for cui in sorted(in_t1, key=lambda x: -top1_prof.get(x, 0))[:5]:
                    f.write(f"  - `{cui}` {names.get(cui,'?')} (w_t1={top1_prof.get(cui,0):.3f}, idf={idf.get(cui,1.0):.2f})\n")
                f.write(f"- Only-in-true ({len(in_tr)}):\n")
                for cui in sorted(in_tr, key=lambda x: -true_prof.get(x, 0))[:5]:
                    f.write(f"  - `{cui}` {names.get(cui,'?')} (w_tr={true_prof.get(cui,0):.3f}, idf={idf.get(cui,1.0):.2f})\n")
            f.write("\n---\n\n")

        f.write("## Success cases\n\n")
        for c in successes: render(c, "Success")
        f.write("\n## Failure cases\n\n")
        for c in failures: render(c, "Failure")

        import statistics
        f.write("\n## Aggregate\n\n")
        if successes:
            f.write(f"- Success: avg pcuis={statistics.mean([len(c['patient_cuis_in_evs']) for c in successes]):.1f}\n")
        if failures:
            f.write(f"- Failure: avg pcuis={statistics.mean([len(c['patient_cuis_in_evs']) for c in failures]):.1f}, gap={statistics.mean([c['top5'][0][2]-c['true_score'] for c in failures]):.4f}\n")
        confusion = defaultdict(int)
        for c in failures:
            if c['rank'] != 1:
                confusion[(c['true_name_en'], c['top5'][0][1])] += 1
        f.write(f"\n### Confusion pairs (true → predicted)\n")
        for (t, p), cnt in sorted(confusion.items(), key=lambda x: -x[1]):
            f.write(f"- {t} → {p} ({cnt}x)\n")

    print(f"Done → {OUT}", flush=True)


if __name__ == "__main__":
    main()
