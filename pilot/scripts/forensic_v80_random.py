#!/usr/bin/env python3
"""v80 forensic — random 10 success + 10 failure on v80 scale=5 KG.

Same config as v71 forensic but using v80_s5 KG.
"""
import json, csv, ast, math, pickle, random, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = "pilot/data/onlykg_graph_v82_s3.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
EV_META = "data/ddxplus/release_evidences.json"
MRCONSO = "/windows/data/umls_subset/MRCONSO.RRF"
OUT = "docs/forensic_v82_random10x10.md"

KAPPA, DF_THRESHOLD, ALPHA, BETA, TAU, SHARP, LAM = 20.0, 0.12, 1.0, 0.75, 2.0, 0.5, 0.4


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
    ev_meta = json.load(open(EV_META))
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    cui2en = {icd[dn]["cui"]: dn for dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    pr_set = set(json.load(open(PR_UNIVERSE)))
    binary_evs = {ev_id for ev_id, m in ev_meta.items()
                  if m.get("data_type") == "B" and m.get("default_value") == 0}

    base_profile = {}
    all_evs = set()
    for d in dcs_list:
        if d not in G: base_profile[d] = {}; continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            cat = ed.get("category")
            if cat is None:
                if p not in pr_set: continue
            elif cat not in {"patient_reportable", "history", "demographic"}: continue
            ed_w[p] += ed.get("weight", 0.0)
        prof = {p: w/(w+KAPPA) for p, w in ed_w.items() if w > 0}
        base_profile[d] = prof
        all_evs.update(prof.keys())

    N = len(base_profile)
    df = defaultdict(int)
    for p in base_profile.values():
        for e, w in p.items():
            if w >= DF_THRESHOLD: df[e] += 1
    idf = {e: math.log((N+1)/(df_e+1))+1.0 for e, df_e in df.items()}
    profile = {d: {e: (p**ALPHA)*(idf.get(e,1.0)**BETA) for e,p in prof.items()}
               for d, prof in base_profile.items()}

    signal = defaultdict(dict)
    for ev_id in binary_evs:
        m = value_cuis.get(ev_id, {})
        cuis = set(m.get("_question", []))
        for d, prof in profile.items():
            best = 0.0
            for c in cuis:
                if c in prof:
                    idf_c = idf.get(c, 1.0)
                    factor = 1.0 / (1.0 + math.exp((idf_c - TAU) / SHARP))
                    val = prof[c] * factor
                    if val > best: best = val
            if best > 0: signal[d][ev_id] = best

    print(f"diseases={len(dcs_list)}, all_evs={len(all_evs)}", flush=True)

    def parse_evs(evs):
        pcuis = set(); ev_text = []; yes_b = set()
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                m = value_cuis.get(base, {})
                ev_text.append(f"{ev_meta.get(base,{}).get('question_en', base)} → {val}")
                for k in ("_question", val):
                    v = m.get(k, [])
                    if isinstance(v, list): pcuis.update(v)
            else:
                if ev in binary_evs: yes_b.add(ev)
                m = value_cuis.get(ev, {})
                ev_text.append(f"{ev_meta.get(ev,{}).get('question_en', ev)} (yes)")
                pcuis.update(m.get("_question", []))
        return pcuis, ev_text, yes_b

    def v71_score(pos, neg_b):
        scores = {}
        pat_vec = {e: idf.get(e, 1.0) ** BETA for e in pos}
        p_norm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
        for d, prof in profile.items():
            if not prof: scores[d] = -1e9; continue
            dot = sum(pat_vec[e] * prof[e] for e in pos if e in prof)
            d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
            ps = dot / (p_norm * d_norm)
            sig = signal.get(d, {})
            np_ = sum(sig.get(ev, 0.0) for ev in neg_b)
            nn = math.sqrt(len(neg_b)) or 1e-9
            ns = np_ / (nn * d_norm)
            scores[d] = ps - LAM * ns
        return scores

    print("Reading patients...", flush=True)
    all_rows = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            all_rows.append(row)
    random.seed(20260520)
    sample_idx = random.sample(range(len(all_rows)), min(5000, len(all_rows)))
    results = []
    for idx in sample_idx:
        row = all_rows[idx]
        true_cui = fr2cui.get(row["PATHOLOGY"])
        if true_cui not in dcs_list: continue
        evs = ast.literal_eval(row["EVIDENCES"])
        pcuis, ev_text, yes_b = parse_evs(evs)
        pos = pcuis & all_evs
        if not pos: continue
        neg_b = binary_evs - yes_b
        scores = v71_score(pos, neg_b)
        ranked = sorted(dcs_list, key=lambda d: -scores[d])
        try: rank = ranked.index(true_cui) + 1
        except: rank = len(dcs_list)
        results.append({
            "patient_id": idx, "AGE": row["AGE"], "SEX": row["SEX"],
            "true_cui": true_cui, "true_name": cui2en.get(true_cui, "?"),
            "rank": rank, "ev_text": ev_text,
            "pcuis_in_evs": list(pos),
            "top5": [(d, cui2en.get(d, "?"), scores[d]) for d in ranked[:5]],
            "true_score": scores[true_cui],
        })

    successes = [r for r in results if r["rank"] == 1]
    failures = [r for r in results if r["rank"] >= 3]
    random.seed(20260520 + 1)
    suc_sample = random.sample(successes, min(10, len(successes)))
    fail_sample = random.sample(failures, min(10, len(failures)))
    print(f"Pool: {len(successes)} success, {len(failures)} failure", flush=True)

    all_target = set()
    for c in suc_sample + fail_sample:
        all_target.update(c["pcuis_in_evs"])
        for d, _, _ in c["top5"]:
            all_target.update(profile.get(d, {}).keys())
    names = load_cui_names(all_target)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(f"# v82 (N=5 discriminative s=3) Forensic — Random 10/10\n\n")
        f.write(f"- Config: v71 algorithm + v82 KG (PubMed + LLM-aug N=5, scale=3)\n")
        f.write(f"- DDXPlus 134K @1=61.90%, MRR=0.7374\n\n")

        def render(c, label):
            f.write(f"### {label} #{c['patient_id']} — True: {c['true_name']} (`{c['true_cui']}`)\n")
            f.write(f"- Age={c['AGE']}, Sex={c['SEX']}, Rank=**{c['rank']}**, Score={c['true_score']:.4f}\n\n")
            f.write(f"**Top-5:**\n| Rank | Score | Disease | Profile | Overlap |\n|---|---|---|---|---|\n")
            for i, (d, dn, sc) in enumerate(c["top5"], 1):
                prof = profile.get(d, {})
                ov = set(c["pcuis_in_evs"]) & set(prof.keys())
                tag = " ← TRUE" if d == c['true_cui'] else ""
                f.write(f"| {i} | {sc:.4f} | {dn[:35]} (`{d}`){tag} | {len(prof)} | {len(ov)} |\n")
            if c['rank'] != 1:
                top1_cui = c["top5"][0][0]
                top1_prof = profile.get(top1_cui, {})
                true_prof = profile.get(c["true_cui"], {})
                pset = set(c["pcuis_in_evs"])
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
        for c in suc_sample: render(c, "Success")
        f.write("\n## Failure cases\n\n")
        for c in fail_sample: render(c, "Failure")

        confusion = defaultdict(int)
        for c in fail_sample:
            if c['rank'] != 1:
                confusion[(c['true_name'], c['top5'][0][1])] += 1
        f.write(f"\n## Confusion pairs\n")
        for (t, p), cnt in sorted(confusion.items(), key=lambda x: -x[1]):
            f.write(f"- {t} → {p} ({cnt}x)\n")

    print(f"Done → {OUT}", flush=True)


if __name__ == "__main__":
    main()
