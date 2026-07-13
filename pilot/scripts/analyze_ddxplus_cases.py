#!/usr/bin/env python3
"""Forensic analysis: success/failure cases of v54/v56 KG-NB on DDXPlus.

Sample 10 success (rank=1) + 10 failure (rank >= 5) patients.
For each, dump:
  - True disease, predicted top-5
  - Patient evidence CUIs (with names)
  - For each top disease: how patient CUIs overlap with KG profile
  - Score breakdown
"""
from __future__ import annotations
import json, csv, ast, math, pickle, random, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = "/mnt/medkg/kg/onlykg_graph_v42_full_universal.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
MRCONSO = "/windows/data/umls_subset/MRCONSO.RRF"
OUT = "docs/forensic_ddxplus_cases.md"

KAPPA = 20.0
P_BASELINE = 0.01
SMOOTH = 1e-3


def load_cui_names(target_cuis):
    """Load English names for target CUIs from MRCONSO."""
    names = {c: '' for c in target_cuis}
    n = 0
    with open(MRCONSO) as f:
        for line in f:
            n += 1
            parts = line.split('|')
            if len(parts) < 15: continue
            c, lang = parts[0], parts[1]
            if lang != 'ENG': continue
            if c in names and not names[c]:
                names[c] = parts[14]
    return names


def main():
    print("Loading KG + data...", flush=True)
    G = pickle.load(open(GRAPH, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f: ev_meta = json.load(f)

    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    cui2en = {icd[dn]["cui"]: dn for dn in icd}
    dcs_list = sorted(set(fr2cui.values()))

    # Build profile (lay mode)
    pr_set = set(json.load(open(PR_UNIVERSE)))
    profile = {}
    all_evs = set()
    for d in dcs_list:
        if d not in G: profile[d] = {}; continue
        ed_w = defaultdict(float)
        for _, p, edge in G.out_edges(d, data=True):
            if edge.get("etype") != "HAS_PHENOTYPE": continue
            cat = edge.get("category")
            if cat is None:
                if p not in pr_set: continue
            else:
                if cat not in {"patient_reportable", "history", "demographic"}: continue
            ed_w[p] += edge.get("weight", 0.0)
        prof_d = {p: w/(w+KAPPA) for p, w in ed_w.items() if w > 0}
        profile[d] = prof_d
        all_evs.update(prof_d.keys())
    log_prior = math.log(1.0/len(dcs_list))
    print(f"  diseases={len(dcs_list)}, all_evs={len(all_evs)}", flush=True)

    def parse_evs(evs):
        pcuis = set(); ev_text = []
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                m = value_cuis.get(base, {})
                question = ev_meta.get(base, {}).get("question_en", base)
                ev_text.append(f"{question} → {val}")
                for k in ("_question", val):
                    v = m.get(k, [])
                    if isinstance(v, list): pcuis.update(v)
            else:
                m = value_cuis.get(ev, {})
                question = ev_meta.get(ev, {}).get("question_en", ev)
                ev_text.append(f"{question} (yes)")
                pcuis.update(m.get("_question", []))
        return pcuis, ev_text

    def nb_score(patient_cuis):
        scores = {}
        for d, prof_d in profile.items():
            log_p = log_prior
            for e in all_evs:
                p = prof_d.get(e, P_BASELINE)
                p = max(SMOOTH, min(1-SMOOTH, p))
                if e in patient_cuis:
                    log_p += math.log(p)
                else:
                    log_p += math.log(1-p)
            scores[d] = log_p
        return scores

    # Eval patients, collect success/failure
    print("Eval patients to collect cases...", flush=True)
    random.seed(42)
    successes = []  # rank == 1
    failures = []   # rank >= 5

    n = 0; target_succ = 10; target_fail = 10
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis, ev_text = parse_evs(evs)
            pcuis_in_evs = pcuis & all_evs
            if not pcuis_in_evs: continue

            scores = nb_score(pcuis_in_evs)
            ranked = sorted(dcs_list, key=lambda d: -scores[d])
            try: rank = ranked.index(true_cui) + 1
            except ValueError: rank = len(dcs_list)
            n += 1

            case = {
                "patient_id": n,
                "true_cui": true_cui,
                "true_name_en": cui2en.get(true_cui, "?"),
                "true_name_fr": row["PATHOLOGY"],
                "rank": rank,
                "ev_text": ev_text,
                "patient_cuis_raw": list(pcuis),
                "patient_cuis_in_evs": list(pcuis_in_evs),
                "top5": [(d, cui2en.get(d, "?"), scores[d]) for d in ranked[:5]],
                "true_score": scores[true_cui],
            }
            if rank == 1 and len(successes) < target_succ:
                successes.append(case)
            elif rank >= 5 and len(failures) < target_fail:
                failures.append(case)
            if len(successes) >= target_succ and len(failures) >= target_fail: break
            if n > 5000: break

    print(f"Collected: {len(successes)} success, {len(failures)} failure", flush=True)
    print(f"Loading CUI names for analysis...", flush=True)

    # Collect all CUIs we need names for
    all_target = set()
    for c in successes + failures:
        all_target.update(c["patient_cuis_in_evs"])
        for d, _, _ in c["top5"]:
            all_target.update(profile.get(d, {}).keys())
    names = load_cui_names(all_target)
    print(f"  loaded {sum(1 for v in names.values() if v):,}/{len(names):,} names", flush=True)

    # Write report
    print(f"Writing {OUT}...", flush=True)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("# DDXPlus Forensic Analysis — v56 KG-NB+Cat (v42 KG, lay mode)\n\n")
        f.write(f"- KG: `{GRAPH}`\n")
        f.write(f"- Mode: lay (patient_reportable + history + demographic)\n")
        f.write(f"- κ={KAPPA}, p_baseline={P_BASELINE}\n")
        f.write(f"- |all_evs|={len(all_evs)}, |diseases|={len(dcs_list)}\n\n")

        def render_case(c, label):
            f.write(f"### {label} #{c['patient_id']} — True: {c['true_name_en']} ({c['true_cui']})\n\n")
            f.write(f"- Rank of true: **{c['rank']}** / {len(dcs_list)}\n")
            f.write(f"- True disease score: {c['true_score']:.2f}\n\n")
            f.write(f"**Patient evidence (raw input):**\n")
            for t in c["ev_text"][:20]:
                f.write(f"- {t}\n")
            if len(c["ev_text"]) > 20:
                f.write(f"- ... +{len(c['ev_text'])-20} more\n")
            f.write(f"\n**Patient CUIs in profile (count={len(c['patient_cuis_in_evs'])}):**\n")
            for cui in c["patient_cuis_in_evs"]:
                f.write(f"- `{cui}` {names.get(cui, '?')}\n")

            f.write(f"\n**Top-5 predictions:**\n")
            f.write(f"| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |\n")
            f.write(f"|---|---|---|---|---|\n")
            for i, (d, dn, sc) in enumerate(c["top5"], 1):
                prof = profile.get(d, {})
                pcui_in_prof = set(c["patient_cuis_in_evs"]) & set(prof.keys())
                tag = " ← **TRUE**" if d == c['true_cui'] else ""
                f.write(f"| {i} | {sc:.2f} | {dn[:40]} (`{d}`){tag} | {len(pcui_in_prof)} | {len(prof)} |\n")

            # Show why top-1 differs from true (if applicable)
            if c['rank'] != 1:
                top1_cui = c["top5"][0][0]
                top1_prof = profile.get(top1_cui, {})
                true_prof = profile.get(c["true_cui"], {})
                f.write(f"\n**Top-1 vs True profile comparison:**\n")
                f.write(f"- Top-1 ({c['top5'][0][1]}) has {len(top1_prof)} CUIs, score {c['top5'][0][2]:.2f}\n")
                f.write(f"- True ({c['true_name_en']}) has {len(true_prof)} CUIs, score {c['true_score']:.2f}\n")
                # Patient CUIs in top1 but not in true
                in_top1_only = (set(c['patient_cuis_in_evs']) & set(top1_prof.keys())) - set(true_prof.keys())
                in_true_only = (set(c['patient_cuis_in_evs']) & set(true_prof.keys())) - set(top1_prof.keys())
                in_both = (set(c['patient_cuis_in_evs']) & set(top1_prof.keys())) & set(true_prof.keys())
                f.write(f"- Patient CUIs in **both** profiles: {len(in_both)}\n")
                f.write(f"- Patient CUIs **only in top-1** profile: {len(in_top1_only)}\n")
                if in_top1_only:
                    for cui in sorted(in_top1_only)[:5]:
                        f.write(f"  - `{cui}` {names.get(cui,'?')} (p_top1={top1_prof.get(cui,0):.2f})\n")
                f.write(f"- Patient CUIs **only in true** profile: {len(in_true_only)}\n")
                if in_true_only:
                    for cui in sorted(in_true_only)[:5]:
                        f.write(f"  - `{cui}` {names.get(cui,'?')} (p_true={true_prof.get(cui,0):.2f})\n")

            f.write("\n---\n\n")

        f.write("## Success cases (rank=1)\n\n")
        for c in successes:
            render_case(c, "Success")
        f.write("\n## Failure cases (rank≥5)\n\n")
        for c in failures:
            render_case(c, "Failure")

        # Summary
        f.write("\n## Aggregate analysis\n\n")
        f.write(f"### Success vs Failure pattern\n\n")
        s_n_evs = [len(c["patient_cuis_in_evs"]) for c in successes]
        f_n_evs = [len(c["patient_cuis_in_evs"]) for c in failures]
        s_true_score = [c["true_score"] for c in successes]
        f_true_score = [c["true_score"] for c in failures]
        f_top1_score = [c["top5"][0][2] for c in failures]
        f_gap = [c["top5"][0][2] - c["true_score"] for c in failures]
        import statistics
        f.write(f"- Success patients: avg patient_cuis={statistics.mean(s_n_evs):.1f}, avg true_score={statistics.mean(s_true_score):.2f}\n")
        f.write(f"- Failure patients: avg patient_cuis={statistics.mean(f_n_evs):.1f}, avg true_score={statistics.mean(f_true_score):.2f}\n")
        f.write(f"- Failure score gap (top1 - true): avg={statistics.mean(f_gap):.2f}\n")

    print(f"Done → {OUT}", flush=True)


if __name__ == "__main__":
    main()
