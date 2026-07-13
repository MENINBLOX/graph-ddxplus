#!/usr/bin/env python3
"""Severity + SEX-aware reranking on v13.

NEW signals not yet used:
1. Pain intensity 1-10 (douleurxx_intens, lesions_peau_intens, ...)
2. Patient SEX vs disease's sex-marker phenotypes (Pregnancy/Menstrual/Prostate/Uterine)

Both are universal (KG-derived sex markers, patient row data).
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v13.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"

# Sex-specific CUIs (universal medical knowledge)
FEMALE_CUIS = {
    "C0032961",  # Pregnancy
    "C0025322",  # Menstruation
    "C0025323",  # Menorrhagia
    "C0024103",  # Lactation
    "C0033936",  # Pregnancy Complications
    "C0042232",  # Vagina
    "C0042740",  # Vulva
    "C0042267",  # Uterus
}
MALE_CUIS = {
    "C0033572",  # Prostate
    "C0033577",  # Prostatic Diseases
    "C0017428",  # Genitalia, Male
    "C0040855",  # Testis
    "C0040858",  # Testicular Diseases
    "C0030851",  # Penis
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--cov_weight", type=float, default=0.5)
    ap.add_argument("--severity_weight", type=float, default=0.0)
    ap.add_argument("--sex_penalty", type=float, default=0.0)
    ap.add_argument("--core_k", type=int, default=28)
    ap.add_argument("--alpha", type=float, default=0.2)
    args = ap.parse_args()

    G = pickle.load(open(GRAPH, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))
    dcs_set = set(dcs_list)

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)

    d_q = {}; d_all = {}; d_sex = {}
    for d in dcs_list:
        if d not in G: d_q[d] = {}; d_all[d] = {}; d_sex[d] = None; continue
        phen_w = {}
        for _, p, edata in G.out_edges(d, data=True):
            if edata.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + edata.get("weight", 0)
        all_phens = dict(phen_w)
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, edata2 in G.out_edges(p_direct, data=True):
                if edata2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + 0.5 * dw * edata2.get("weight", 0)
        d_q[d] = {p: w for p, w in phen_w.items() if p in Q}
        d_all[d] = all_phens
        # Detect sex marker
        has_female = any(p in FEMALE_CUIS for p in all_phens)
        has_male = any(p in MALE_CUIS for p in all_phens)
        if has_female and not has_male: d_sex[d] = "F"
        elif has_male and not has_female: d_sex[d] = "M"
        else: d_sex[d] = None

    n_F_disease = sum(1 for s in d_sex.values() if s == "F")
    n_M_disease = sum(1 for s in d_sex.values() if s == "M")
    print(f"Sex-specific diseases: F={n_F_disease}, M={n_M_disease}")

    phen_freq = Counter()
    for d, qp in d_q.items():
        for p in qp: phen_freq[p] += 1
    N_D = len(dcs_list)
    idf = {p: math.log(N_D / max(c, 1)) ** 0.5 for p, c in phen_freq.items()}
    d_q_idf = {d: {p: w * idf.get(p, 1.0) for p, w in qp.items()} for d, qp in d_q.items()}
    d_core = {d: set(sorted(qp.keys(), key=lambda p: -qp[p])[:args.core_k]) for d, qp in d_q_idf.items()}

    def get_pcuis_with_severity(evs):
        cuis = set()
        severity = {}  # base_ev → intensity 0-10
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                m = value_cuis.get(base, {})
                cuis.update(m.get("_question", []))
                cuis.update(m.get(val, []))
                # Capture intensity
                if "intens" in base or "soudain" in base:
                    try: severity[base] = int(val)
                    except: pass
            else:
                m = value_cuis.get(ev, {})
                cuis.update(m.get("_question", []))
        return cuis, severity

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis, severity = get_pcuis_with_severity(evs)
            patient_sex = row.get("SEX", "?")
            identity = pcuis & dcs_set

            s1_scores = {}
            for d in dcs_list:
                qp = d_q_idf.get(d, {})
                if not qp: s1_scores[d] = -1e6; continue
                pos = sum(w for q, w in qp.items() if q in pcuis)
                core = d_core[d]
                neg = sum(qp.get(c, 0) for c in core if c not in pcuis)
                total = sum(qp.values()) or 1
                s = (pos - args.alpha * neg) / math.sqrt(total)
                if d in identity: s += 1.0
                # Sex penalty
                if args.sex_penalty > 0 and d_sex.get(d):
                    if d_sex[d] != patient_sex:
                        s -= args.sex_penalty
                s1_scores[d] = s
            ranked1 = sorted(dcs_list, key=lambda d: -s1_scores.get(d, -1e9))
            top_k = ranked1[:25]

            # Stage 2 patient-coverage + severity boost
            cov_scores = {}
            severity_scores = {}
            for d in top_k:
                phens = set(d_all.get(d, {}).keys())
                match = pcuis & phens
                cov_scores[d] = len(match) / max(len(pcuis), 1)
                # Severity: high pain intensity matches diseases with severe pain markers
                if severity and args.severity_weight > 0:
                    max_intens = max(severity.values()) if severity else 0
                    # diseases with "severe" CUI (C0205082) get boost when patient has high intensity
                    SEVERE_CUI = "C0205082"
                    if max_intens >= 7 and SEVERE_CUI in phens:
                        severity_scores[d] = max_intens / 10.0
                    else:
                        severity_scores[d] = 0
                else:
                    severity_scores[d] = 0

            max_s1 = max(s1_scores[d] for d in top_k) or 1
            max_cov = max(cov_scores.values()) or 1
            max_sev = max(severity_scores.values()) or 1
            combined = {}
            for d in top_k:
                s1_n = s1_scores[d] / abs(max_s1) if max_s1 != 0 else 0
                cov_n = cov_scores[d] / max_cov if max_cov != 0 else 0
                sev_n = severity_scores[d] / max_sev if max_sev != 0 else 0
                combined[d] = (1 - args.cov_weight - args.severity_weight) * s1_n + \
                              args.cov_weight * cov_n + args.severity_weight * sev_n

            final = sorted(top_k, key=lambda d: -combined.get(d, -1e9)) + [d for d in ranked1 if d not in top_k]
            n += 1
            try: rank = final.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"cov={args.cov_weight} sev={args.severity_weight} sex_pen={args.sex_penalty}: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
