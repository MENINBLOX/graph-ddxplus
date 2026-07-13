#!/usr/bin/env python3
"""v48: Value-aware scoring — patient's numeric values as weight multipliers.

Architectural change: extract NUMERIC values from EVIDENCES (intensity, suddenness,
precision = 0-10 scale) and use them as multipliers on the corresponding signal.

Instead of: pain CUI count 1 always
New: pain CUI weighted by patient's intensity (0-10)

So a patient with intensity=10 gets 10x the pain signal vs intensity=1.
This breaks the symmetry between "mild" and "severe" patients that the current
bag-of-CUIs model collapses into the same representation.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
COMPOUND_PATH = "pilot/data/compound_pain_lookup_lt5.json"

# Numeric value evidences (0-10 scale)
NUMERIC_EVS = {
    'douleurxx_intens', 'douleurxx_soudain', 'douleurxx_precis',
    'lesions_peau_intens',
}
# CUI groups for value-weighted boost
PAIN_GENERIC_CUI = 'C0030193'  # Pain
LESION_GENERIC_CUIS = {'C0221198', 'C0037284'}


def normalize_scores(d):
    vals = list(d.values())
    if not vals: return d
    lo, hi = min(vals), max(vals)
    if hi == lo: return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--hop2_decay", type=float, default=0.7)
    ap.add_argument("--idf_pow", type=float, default=0.5)
    ap.add_argument("--core_k", type=int, default=35)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--identity_boost", type=float, default=1.5)
    ap.add_argument("--sig_k", type=int, default=10)
    ap.add_argument("--sig_w", type=float, default=9.0)
    ap.add_argument("--w_s1", type=float, default=0.7)
    ap.add_argument("--w_cov", type=float, default=0.1)
    ap.add_argument("--w_prcov", type=float, default=0.1)
    ap.add_argument("--w_compound", type=float, default=0.1)
    # NEW: value-aware channel
    ap.add_argument("--w_value", type=float, default=0.1)
    ap.add_argument("--w_intensity", type=float, default=1.0,
                    help="multiplier for intensity weighting (0=disabled)")
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    dcs_set = set(dcs_list)

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)

    PR = set(json.load(open(PR_UNIVERSE))) if Path(PR_UNIVERSE).exists() else set()

    compound = defaultdict(set)
    raw = json.load(open(COMPOUND_PATH))
    for k, v_list in raw.items():
        q, v = k.split('|')
        compound[(q, v)].update(v_list)

    d_q = {}
    for d in dcs_list:
        if d not in G: d_q[d] = {}; continue
        phen_w = {}
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + ed.get("weight", 0)
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, ed2 in G.out_edges(p_direct, data=True):
                if ed2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + args.hop2_decay * dw * ed2.get("weight", 0)
        d_q[d] = {p: w for p, w in phen_w.items() if p in Q}

    phen_freq = Counter()
    for d, qp in d_q.items():
        for p in qp: phen_freq[p] += 1
    N_D = len(dcs_list)
    idf = {p: math.log(N_D / max(c, 1)) ** args.idf_pow for p, c in phen_freq.items()}
    d_q_idf = {d: {p: w * idf.get(p, 1.0) for p, w in qp.items()} for d, qp in d_q.items()}
    d_core = {d: set(sorted(qp.keys(), key=lambda p: -qp[p])[:args.core_k]) for d, qp in d_q_idf.items()}
    d_sig = {d: set(sorted(qp.keys(), key=lambda p: -qp[p])[:args.sig_k]) for d, qp in d_q_idf.items()}

    disease_full_phens = {d: {p for _, p, ed in G.out_edges(d, data=True) if ed.get("etype")=="HAS_PHENOTYPE"} if d in G else set() for d in dcs_list}
    compound_cuis_all = set()
    for cuis in compound.values(): compound_cuis_all.update(cuis)
    compound_doc_freq = {c: sum(1 for p in disease_full_phens.values() if c in p) for c in compound_cuis_all}
    compound_idf = {c: math.log(49 / max(compound_doc_freq.get(c, 1), 1)) for c in compound_cuis_all}

    # KEY NEW: identify diseases with specific severe-pain / sudden-onset characteristics
    # These get boosted by intensity / suddenness values
    SEVERE_PAIN_CUIS = {'C0238995', 'C0008033', 'C0234254', 'C0151826'}  # Sharp/pleuritic/radiating/retrosternal
    SUDDEN_ONSET_CUIS = {'C0026703'}  # Sudden onset
    # Disease score for severity dimension
    disease_severity_score = {}
    for d in dcs_list:
        full = disease_full_phens[d]
        sev = sum(1 for c in SEVERE_PAIN_CUIS if c in full)
        disease_severity_score[d] = sev

    def get_pcuis_and_structured(evs):
        cuis = set()
        compound_targets = set()
        # Numeric values
        intensity = None
        suddenness = None
        precision = None
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                m = value_cuis.get(base, {})
                q_cuis = m.get("_question", [])
                v_cuis = m.get(val, [])
                for q in q_cuis:
                    for v in v_cuis:
                        if (q, v) in compound: compound_targets.update(compound[(q, v)])
                cuis.update(q_cuis); cuis.update(v_cuis)
                # Extract numeric values
                if base == 'douleurxx_intens':
                    try: intensity = int(val)
                    except: pass
                elif base == 'douleurxx_soudain':
                    try: suddenness = int(val)
                    except: pass
                elif base == 'douleurxx_precis':
                    try: precision = int(val)
                    except: pass
            else:
                m = value_cuis.get(ev, {})
                cuis.update(m.get("_question", []))
        return cuis, compound_targets, intensity, suddenness, precision

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis, compound_targets, intensity, suddenness, precision = get_pcuis_and_structured(evs)
            identity_diseases = pcuis & dcs_set

            s1_scores = {}; cov_scores = {}; prcov_scores = {}; comp_scores = {}; value_scores = {}
            for d in dcs_list:
                qp = d_q_idf.get(d, {})
                pos = sum(w for q, w in qp.items() if q in pcuis)
                core = d_core.get(d, set())
                neg = sum(qp.get(c, 0) for c in core if c not in pcuis)
                s1 = pos - args.alpha * neg
                total = sum(qp.values()) if qp else 1
                s1 = s1 / (math.sqrt(total) or 1)
                sig = d_sig.get(d, set())
                if sig:
                    s1 += args.sig_w * (sum(1 for p in sig if p in pcuis) / len(sig))
                if d in identity_diseases:
                    s1 += args.identity_boost
                s1_scores[d] = s1

                cov_scores[d] = sum(1 for p in pcuis if p in qp) / max(len(pcuis), 1) if pcuis and qp else 0
                if PR and pcuis and qp:
                    pr_pcuis = pcuis & PR
                    pr_qp = {p: w for p, w in qp.items() if p in PR}
                    prcov_scores[d] = sum(1 for p in pr_pcuis if p in pr_qp) / max(len(pr_pcuis), 1) if (pr_pcuis and pr_qp) else 0
                else:
                    prcov_scores[d] = 0

                comp = 0
                if compound_targets and disease_full_phens[d]:
                    comp = sum(compound_idf.get(c, 0) for c in (compound_targets & disease_full_phens[d]))
                comp_scores[d] = comp

                # VALUE-AWARE: scale severity by intensity*suddenness
                v_score = 0
                if intensity is not None and PAIN_GENERIC_CUI in qp:
                    # Severity match: high-intensity patient × disease with severe-pain CUIs
                    severity_match = disease_severity_score[d] * (intensity / 10.0)
                    v_score += args.w_intensity * severity_match
                    if suddenness is not None:
                        # Sudden-onset boost for diseases with "acute" in name
                        v_score += args.w_intensity * (intensity * suddenness / 100.0) * disease_severity_score[d]
                value_scores[d] = v_score

            s1_n = normalize_scores(s1_scores)
            cov_n = normalize_scores(cov_scores)
            prcov_n = normalize_scores(prcov_scores)
            comp_n = normalize_scores(comp_scores)
            value_n = normalize_scores(value_scores)

            final = {d: args.w_s1*s1_n[d] + args.w_cov*cov_n[d] + args.w_prcov*prcov_n[d] + args.w_compound*comp_n[d] + args.w_value*value_n[d] for d in dcs_list}
            ranked = sorted(dcs_list, key=lambda d: -final.get(d, -1e9))
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"v48 [w_value={args.w_value},w_int={args.w_intensity}]: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
