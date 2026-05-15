#!/usr/bin/env python3
"""v36: compound matching as separate Stage 2 channel (not pcuis injection).

For each (q_cui, v_cui) pair in patient evidence, lookup compound CUI(s).
For each disease, count how many compound CUIs are in disease's KG phens.
This is the "compound match score" — added as 4th channel.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"


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
    ap.add_argument("--w_s1", type=float, default=0.4)
    ap.add_argument("--w_cov", type=float, default=0.3)
    ap.add_argument("--w_prcov", type=float, default=0.3)
    ap.add_argument("--w_compound", type=float, default=0.0, help="compound match channel weight (4th)")
    ap.add_argument("--pr_universe", default="pilot/data/pr_universe.json")
    ap.add_argument("--compound_path", default="pilot/data/compound_pain_lookup_lt5.json")
    ap.add_argument("--opt_c_fill_loc", action="store_true")
    ap.add_argument("--opt_d_noise", action="store_true")
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

    compound = {}
    raw = json.load(open(args.compound_path))
    for k, v_list in raw.items():
        q, v = k.split('|')
        compound.setdefault((q, v), set()).update(v_list)

    PR = set(json.load(open(args.pr_universe))) if Path(args.pr_universe).exists() else set()

    # (c) loc fill + (d) noise
    fill_evval = {}
    fill_loc = {}
    noise_cuis = set()
    if args.opt_c_fill_loc:
        if Path("pilot/data/fr_evidence_value_fill.json").exists():
            for k, v in json.load(open("pilot/data/fr_evidence_value_fill.json")).items():
                ev_name, val = k.split('|', 1)
                fill_evval[(ev_name, val)] = v
        if Path("pilot/data/fr_body_location_cui.json").exists():
            fill_loc = json.load(open("pilot/data/fr_body_location_cui.json"))
    if args.opt_d_noise and Path("pilot/data/q_noise_cuis.json").exists():
        noise_cuis = set(json.load(open("pilot/data/q_noise_cuis.json")))

    # disease KG phens (full, not Q-restricted) for compound matching
    disease_full_phens = {}
    for d in dcs_list:
        phens = set()
        if d in G:
            for _, p, ed in G.out_edges(d, data=True):
                if ed.get("etype") == "HAS_PHENOTYPE":
                    phens.add(p)
        disease_full_phens[d] = phens

    # Compute IDF for compound CUIs (across 49 disease)
    compound_cuis_all = set()
    for cuis in compound.values(): compound_cuis_all.update(cuis)
    compound_doc_freq = {c: sum(1 for p in disease_full_phens.values() if c in p) for c in compound_cuis_all}
    compound_idf = {c: math.log(49 / max(compound_doc_freq.get(c, 1), 1)) for c in compound_cuis_all}

    # d_q_idf (standard)
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

    def get_pcuis_and_compound(evs):
        cuis = set()
        compound_targets = set()
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                m = value_cuis.get(base, {})
                q_cuis = m.get("_question", [])
                v_cuis = m.get(val, [])
                # (c) fill empty
                if not v_cuis and args.opt_c_fill_loc:
                    if (base, val) in fill_evval:
                        v_cuis = [fill_evval[(base, val)]]
                    elif val in fill_loc:
                        v_cuis = [fill_loc[val]]
                # Compound lookup
                for q in q_cuis:
                    for v in v_cuis:
                        if (q, v) in compound:
                            compound_targets.update(compound[(q, v)])
                cuis.update(q_cuis)
                cuis.update(v_cuis)
            else:
                m = value_cuis.get(ev, {})
                cuis.update(m.get("_question", []))
        if args.opt_d_noise:
            cuis -= noise_cuis
        return cuis, compound_targets

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis, compound_targets = get_pcuis_and_compound(evs)
            identity_diseases = pcuis & dcs_set

            scores = {}
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

                if pcuis and qp:
                    cov = sum(1 for p in pcuis if p in qp) / len(pcuis)
                else:
                    cov = 0
                if PR and pcuis and qp:
                    pr_pcuis = pcuis & PR
                    pr_qp = {p: w for p, w in qp.items() if p in PR}
                    prcov = sum(1 for p in pr_pcuis if p in pr_qp) / max(len(pr_pcuis), 1) if (pr_pcuis and pr_qp) else 0
                else:
                    prcov = 0

                # Compound channel: IDF-weighted compound matches
                compound_score = 0.0
                if compound_targets and disease_full_phens[d]:
                    matched = compound_targets & disease_full_phens[d]
                    compound_score = sum(compound_idf.get(c, 0) for c in matched)

                scores[d] = (s1, cov, prcov, compound_score)

            final = {d: args.w_s1 * scores[d][0] + args.w_cov * scores[d][1] + args.w_prcov * scores[d][2] + args.w_compound * scores[d][3] for d in dcs_list}
            ranked = sorted(dcs_list, key=lambda d: -final.get(d, -1e9))
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"  w_compound={args.w_compound}: @1={100*c1/n:.2f}%  @3={100*c3/n:.2f}%  @5={100*c5/n:.2f}%  @10={100*c10/n:.2f}%  MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
