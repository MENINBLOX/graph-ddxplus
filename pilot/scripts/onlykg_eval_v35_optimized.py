#!/usr/bin/env python3
"""v35: evaluation 4-axis optimization on top of v28 KG.

(a)+(b) Compositional matching: (question_CUI, value_CUI) pair → compound CUI lookup
(c) Fill empty value mappings: French body locations / pain characters → CUIs
(d) Modifier filter: exclude noise CUIs (non-medical TUIs) from patient evidence

Each is a flag; can be combined.
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
    ap.add_argument("--pr_universe", default="pilot/data/pr_universe.json")
    # Optimizations
    ap.add_argument("--opt_a_compound", action="store_true", help="(a) compound CUI from (q_cui, v_cui) pair")
    ap.add_argument("--compound_path", default="pilot/data/compound_pain_lookup.json")
    ap.add_argument("--opt_c_fill_char", action="store_true", help="(c) fill empty pain character/color CUIs")
    ap.add_argument("--opt_c_fill_loc", action="store_true", help="(c) fill empty body location CUIs")
    ap.add_argument("--opt_d_noise", action="store_true", help="(d) filter noise/modifier CUIs")
    args = ap.parse_args()

    print(f"Loading {args.graph}", flush=True)
    G = pickle.load(open(args.graph, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    cui2name = {icd[dn]["cui"]: dn for dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    dcs_set = set(dcs_list)

    Q_all = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q_all.update(v)

    # Noise filter (d)
    noise_cuis = set()
    if args.opt_d_noise:
        noise_cuis = set(json.load(open("pilot/data/q_noise_cuis.json")))
        print(f"  (d) noise CUIs to filter: {len(noise_cuis)}", flush=True)
    Q = Q_all - noise_cuis

    # Compound CUI lookup (a)+(b)
    compound = {}
    if args.opt_a_compound:
        raw = json.load(open(args.compound_path))
        for k, v_list in raw.items():
            q, v = k.split('|')
            compound.setdefault((q, v), set()).update(v_list)
        print(f"  (a) compound lookups: {len(compound)}", flush=True)
        # Add compound CUIs to Q if not already
        for cuis in compound.values():
            Q.update(cuis)
        print(f"  Q after compound expansion: {len(Q)}", flush=True)

    # Fill empty values (c)
    fill_char = {}
    fill_loc = {}
    fill_evval = {}  # (ev, val) → cui (specific to evidence-value pair)
    if args.opt_c_fill_char:
        raw = json.load(open("pilot/data/fr_value_cui_fill.json"))
        for k, v_list in raw.items():
            fill_char[k] = v_list
        print(f"  (c) fill char/color: {len(fill_char)}", flush=True)
    if args.opt_c_fill_loc:
        raw = json.load(open("pilot/data/fr_body_location_cui.json"))
        for k, v in raw.items():
            fill_loc[k] = v
        # Also load comprehensive evidence-value fill
        if Path("pilot/data/fr_evidence_value_fill.json").exists():
            ev_raw = json.load(open("pilot/data/fr_evidence_value_fill.json"))
            for k, v in ev_raw.items():
                ev_name, val = k.split('|', 1)
                fill_evval[(ev_name, val)] = v
        print(f"  (c) fill body location: {len(fill_loc)}, ev-val: {len(fill_evval)}", flush=True)

    PR = set()
    if Path(args.pr_universe).exists():
        PR = set(json.load(open(args.pr_universe)))

    # Build d_q_idf with Q (post-noise-filter, post-compound-expand)
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

    def get_pcuis(evs):
        cuis = set()
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                m = value_cuis.get(base, {})
                q_cuis = m.get("_question", [])
                v_cuis = m.get(val, [])
                # (c) fill empty
                if not v_cuis:
                    if args.opt_c_fill_loc and (base, val) in fill_evval:
                        v_cuis = [fill_evval[(base, val)]]
                    elif args.opt_c_fill_loc and val in fill_loc:
                        v_cuis = [fill_loc[val]]
                    elif args.opt_c_fill_char and val in fill_char:
                        v_cuis = fill_char[val]
                # (a) compound matching: (q_cui, v_cui) → compound CUI
                if args.opt_a_compound:
                    for q in q_cuis:
                        for v in v_cuis:
                            comp = compound.get((q, v))
                            if comp:
                                cuis.update(comp)
                cuis.update(q_cuis)
                cuis.update(v_cuis)
            else:
                m = value_cuis.get(ev, {})
                cuis.update(m.get("_question", []))
        # (d) filter noise
        if args.opt_d_noise:
            cuis -= noise_cuis
        return cuis

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis = get_pcuis(evs)
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
                scores[d] = (s1, cov, prcov)

            final = {d: args.w_s1 * scores[d][0] + args.w_cov * scores[d][1] + args.w_prcov * scores[d][2] for d in dcs_list}
            ranked = sorted(dcs_list, key=lambda d: -final.get(d, -1e9))
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    flags = []
    if args.opt_a_compound: flags.append("a-compound")
    if args.opt_c_fill_char: flags.append("c-char")
    if args.opt_c_fill_loc: flags.append("c-loc")
    if args.opt_d_noise: flags.append("d-noise")
    flag_str = ",".join(flags) if flags else "none"
    print(f"\n[{flag_str}] ({n} patients): @1={100*c1/n:.2f}%  @3={100*c3/n:.2f}%  @5={100*c5/n:.2f}%  @10={100*c10/n:.2f}%  MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
