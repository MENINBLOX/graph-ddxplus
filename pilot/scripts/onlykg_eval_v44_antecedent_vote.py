#!/usr/bin/env python3
"""v44: Antecedent-evidence direct disease vote.

Architecture insight: DDXPlus has 113 antecedent (medical history) evidences.
Many are highly disease-specific (e.g., 'ap_pneumothorax', 'atcd_cluster').
Current model treats them as ordinary CUIs in pcuis bag.

New: For each patient antecedent evidence A, compute a DIRECT disease vote:
- For each disease D, score how strongly D is associated with A's CUIs in KG
- If patient has A, this adds disease-specific vote

This complements (not replaces) the standard scoring. We use it as a 5th channel.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
EVIDENCES_DEF = "data/ddxplus/release_evidences.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"


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
    # New channel: antecedent vote
    ap.add_argument("--w_antecedent", type=float, default=0.5)
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))
    ev_def = json.load(open(EVIDENCES_DEF))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    cui2name = {icd[dn]["cui"]: dn for dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    dcs_set = set(dcs_list)

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)

    PR = set(json.load(open(PR_UNIVERSE))) if Path(PR_UNIVERSE).exists() else set()

    # Antecedent evidence names (113)
    antecedent_evs = {k for k, v in ev_def.items() if v.get('is_antecedent')}
    # CUIs for each antecedent
    antecedent_cuis = {}
    for ev_name in antecedent_evs:
        m = value_cuis.get(ev_name, {})
        if isinstance(m, dict):
            cuis = set()
            for v in m.values():
                if isinstance(v, list): cuis.update(v)
            if cuis:
                antecedent_cuis[ev_name] = cuis
    print(f"Antecedents with CUIs: {len(antecedent_cuis)}", flush=True)

    # d_q with hop2
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

    # Full disease phens (not Q-restricted) for antecedent matching
    d_full = {}
    for d in dcs_list:
        if d not in G: d_full[d] = {}; continue
        phen_w = {}
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + ed.get("weight", 0)
        d_full[d] = phen_w

    # Pre-compute disease scores for each antecedent (IDF-weighted)
    # P(A|D) proxy = sum of weights of A's CUIs in D's KG
    antecedent_idf = {}
    for ev, ev_cuis in antecedent_cuis.items():
        # Count: how many diseases have any of A's CUIs
        n_pos = sum(1 for d in dcs_list if any(c in d_full.get(d, {}) for c in ev_cuis))
        antecedent_idf[ev] = math.log(len(dcs_list) / max(n_pos, 1))

    # disease_ant_score[d][ev] = log-weighted match score
    disease_ant_score = defaultdict(dict)
    for ev, ev_cuis in antecedent_cuis.items():
        for d in dcs_list:
            qp = d_full.get(d, {})
            score = max((qp.get(c, 0) for c in ev_cuis), default=0)
            disease_ant_score[d][ev] = score * antecedent_idf[ev]

    # IDF for d_q
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
        ev_bases = set()
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                m = value_cuis.get(base, {})
                cuis.update(m.get("_question", []))
                cuis.update(m.get(val, []))
                ev_bases.add(base)
            else:
                m = value_cuis.get(ev, {})
                cuis.update(m.get("_question", []))
                ev_bases.add(ev)
        return cuis, ev_bases

    def normalize(d):
        vals = list(d.values())
        if not vals: return d
        lo, hi = min(vals), max(vals)
        if hi == lo: return {k: 0.5 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis, ev_bases = get_pcuis(evs)
            identity_diseases = pcuis & dcs_set

            # Patient antecedent set
            patient_antecedents = ev_bases & antecedent_cuis.keys()

            s1_scores = {}; cov_scores = {}; prcov_scores = {}; ant_scores = {}
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

                if pcuis and qp:
                    cov_scores[d] = sum(1 for p in pcuis if p in qp) / len(pcuis)
                else:
                    cov_scores[d] = 0
                if PR and pcuis and qp:
                    pr_pcuis = pcuis & PR
                    pr_qp = {p: w for p, w in qp.items() if p in PR}
                    prcov_scores[d] = sum(1 for p in pr_pcuis if p in pr_qp) / max(len(pr_pcuis), 1) if (pr_pcuis and pr_qp) else 0
                else:
                    prcov_scores[d] = 0

                # Antecedent vote: sum of disease's KG weights for each patient antecedent
                ant_score = sum(disease_ant_score[d].get(ev, 0) for ev in patient_antecedents)
                ant_scores[d] = ant_score

            # Normalize each channel to [0,1] then weighted sum
            s1_n = normalize(s1_scores)
            cov_n = normalize(cov_scores)
            prcov_n = normalize(prcov_scores)
            ant_n = normalize(ant_scores)

            final = {d: args.w_s1*s1_n[d] + args.w_cov*cov_n[d] + args.w_prcov*prcov_n[d] + args.w_antecedent*ant_n[d] for d in dcs_list}
            ranked = sorted(dcs_list, key=lambda d: -final.get(d, -1e9))
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"v44 [w_s1={args.w_s1},w_cov={args.w_cov},w_prcov={args.w_prcov},w_ant={args.w_antecedent}]: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
