#!/usr/bin/env python3
"""Q-aware scoring with 49-disease IDF reweighting.

Phenotypes appearing in many of the 49 candidate diseases (e.g., "Coughing"
in respiratory cluster) get downweighted; disease-specific phenotypes get
upweighted. This is benchmark-AWARE in that it uses the disease list, but
not benchmark-OVERFIT since it just measures phenotype specificity within
the candidate set.

Final score(D | E):
  score = sum_{p in Q∩phens(D) ∩ E}  w(D,p) × idf49(p)
        / sqrt(sum_{p in Q∩phens(D)} w(D,p) × idf49(p))
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--hop2_decay", type=float, default=0.5)
    ap.add_argument("--idf_pow", type=float, default=1.0)
    ap.add_argument("--core_k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--use_neg_core", action="store_true", default=False)
    args = ap.parse_args()

    G = pickle.load(open(MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl", "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))
    N_D = len(dcs_list)

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if not isinstance(mapping, dict): continue
        for vkey, vcuis in mapping.items():
            if isinstance(vcuis, list): Q.update(vcuis)

    # Compute Q-phens per disease (with hierarchy)
    d_q = {}
    for d in dcs_list:
        if d not in G: d_q[d] = {}; continue
        phen_w = {}
        for _, p, edata in G.out_edges(d, data=True):
            if edata.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + edata.get("weight", 0)
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, edata2 in G.out_edges(p_direct, data=True):
                if edata2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + args.hop2_decay * dw * edata2.get("weight", 0)
        d_q[d] = {p: w for p, w in phen_w.items() if p in Q}

    # 49-disease IDF
    phen_freq = Counter()
    for d, qp in d_q.items():
        for p in qp: phen_freq[p] += 1
    idf49 = {p: math.log(N_D / max(c, 1)) ** args.idf_pow for p, c in phen_freq.items()}
    print(f"IDF stats: min={min(idf49.values()):.2f}, max={max(idf49.values()):.2f}, median={sorted(idf49.values())[len(idf49)//2]:.2f}")

    # Reweight d_q with idf
    d_q_weighted = {}
    for d, qp in d_q.items():
        d_q_weighted[d] = {p: w * idf49.get(p, 1.0) for p, w in qp.items()}

    # Core (top-K by weighted score)
    d_core = {}
    for d, qp in d_q_weighted.items():
        d_core[d] = set(sorted(qp.keys(), key=lambda p: -qp[p])[:args.core_k])

    def get_pcuis(evs):
        cuis = set()
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                m = value_cuis.get(base, {})
                cuis.update(m.get("_question", []))
                cuis.update(m.get(val, []))
            else:
                m = value_cuis.get(ev, {})
                cuis.update(m.get("_question", []))
        return cuis

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    fail_per_d = Counter(); total_per_d = Counter()
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis = get_pcuis(evs)
            scores = {}
            for d in dcs_list:
                qp = d_q_weighted.get(d, {})
                if not qp: scores[d] = -1e6; continue
                pos = sum(w for q, w in qp.items() if q in pcuis)
                if args.use_neg_core:
                    core = d_core.get(d, set())
                    neg = sum(qp.get(c, 0) for c in core if c not in pcuis)
                    s = pos - args.alpha * neg
                else:
                    s = pos
                total = sum(qp.values())
                scores[d] = s / (math.sqrt(total) or 1)
            ranked = sorted(dcs_list, key=lambda d: -scores.get(d, -1e9))
            true_name = cui2name.get(true_cui, "?")
            n += 1; total_per_d[true_name] += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            else: fail_per_d[true_name] += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"idf_pow={args.idf_pow} use_neg_core={args.use_neg_core} core_k={args.core_k} alpha={args.alpha}:")
    print(f"  @1={100*c1/n:.2f}%  @3={100*c3/n:.2f}%  @5={100*c5/n:.2f}%  @10={100*c10/n:.2f}%  MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
