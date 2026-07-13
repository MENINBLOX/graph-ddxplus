#!/usr/bin/env python3
"""Q-aware scoring with different normalization variants."""
from __future__ import annotations
import sys, json, csv, ast, math, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--hop2_decay", type=float, default=0.5)
    ap.add_argument("--norm", default="sqrt",
                    help="sqrt|none|log|cosine|max|patient_size")
    args = ap.parse_args()

    G = pickle.load(open(MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl", "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if not isinstance(mapping, dict): continue
        for vkey, vcuis in mapping.items():
            if isinstance(vcuis, list): Q.update(vcuis)

    # Extended phens with hierarchy
    d_q_phens = {}
    for d in dcs_list:
        if d not in G: d_q_phens[d] = {}; continue
        phen_w = {}
        for _, p, edata in G.out_edges(d, data=True):
            if edata.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + edata.get("weight", 0)
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, edata2 in G.out_edges(p_direct, data=True):
                if edata2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + args.hop2_decay * dw * edata2.get("weight", 0)
        d_q_phens[d] = {p: w for p, w in phen_w.items() if p in Q}

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
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis = get_pcuis(evs)
            scores = {}
            for d in dcs_list:
                qp = d_q_phens.get(d, {})
                if not qp: scores[d] = -1e6; continue
                pos = sum(w for q, w in qp.items() if q in pcuis)
                total = sum(qp.values())
                if args.norm == "sqrt": denom = math.sqrt(total) or 1
                elif args.norm == "none": denom = 1
                elif args.norm == "log": denom = math.log1p(total) or 1
                elif args.norm == "cosine":
                    # standard cosine: pos / (sqrt(disease_norm) × sqrt(patient_norm_in_Q))
                    patient_in_q_w = len(pcuis & Q) or 1
                    denom = math.sqrt(total * patient_in_q_w) or 1
                elif args.norm == "max":
                    denom = max(qp.values()) or 1
                elif args.norm == "patient_size":
                    n_matched = sum(1 for q in qp if q in pcuis)
                    denom = max(n_matched, 1) ** 0.5
                else:
                    denom = 1
                scores[d] = pos / denom
            ranked = sorted(dcs_list, key=lambda d: -scores.get(d, -1e9))
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"norm={args.norm} hop2={args.hop2_decay}:  @1={100*c1/n:.2f}%  @3={100*c3/n:.2f}%  @5={100*c5/n:.2f}%  @10={100*c10/n:.2f}%  MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
