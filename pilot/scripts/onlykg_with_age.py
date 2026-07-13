#!/usr/bin/env python3
"""Best Q-aware + identity + age preference boost."""
from __future__ import annotations
import sys, json, csv, ast, math, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
AGE_PREF = MEDKG_ROOT / "kg" / "disease_age_preferences.json"


def age_to_category(age):
    if age is None or age == "": return "adult"
    try: age = int(age)
    except: return "adult"
    if age < 3: return "infant"
    if age < 18: return "child"
    if age < 65: return "adult"
    return "elderly"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--age_boost", type=float, default=1.0)
    ap.add_argument("--identity_boost", type=float, default=1.5)
    ap.add_argument("--age_min_count", type=int, default=2, help="ignore preference if total raw counts < this")
    args = ap.parse_args()

    G = pickle.load(open(GRAPH, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))
    age_data = json.load(open(AGE_PREF))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))
    dcs_set = set(dcs_list)

    # Filter age preferences: only trust if raw counts >= threshold
    age_pref = {}
    for cui, info in age_data.items():
        counts = info.get("raw_counts", {})
        total_count = sum(counts.values())
        if total_count >= args.age_min_count:
            age_pref[cui] = info["age_pref"]

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)

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
                    phen_w[p2] = phen_w.get(p2, 0) + 0.5 * dw * edata2.get("weight", 0)
        d_q[d] = {p: w for p, w in phen_w.items() if p in Q}

    phen_freq = Counter()
    for d, qp in d_q.items():
        for p in qp: phen_freq[p] += 1
    N_D = len(dcs_list)
    idf = {p: math.log(N_D / max(c, 1)) ** 0.5 for p, c in phen_freq.items()}
    d_q_idf = {d: {p: w * idf.get(p, 1.0) for p, w in qp.items()} for d, qp in d_q.items()}

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
            identity_diseases = pcuis & dcs_set
            patient_age_cat = age_to_category(row.get("AGE"))

            scores = {}
            for d in dcs_list:
                qp = d_q_idf.get(d, {})
                if not qp: scores[d] = -1e6; continue
                pos = sum(w for q, w in qp.items() if q in pcuis)
                total = sum(qp.values()) or 1
                s = pos / math.sqrt(total)
                if d in identity_diseases:
                    s += args.identity_boost
                # Age boost
                if d in age_pref:
                    pref = age_pref[d].get(patient_age_cat, 0.25)
                    s += args.age_boost * pref
                scores[d] = s
            ranked = sorted(dcs_list, key=lambda d: -scores.get(d, -1e9))
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"age_boost={args.age_boost} identity={args.identity_boost} age_min_count={args.age_min_count}:")
    print(f"  @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
