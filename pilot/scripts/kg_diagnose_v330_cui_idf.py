#!/usr/bin/env python3
"""v330: CUI matching with IDF weighting.

Each CUI weighted by IDF (inverse disease frequency in KG):
  IDF(c) = log(N_diseases / DF(c) + 1)

Score(D) = Σ_c (c in patient ∩ KG[D]) IDF(c)

Common CUIs (cough, fever) get low weight; rare specific CUIs get high weight.
"""
from __future__ import annotations
import json, csv, ast, math, time
from collections import defaultdict, Counter

def main():
    print("Loading...")
    kg_cuis = {k: set(v) for k, v in json.load(open('/mnt/medkg/kg/disease_kg_cuis.json')).items()}
    ev_cuis = json.load(open('/mnt/medkg/kg/ddxplus_evidence_cuis.json'))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    dcs_list = sorted(set(fr2cui.values()))

    # Compute IDF using only DDXPlus 49 disease scope (consistent with eval)
    df = Counter()
    for dc in dcs_list:
        for c in kg_cuis.get(dc, set()):
            df[c] += 1
    N_d = len(dcs_list)
    idf = {c: math.log((N_d + 1) / (df[c] + 1)) + 1 for c in df}
    print(f"Diseases in scope: {N_d}, unique CUIs across DDXPlus 49: {len(df):,}")

    # Eval
    print("Evaluating 30K with IDF-weighted CUI score...")
    N = 30000; correct = 0; correct3 = 0; correct5 = 0; correct10 = 0; total = 0
    fail_per_disease = Counter(); total_per_disease = Counter()
    t0 = time.time()
    cui_results = []  # save scores for combining
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if total >= N: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            patient_cuis = set()
            for ev in evs:
                base = ev.split('_@_')[0]
                patient_cuis.update(ev_cuis.get(base, []))
            scores = []
            for dc in dcs_list:
                kg = kg_cuis.get(dc, set())
                ov = patient_cuis & kg
                # IDF-weighted score, normalized by sqrt(|D|) to discourage broad-disease bias
                raw = sum(idf.get(c, 1.0) for c in ov)
                norm = raw / max(math.sqrt(len(kg)), 1)
                scores.append((dc, norm))
            scores.sort(key=lambda x: -x[1])
            ranked = [dc for dc, s in scores]
            true_name = next((dn for dn,info in cond.items() if dn in icd_map and icd_map[dn]['cui']==true_cui), "?")
            total += 1
            total_per_disease[true_name] += 1
            if ranked[0] == true_cui: correct += 1
            else: fail_per_disease[true_name] += 1
            if true_cui in ranked[:3]: correct3 += 1
            if true_cui in ranked[:5]: correct5 += 1
            if true_cui in ranked[:10]: correct10 += 1
            cui_results.append((scores))
            if total % 5000 == 0:
                print(f"  {total}/{N}: @1={100*correct/total:.2f}% ({time.time()-t0:.0f}s)")
    print(f"\nv330 IDF-weighted CUI-match GTPA@1 = {100*correct/total:.2f}%")
    print(f"  @3 = {100*correct3/total:.2f}%  @5 = {100*correct5/total:.2f}%  @10 = {100*correct10/total:.2f}%")

    print("\nTop failures:")
    for d, fc_ in fail_per_disease.most_common(15):
        tot = total_per_disease[d]
        print(f"  {d:30s}  {fc_}/{tot} ({100*fc_/tot:.1f}%)")


if __name__ == "__main__":
    main()
