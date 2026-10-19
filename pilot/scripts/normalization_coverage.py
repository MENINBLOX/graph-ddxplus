"""Measure ACTUAL normalization coverage per benchmark (is normalization really a problem?).

Two metrics, at the concept-vocabulary level and the patient level:
  (1) normalization rate = raw evidence concept -> UMLS CUI success %  (via the provided maps)
  (2) KG-match rate      = normalized CUI that also exists in our KG phenotype vocabulary %
A concept only helps diagnosis if BOTH hold. Reports distinct-concept and per-patient views.
"""
from __future__ import annotations
import json, csv, ast, re, random, statistics as st
from collections import Counter

KG = "pilot/cache/prod/kg_merged2.json"


def kg_vocab():
    kg = json.load(open(KG))
    return set(kg["phenotypes"]) | set(kg["diseases"])   # unified evidence: disease CUIs match too


def report(name, distinct_raw, resolve_fn, patients_raw, KGV):
    """distinct_raw: set of raw concepts; resolve_fn: raw->cui or None;
       patients_raw: list of lists of raw concepts."""
    res = {r: resolve_fn(r) for r in distinct_raw}
    n_res = sum(1 for v in res.values() if v)
    n_kg = sum(1 for v in res.values() if v and v in KGV)
    # patient-level: fraction of a patient's raw evidence that is resolved AND in-KG
    pr_res, pr_kg = [], []
    for evs in patients_raw:
        if not evs:
            continue
        cuis = [res.get(e) for e in evs]
        pr_res.append(sum(1 for c in cuis if c) / len(evs))
        pr_kg.append(sum(1 for c in cuis if c and c in KGV) / len(evs))
    print(f"\n{name}")
    print(f"  distinct raw concepts : {len(distinct_raw)}")
    print(f"  -> normalized to CUI  : {n_res} ({100*n_res/len(distinct_raw):.1f}%)")
    print(f"  -> CUI also in our KG : {n_kg} ({100*n_kg/len(distinct_raw):.1f}%)")
    print(f"  patient-level (mean): evidence normalized {100*st.mean(pr_res):.1f}% | "
          f"normalized & in-KG {100*st.mean(pr_kg):.1f}%  (n_pat={len(pr_res)})")


def main():
    KGV = kg_vocab()
    print(f"KG vocab (phenotypes+diseases): {len(KGV)} CUIs")

    # ---- DDXPlus: evidence codes -> CUI via umls_mapping.json --------------
    um = json.load(open("data/ddxplus/umls_mapping.json"))["mapping"]
    def ddx_resolve(e):
        for c in (e, e.split("_@_")[0], e.split("_")[0]):
            if c in um:
                return um[c]["cui"]
        return None
    draw, dpats = set(), []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= 30000:
                break
            evs = ast.literal_eval(row["EVIDENCES"])
            dpats.append(evs)
            draw.update(evs)
    report("DDXPlus (evidence codes -> umls_mapping.json)", draw, ddx_resolve, dpats, KGV)

    # ---- SymCat: symptom phrase -> CUI via symptom_umls_mapping.json -------
    sm = json.load(open("data/symcat/symptom_umls_mapping.json"))["mapping"]
    name2cui = {nm: i.get("umls_cui") for nm, i in sm.items()}
    sc = json.load(open("data/symcat/symcat_parsed.json"))
    slug2name = {i["symcat_slug"]: nm for nm, i in sm.items() if i.get("symcat_slug")}
    sraw, spats = set(), []
    rng = random.Random(42)
    for dn, syms in sc["disease_symptom_pairs"].items():
        names = [s for s, f in syms]         # s here is already the display name key
        sraw.update(names)
        for _ in range(20):
            spats.append([s for s, fr in syms if rng.random() * 100 < fr])
    report("SymCat (symptom phrase -> symptom_umls_mapping.json)", sraw,
           lambda s: name2cui.get(s), spats, KGV)

    # ---- RareBench: HPO code -> CUI via hpo_umls_mapping.json --------------
    hm = json.load(open("data/rarebench/hpo_umls_mapping.json"))["mapping"]
    def hpo_resolve(h):
        info = hm.get(h)
        return info.get("umls_cui") if isinstance(info, dict) else info if isinstance(info, str) else None
    for dsn in ["RAMEDIS", "LIRICAL", "MME", "HMS"]:
        hraw, hpats = set(), []
        for line in open(f"data/rarebench/data/{dsn}.jsonl"):
            r = json.loads(line)
            ph = r["Phenotype"]
            hpats.append(ph); hraw.update(ph)
        report(f"RareBench:{dsn} (HPO -> hpo_umls_mapping.json)", hraw, hpo_resolve, hpats, KGV)


if __name__ == "__main__":
    main()
