"""Crack open individual NLICE eval failures to locate EVAL-PIPELINE flaws
(not IE quality). For failing patients, trace CUI-level why the true disease loses:
does the patient's symptom concept exist in the true disease under a DIFFERENT CUI
(synonym/granularity mismatch)? Does the scorer's normalization sink it?
"""
from __future__ import annotations
import json, math, sys
from collections import defaultdict

sys.path.insert(0, "pilot/scripts")

kg = json.load(open("pilot/cache/prod/kg.json"))
Q = json.load(open("pilot/cache/prod/nlice_queries.json"))
PH = kg["phenotypes"]
DZ = kg["diseases"]

N = len(DZ)
df = defaultdict(int)
prof = defaultdict(dict)
for e in kg["edges"]:
    df[e["p"]] += 1
for e in kg["edges"]:
    prof[e["d"]][e["p"]] = PH.get(e["p"], {}).get("idf", 0.0)

cand = sorted(set(Q["meta"]["diseases"].values()) & set(prof.keys()))


def nm(c):
    return PH.get(c, {}).get("name") or DZ.get(c, {}).get("name") or c


def score(findings):
    sc = {}
    for d in cand:
        pf = prof.get(d, {})
        s = sum(pf.get(f["cui"], 0.0) for f in findings)
        sc[d] = s / max(math.sqrt(len(pf)) if pf else 1.0, 1.0)
    return sc


# group patients by disease, take a few failing ones
by_dz = defaultdict(list)
for p in Q["patients"]:
    by_dz[p["true_cui"]].append(p)

shown = 0
for tcui, pats in by_dz.items():
    if tcui not in cand:
        continue
    for p in pats:
        sc = score(p["findings"])
        ranked = sorted(cand, key=lambda d: -sc[d])
        rank = ranked.index(tcui) + 1
        if rank == 1:
            continue
        # a failure -- dissect it
        print("=" * 90)
        print(f"DISEASE {nm(tcui)} ({tcui})  true-rank={rank}/{len(cand)}  "
              f"score={sc[tcui]:.3f}  top={nm(ranked[0])} ({sc[ranked[0]]:.3f})")
        print(f"  true-disease KG profile size = {len(prof[tcui])} phenotypes")
        pcuis = [f["cui"] for f in p["findings"]]
        print(f"  patient findings ({len(pcuis)}):")
        for f in p["findings"]:
            c = f["cui"]
            in_true = c in prof[tcui]
            in_kg = c in PH
            print(f"    {c}  {nm(c):40.40}  in_true_profile={in_true}  in_KG={in_kg}")
        # for each patient CUI NOT in true profile, is a same-NAME concept in true profile
        # under a different CUI?
        true_names = {PH.get(c, {}).get("name", "").lower(): c for c in prof[tcui]}
        print("  --- concept-level miss analysis (patient CUI absent from true profile) ---")
        for f in p["findings"]:
            c = f["cui"]
            if c in prof[tcui]:
                continue
            cn = nm(c).lower()
            # any true-profile phenotype whose name shares a content word?
            hits = [(cc, PH.get(cc, {}).get("name")) for nmn, cc in true_names.items()
                    if nmn and (cn in nmn or nmn in cn or
                                (set(cn.split()) & set(nmn.split()) - {"of", "the", "a"}))]
            tag = "SYNONYM-MISS" if hits else "genuinely-absent"
            print(f"    {c} '{cn}'  -> {tag}  {hits[:3]}")
        shown += 1
        if shown >= 3:
            break
    if shown >= 3:
        break
