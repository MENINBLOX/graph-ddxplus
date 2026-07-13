#!/usr/bin/env python3
"""v103 full evaluation — CUI-grounded attribute-aware diagnosis.

Patient evidence (CUI + attributes) ↔ Disease profile (CUI edges + attribute distributions).
CUI-to-CUI exact match (base) + attribute alignment (refinement) + TF-IDF weighting.

학술적: bipartite KG, single algorithm, GTPA@1.
"""
from __future__ import annotations
import sys, json, math, pickle, argparse
from pathlib import Path
from collections import defaultdict


def compute_phen_idf(G):
    n_dis = sum(1 for n,d in G.nodes(data=True) if d.get("ntype")=="disease")
    df = defaultdict(int)
    for u, v, ed in G.edges(data=True):
        if ed.get("etype")=="HAS_PHENOTYPE": df[v] += 1
    idf = {p: math.log((n_dis+1)/(df_p+1))+1 for p, df_p in df.items()}
    return idf, n_dis


# anatomy adjacency for partial credit
NEARBY = [
    {"face","cheek","lip","tongue","mouth","throat","larynx","head","eye","eyelid","ear","nose","neck"},
    {"leg","thigh","knee","ankle","foot"},
    {"arm","shoulder","elbow","wrist","hand","finger"},
    {"chest","lung","heart","back"},
    {"abdomen","epigastric","liver","kidney","pelvis","groin"},
    {"skin","generalized","systemic"},
]


def loc_align(pat_locs, prof_dist):
    if not pat_locs or not prof_dist: return 0.5
    ps = set(pat_locs); prs = set(prof_dist)
    if ps & prs:
        return min(1.0, sum(prof_dist.get(l,0) for l in ps) + 0.3)  # base credit + mass
    for g in NEARBY:
        if (ps & g) and (prs & g): return 0.4
    return 0.0


def attr_align(pat_attrs, ed):
    scores = []
    if pat_attrs.get("location") or ed.get("location_dist"):
        scores.append(loc_align(pat_attrs.get("location",[]), ed.get("location_dist",{})))
    pat_sev = pat_attrs.get("severity")
    if pat_sev and ed.get("severity_dist"):
        sd = ed["severity_dist"]
        adj = {"mild":["moderate"],"moderate":["mild","severe"],"severe":["moderate","critical","profound"],"critical":["severe"],"profound":["severe"]}
        sc = sd.get(pat_sev,0) + 0.4*sum(sd.get(a,0) for a in adj.get(pat_sev,[]))
        scores.append(min(1.0, sc))
    pat_ch = pat_attrs.get("character",[])
    if pat_ch and ed.get("character_dist"):
        cd = ed["character_dist"]
        scores.append(min(1.0, sum(cd.get(c,0) for c in pat_ch)))
    return sum(scores)/len(scores) if scores else 0.5


def score_disease(patient_ev, dcui, G, idf, alpha):
    """Σ over patient CUI evidence: (if CUI in disease profile)
        (alpha + (1-alpha)*attr_align) * log(1+n_mentions) * idf"""
    if dcui not in G: return 0
    # Build disease's phenotype edge map: pcui → best edge
    prof = {}
    for _, pcui, ed in G.out_edges(dcui, data=True):
        if ed.get("etype")!="HAS_PHENOTYPE": continue
        if pcui not in prof or ed["n_mentions"] > prof[pcui]["n_mentions"]:
            prof[pcui] = ed
    total = 0
    for ev in patient_ev:
        pcui = ev["cui"]
        if pcui not in prof: continue
        ed = prof[pcui]
        aa = attr_align(ev.get("attributes",{}), ed)
        tf = math.log(1 + ed["n_mentions"])
        total += (alpha + (1-alpha)*aa) * tf * idf.get(pcui, 1.0)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--patients", default="pilot/data/cache/v103_patients.jsonl")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--candidate_pool", default="ddxplus",
                    help="ddxplus=49 disease pool, all=full KG")
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    idf, n_dis = compute_phen_idf(G)
    print(f"KG: {n_dis} diseases, IDF range [{min(idf.values()):.2f},{max(idf.values()):.2f}]", flush=True)

    # Candidate pool
    if args.candidate_pool == "ddxplus":
        icd = json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
        pool = sorted({info["cui"] for info in icd.values()
                       if info.get("cui") and info["cui"] in G})
    else:
        pool = sorted(n for n,d in G.nodes(data=True) if d.get("ntype")=="disease")
    print(f"Candidate pool: {len(pool)} diseases", flush=True)

    patients = [json.loads(l) for l in open(args.patients)][:args.n]
    n=c1=c3=c5=c10=0; rr=0.0; n_skip=0
    for p in patients:
        true_cui = p["true_cui"]
        if true_cui not in pool: n_skip += 1; continue
        ev = p["evidence"]
        scores = {d: score_disease(ev, d, G, idf, args.alpha) for d in pool}
        ranked = sorted(pool, key=lambda d: -scores[d])
        n += 1
        try: rk = ranked.index(true_cui)+1
        except: rk = len(pool)
        if rk==1: c1+=1
        if rk<=3: c3+=1
        if rk<=5: c5+=1
        if rk<=10: c10+=1
        rr += 1/rk

    print(f"\n=== v103 full eval (alpha={args.alpha}, pool={args.candidate_pool}) ===")
    print(f"  N={n} (skipped {n_skip} truth-not-in-pool)")
    print(f"  @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr/n:.4f}")


if __name__ == "__main__":
    main()
