#!/usr/bin/env python3
"""Validate soft attribute applicability on the v103 deep KG.

Compares DDXPlus @1 under 3 modes on an identical cosine base:
  none  — CUI name-match only (no attributes)
  unif  — attribute alignment averaged over available attrs (current attr_align)
  appl  — attribute alignment weighted by data-driven applicability;
          inapplicable attributes (empty dist for that phenotype) are NEUTRAL

applicability[phen_cui][attr] = fraction of edges INTO that phenotype (across all
diseases) whose <attr>_dist is non-empty. Derived from the KG, never hand-coded.
"""
from __future__ import annotations
import sys, json, math, pickle, argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from onlykg_eval_v71_selfaware import compute_idf, reweight, load_ddxplus_full, VALUE_CUIS

ATTRS = ["location", "severity", "onset", "character"]
DISCRIM = None
NEARBY = [
    {"face","cheek","lip","tongue","mouth","throat","larynx","head","eye","eyelid","ear","nose","neck"},
    {"leg","thigh","knee","ankle","foot"}, {"arm","shoulder","elbow","wrist","hand","finger"},
    {"chest","lung","heart","back"}, {"abdomen","epigastric","liver","kidney","pelvis","groin"},
    {"skin","generalized","systemic"},
]


def compute_applicability(G):
    """applicability[pcui][attr] = frac of in-edges with non-empty <attr>_dist."""
    deg = defaultdict(int)
    nonempty = defaultdict(lambda: defaultdict(int))
    for _, v, ed in G.edges(data=True):
        if ed.get("etype") != "HAS_PHENOTYPE":
            continue
        deg[v] += 1
        for a in ATTRS:
            if ed.get(a + "_dist"):
                nonempty[v][a] += 1
    appl = {}
    for v, d in deg.items():
        appl[v] = {a: nonempty[v][a] / d for a in ATTRS}
    return appl


def compute_discriminativeness(G):
    """discrim[pcui][attr] = normalized entropy of the POOLED attribute-value
    distribution across all diseases having that phenotype (Idea A2).
    High spread (location varies face/chest/abdomen across diseases) → the attribute
    separates candidates → up-weight. Low spread (always 'generalized') → down-weight.
    This is the 'differentiating power' of a qualifier (Bordage), made per (symptom, attr)."""
    pooled = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # pcui->attr->val->mass
    for _, v, ed in G.edges(data=True):
        if ed.get("etype") != "HAS_PHENOTYPE":
            continue
        for a in ATTRS:
            for val, m in (ed.get(a + "_dist") or {}).items():
                pooled[v][a][val] += m
    discrim = {}
    for v, ad in pooled.items():
        discrim[v] = {}
        for a in ATTRS:
            dist = ad.get(a, {})
            tot = sum(dist.values())
            if tot <= 0 or len(dist) <= 1:
                discrim[v][a] = 0.0; continue
            ent = -sum((m/tot)*math.log(m/tot) for m in dist.values() if m > 0)
            discrim[v][a] = ent / math.log(len(dist))  # normalized 0~1
    return discrim


def compute_value_idf(G):
    """value_idf[attr][value] = log(N_edges / edges that put mass>0 on value).
    Idea A (contrastive): a SPECIFIC attribute value (e.g. 'epigastric') discriminates
    more than a generic one ('generalized'/'moderate'). Grounded in semantic-qualifier
    'differentiating power' (Bordage). IDF family — same principle as symptom IDF."""
    df = defaultdict(lambda: defaultdict(int))
    n = 0
    for _, _, ed in G.edges(data=True):
        if ed.get("etype") != "HAS_PHENOTYPE":
            continue
        n += 1
        for a in ATTRS:
            for val in (ed.get(a + "_dist") or {}):
                df[a][val] += 1
    vidf = {}
    for a in ATTRS:
        vidf[a] = {v: math.log((n+1)/(c+1)) + 1.0 for v, c in df[a].items()}
    return vidf, n


def build_profile(G, dcs_list, kappa=2.0, top_k=80):
    profile, edges_by_d = {}, {}
    all_evs = set()
    for d in dcs_list:
        if d not in G:
            profile[d] = {}; continue
        ed_w = defaultdict(float); best_edge = {}
        for _, pcui, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE":
                continue
            nm = ed.get("n_mentions", 0.0)
            ed_w[pcui] += nm
            if pcui not in best_edge or nm > best_edge[pcui].get("n_mentions", 0):
                best_edge[pcui] = ed
        prof = {p: w/(w+kappa) for p, w in ed_w.items() if w > 0}
        if top_k and len(prof) > top_k:
            keep = dict(sorted(prof.items(), key=lambda x:-x[1])[:top_k])
            prof = keep
        profile[d] = prof
        edges_by_d[d] = {p: best_edge[p] for p in prof}
        all_evs.update(prof.keys())
    return profile, edges_by_d, all_evs


def loc_align(pat, dist):
    if not pat or not dist: return 0.5
    ps, prs = set(pat), set(dist)
    if ps & prs: return min(1.0, sum(dist.get(l,0) for l in ps) + 0.3)
    for g in NEARBY:
        if (ps & g) and (prs & g): return 0.4
    return 0.0


def one_attr_align(a, pat_attrs, ed):
    dist = ed.get(a + "_dist", {})
    if a == "location":
        return loc_align(pat_attrs.get("location", []), dist)
    if a == "severity":
        pv = pat_attrs.get("severity")
        if not pv or not dist: return 0.5
        adj = {"mild":["moderate"],"moderate":["mild","severe"],"severe":["moderate","critical","profound"]}
        return min(1.0, dist.get(pv,0) + 0.4*sum(dist.get(x,0) for x in adj.get(pv,[])))
    if a == "character":
        pv = pat_attrs.get("character", [])
        if not pv or not dist: return 0.5
        return min(1.0, sum(dist.get(c,0) for c in pv))
    if a == "onset":
        pv = pat_attrs.get("onset")
        if not pv or not dist: return 0.5
        return min(1.0, dist.get(pv, 0))
    return 0.5


def patient_value_specificity(a, pat_attrs, vidf):
    """평균 value-IDF of the patient's reported value(s) for attribute a (Idea A).
    Specific value (epigastric) → high; generic (generalized/moderate) → low.
    Normalized to ~[0.5, 1.5] around the attribute's mean IDF."""
    pv = pat_attrs.get(a)
    if pv is None:
        return 1.0
    vals = pv if isinstance(pv, list) else [pv]
    idfs = [vidf[a][v] for v in vals if v in vidf.get(a, {})]
    if not idfs:
        return 1.0
    m = sum(vidf[a].values())/len(vidf[a]) if vidf.get(a) else 1.0
    return max(0.5, min(1.5, (sum(idfs)/len(idfs)) / (m or 1.0)))


def attr_factor(pcui, pat_attrs, ed, mode, appl, alpha, vidf=None):
    if mode == "none" or not pat_attrs:
        return 1.0
    terms, weights = [], []
    for a in ATTRS:
        al = one_attr_align(a, pat_attrs, ed)
        if mode == "unif":
            if (a in pat_attrs) or ed.get(a+"_dist"):
                terms.append(al); weights.append(1.0)
        else:  # appl or contrast
            w = appl.get(pcui, {}).get(a, 0.0)
            if w < 0.05:  # inapplicable -> neutral (skip)
                continue
            if mode == "contrast" and vidf is not None:
                w *= patient_value_specificity(a, pat_attrs, vidf)  # Idea A1
            if mode == "discrim" and DISCRIM is not None:
                w *= (0.3 + DISCRIM.get(pcui, {}).get(a, 0.0))  # Idea A2: cross-disease spread
            terms.append(al); weights.append(w)
    if not weights:
        aa = 0.5
    else:
        aa = sum(t*w for t,w in zip(terms,weights)) / sum(weights)
    return alpha + (1-alpha)*aa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="pilot/data/cache/v103deep120_kg.pkl")
    ap.add_argument("--patients", default="pilot/data/cache/v103_patients.jsonl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--top_k", type=int, default=80)
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    appl = compute_applicability(G)
    vidf, _ = compute_value_idf(G)
    global DISCRIM
    DISCRIM = compute_discriminativeness(G)
    dcs_list, _, _ = load_ddxplus_full(1)  # just for disease pool
    import json as _j
    icd = _j.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    cond = _j.load(open("data/ddxplus/release_conditions_en.json"))
    fr2cui = {i.get("cond-name-fr",""): icd[dn]["cui"] for dn,i in cond.items() if dn in icd}
    pool = sorted(set(fr2cui.values()))
    present = [d for d in pool if d in G and G.out_degree(d) > 0]

    base, edges_by_d, all_evs = build_profile(G, present, top_k=args.top_k)
    base = {d:p for d,p in base.items() if p}
    idf = compute_idf(base, 0.12)
    profile = reweight(base, idf, 1.0, args.beta)

    patients = [json.loads(l) for l in open(args.patients)][:args.n]
    # attribute map per patient: cui -> attributes
    print(f"pool={len(profile)} | evs={len(all_evs)} | applicability computed for {len(appl)} phen", flush=True)

    for mode in ["appl", "discrim"]:
        n=c1=c3=c10=0; rr=0.0
        for p in patients:
            tc = p["true_cui"]
            if tc not in profile: continue
            ev_attr = {e["cui"]: e.get("attributes", {}) for e in p["evidence"]}
            pos = set(ev_attr) & all_evs
            if not pos: continue
            pat_vec = {e: idf.get(e,1.0)**args.beta for e in pos}
            pnorm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
            scores = {}
            for d, prof in profile.items():
                dnorm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
                dot = 0.0
                eb = edges_by_d.get(d, {})
                for e in pos:
                    if e in prof:
                        f = attr_factor(e, ev_attr.get(e,{}), eb.get(e,{}), mode, appl, args.alpha, vidf)
                        dot += pat_vec[e]*prof[e]*f
                scores[d] = dot/(pnorm*dnorm)
            ranked = sorted(scores, key=lambda d:-scores[d])
            n+=1; rk = ranked.index(tc)+1
            if rk==1: c1+=1
            if rk<=3: c3+=1
            if rk<=10: c10+=1
            rr+=1/rk
        print(f"  mode={mode:5s}: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @10={100*c10/n:.2f}% MRR={rr/n:.4f} (N={n})", flush=True)


if __name__ == "__main__":
    main()
