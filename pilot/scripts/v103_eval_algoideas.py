#!/usr/bin/env python3
"""Sequentially validate algorithm ideas on the v103 deep KG (DDXPlus).

Baseline: cosine(patient pos CUIs, disease profile) with IDF^beta weighting.
Modes:
  base   — baseline cosine+IDF
  chief  — Idea 1: × chief-complaint prior P(D | INITIAL_EVIDENCE)^gamma
  nb     — Idea 2: Bayesian likelihood-ratio scoring (P(E|D) from KG frequency)
  hier   — Idea 4: (placeholder, added later)

All benchmark-blind; chief complaint token is allowed (disease/evidence name only).
"""
from __future__ import annotations
import sys, json, math, pickle, argparse, csv, ast
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from onlykg_eval_v71_selfaware import compute_idf, reweight, VALUE_CUIS


def load_patients(n_max):
    value_cuis = json.load(open(VALUE_CUIS))
    icd = json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    cond = json.load(open("data/ddxplus/release_conditions_en.json"))
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    dcs = sorted(set(fr2cui.values()))
    pats = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if len(pats) >= n_max: break
            tc = fr2cui.get(row["PATHOLOGY"])
            if tc not in dcs: continue
            pos = set()
            for ev in ast.literal_eval(row["EVIDENCES"]):
                base = ev.split("_@_",1)[0] if "_@_" in ev else ev
                m = value_cuis.get(base, {})
                v = m.get("_question", [])
                if isinstance(v, list): pos.update(v)
                if "_@_" in ev:
                    val = ev.split("_@_",1)[1]
                    vv = m.get(val, [])
                    if isinstance(vv, list): pos.update(vv)
            # chief complaint
            chief = set()
            ie = row.get("INITIAL_EVIDENCE","").strip()
            if ie:
                m = value_cuis.get(ie, {})
                v = m.get("_question", [])
                if isinstance(v, list): chief.update(v)
            pats.append({"true": tc, "pos": pos, "chief": chief})
    return dcs, pats


def build_profile(G, dcs, kappa=2.0, top_k=80):
    profile, all_evs = {}, set()
    for d in dcs:
        if d not in G: profile[d] = {}; continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            ed_w[p] += ed.get("n_mentions", 0.0)
        prof = {p: w/(w+kappa) for p,w in ed_w.items() if w>0}
        if top_k and len(prof)>top_k:
            prof = dict(sorted(prof.items(), key=lambda x:-x[1])[:top_k])
        profile[d] = prof; all_evs |= set(prof)
    return profile, all_evs


def build_freq_profile(G, dcs):
    """P(E|D) = frequency_in_abstracts (the 'frequency' edge attr), for Bernoulli NB."""
    prof = {}
    for d in dcs:
        if d not in G: prof[d] = {}; continue
        pe = {}
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            f = ed.get("frequency", 0.0) or 0.0
            pe[p] = max(pe.get(p, 0.0), f)
        prof[d] = pe
    return prof


def evaluate_nb(freq_prof, pats, vocab, eps):
    """Bernoulli NB: score(D)=Σ_E∈vocab [x_E logP(E|D) + (1-x_E)log(1-P(E|D))].
    P(E|D)=freq clamped to [eps,1-eps]; absent-in-profile → eps.
    The (1-x_E)log(1-P) term = principled pertinent-negative (D typically has E but patient lacks it)."""
    dlist = [d for d in freq_prof if freq_prof[d]]
    logc = {d: {e: (math.log(min(max(freq_prof[d].get(e,eps),eps),1-eps)),
                    math.log(1-min(max(freq_prof[d].get(e,eps),eps),1-eps))) for e in vocab}
            for d in dlist}
    n=c1=c3=c10=0; rr=0.0
    for p in pats:
        tc = p["true"]
        if tc not in freq_prof or not freq_prof[tc]: continue
        pos = p["pos"] & vocab
        if not pos: continue
        scores = {}
        for d in dlist:
            lc = logc[d]; s = 0.0
            for e in vocab:
                s += lc[e][0] if e in pos else lc[e][1]
            scores[d] = s
        ranked = sorted(dlist, key=lambda d:-scores[d])
        n+=1; rk = ranked.index(tc)+1
        if rk==1: c1+=1
        if rk<=3: c3+=1
        if rk<=10: c10+=1
        rr+=1/rk
    return n, 100*c1/n, 100*c3/n, 100*c10/n, rr/n


def chief_prior(profile, all_evs):
    """prior[chief_cui][d] = prof[d][chief]/max_d, i.e. how prominent the presenting
    complaint is in disease d. Diseases lacking it get a small floor."""
    pri = {}
    for c in all_evs:
        col = {d: prof.get(c, 0.0) for d, prof in profile.items()}
        mx = max(col.values()) or 1.0
        pri[c] = {d: v/mx for d, v in col.items()}
    return pri


def evaluate(profile, idf, beta, pats, all_evs, dcs, mode, gamma, pri):
    n=c1=c3=c10=0; rr=0.0
    dlist = [d for d in profile if profile[d]]
    for p in pats:
        tc = p["true"]
        if tc not in profile or not profile[tc]: continue
        pos = p["pos"] & all_evs
        if not pos: continue
        patv = {e: idf.get(e,1.0)**beta for e in pos}
        pn = math.sqrt(sum(v*v for v in patv.values())) or 1e-9
        scores = {}
        for d in dlist:
            prof = profile[d]
            dn = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
            dot = sum(patv[e]*prof[e] for e in pos if e in prof)
            s = dot/(pn*dn)
            if mode == "chief" and p["chief"]:
                cp = max((pri[c][d] for c in p["chief"] if c in pri), default=0.0)
                s *= (0.1 + 0.9*cp) ** gamma   # soft gate, floor 0.1
            scores[d] = s
        ranked = sorted(dlist, key=lambda d:-scores[d])
        n+=1; rk = ranked.index(tc)+1
        if rk==1: c1+=1
        if rk<=3: c3+=1
        if rk<=10: c10+=1
        rr+=1/rk
    return n, 100*c1/n, 100*c3/n, 100*c10/n, rr/n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="pilot/data/cache/v103deep120_kg.pkl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--gamma_sweep", default="0.5,1.0,2.0")
    ap.add_argument("--eps_sweep", default="0.01,0.05,0.1")
    ap.add_argument("--idea", default="chief", choices=["chief","nb"])
    args = ap.parse_args()

    G = pickle.load(open(args.graph,"rb"))
    dcs, pats = load_patients(args.n)
    profile, all_evs = build_profile(G, dcs)
    idf = compute_idf(profile, 0.12)
    profile = reweight(profile, idf, 1.0, args.beta)
    pri = chief_prior(profile, all_evs)
    nchief = sum(1 for p in pats if p["chief"])
    print(f"pool={sum(1 for d in profile if profile[d])} | pats={len(pats)} | with chief={nchief}", flush=True)

    r = evaluate(profile, idf, args.beta, pats, all_evs, dcs, "base", 0, pri)
    print(f"  base       : @1={r[1]:.2f}% @3={r[2]:.2f}% @10={r[3]:.2f}% MRR={r[4]:.4f} (N={r[0]})", flush=True)

    if args.idea == "chief":
        for g in [float(x) for x in args.gamma_sweep.split(",")]:
            r = evaluate(profile, idf, args.beta, pats, all_evs, dcs, "chief", g, pri)
            print(f"  chief g={g:.1f}: @1={r[1]:.2f}% @3={r[2]:.2f}% @10={r[3]:.2f}% MRR={r[4]:.4f}", flush=True)
    elif args.idea == "nb":
        freq_prof = build_freq_profile(G, dcs)
        # vocab = observable patient-evidence space ∩ KG phenotypes
        patspace = set()
        for p in pats: patspace |= p["pos"]
        vocab = patspace & all_evs
        print(f"  NB vocab={len(vocab)} (patient-evidence ∩ KG)", flush=True)
        for eps in [float(x) for x in args.eps_sweep.split(",")]:
            r = evaluate_nb(freq_prof, pats, vocab, eps)
            print(f"  nb eps={eps:.3f}: @1={r[1]:.2f}% @3={r[2]:.2f}% @10={r[3]:.2f}% MRR={r[4]:.4f} (N={r[0]})", flush=True)


if __name__ == "__main__":
    main()
