#!/usr/bin/env python3
"""v99 SymCat eval — tau=1.5 algorithm parameters (no negative penalty applicable since SymCat lacks binary).
SymCat doesn't use negative penalty (no binary evidence default values), so tau has no effect.
This is essentially the same as v93 SymCat (5.08%) — confirm or improve."""
from __future__ import annotations
import sys, json, math, pickle, argparse, random
from pathlib import Path
from collections import defaultdict
PR_UNIVERSE = "pilot/data/pr_universe.json"


def build_profile(G, dcs, kappa, pr):
    profile={}; all_evs=set()
    allowed={"patient_reportable","history","demographic"}
    for d in dcs:
        if d not in G: profile[d]={}; continue
        ed_w=defaultdict(float)
        for _,p,ed in G.out_edges(d, data=True):
            if ed.get("etype")!="HAS_PHENOTYPE": continue
            cat=ed.get("category")
            if cat is None and p not in pr: continue
            if cat is not None and cat not in allowed: continue
            ed_w[p]+=ed.get("weight",0)
        profile[d]={p:w/(w+kappa) for p,w in ed_w.items() if w>0}
        all_evs.update(profile[d].keys())
    return profile, all_evs


def compute_idf(profile, df_thr):
    N=len(profile); df=defaultdict(int)
    for prof in profile.values():
        for e,p in prof.items():
            if p>=df_thr: df[e]+=1
    return {e: math.log((N+1)/(df_e+1))+1.0 for e,df_e in df.items()}


def reweight(profile, idf, beta):
    return {d:{e:p*(idf.get(e,1)**beta) for e,p in prof.items()} for d,prof in profile.items()}


def score_cos(pcuis, profile_rw, idf, beta):
    pat_vec={e:idf.get(e,1)**beta for e in pcuis}
    p_norm=math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
    scores={}
    for d,prof in profile_rw.items():
        if not prof: scores[d]=-1e9; continue
        d_norm=math.sqrt(sum(v*v for v in prof.values())) or 1e-9
        dot=sum(pat_vec[e]*prof[e] for e in pcuis if e in prof)
        scores[d]=dot/(p_norm*d_norm)
    return scores


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--graph", default="pilot/data/onlykg_graph_v93_s3.pkl")
    ap.add_argument("--n_patients_per_d", type=int, default=100)
    ap.add_argument("--beta_sweep", default="0.5,0.75,1.0,1.25")
    args=ap.parse_args()

    G=pickle.load(open(args.graph,"rb"))
    pr=set(json.load(open(PR_UNIVERSE)))
    parsed=json.load(open("data/symcat/symcat_parsed_full.json"))
    sym_map=json.load(open("data/symcat/symptom_umls_mapping.json"))["mapping"]
    dis_map=json.load(open("data/symcat/disease_umls_mapping.json"))["mapping"]
    sym2cui={n:v["umls_cui"] for n,v in sym_map.items() if v.get("umls_cui")}
    dis2cui={n:v["umls_cui"] for n,v in dis_map.items() if v.get("umls_cui")}
    cand=[(dn,dis2cui[dn]) for dn in parsed["disease_symptom_pairs"] if dis2cui.get(dn)]
    dcs_list=sorted({c for _,c in cand})
    base,all_evs=build_profile(G, dcs_list, 20.0, pr)
    idf=compute_idf(base, 0.12)
    random.seed(42)
    patients=[]
    for dname,true_cui in cand:
        sym_prob={sym2cui.get(s):p/100.0 for s,p in parsed["disease_symptom_pairs"][dname] if sym2cui.get(s)}
        sym_prob={c:p for c,p in sym_prob.items() if c is not None}
        for _ in range(args.n_patients_per_d):
            pcuis={c for c,p in sym_prob.items() if random.random()<p}
            if pcuis: patients.append((true_cui, pcuis))
    print(f"SymCat: {len(patients)} patients, {len(dcs_list)} candidates", flush=True)
    for beta_s in args.beta_sweep.split(","):
        beta=float(beta_s)
        profile_rw=reweight(base, idf, beta)
        n=c1=c3=c10=0; rr=0
        for true_cui, raw in patients:
            pcuis=raw & all_evs
            if not pcuis: n+=1; continue
            s=score_cos(pcuis, profile_rw, idf, beta)
            ranked=sorted(dcs_list, key=lambda d:-s.get(d,-1e9))
            try: rk=ranked.index(true_cui)+1
            except: rk=len(dcs_list)
            n+=1
            if rk==1: c1+=1
            if rk<=3: c3+=1
            if rk<=10: c10+=1
            rr+=1/rk
        print(f"  beta={beta:.2f}: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @10={100*c10/n:.2f}% MRR={rr/n:.4f}",
              flush=True)


if __name__=="__main__":
    main()
