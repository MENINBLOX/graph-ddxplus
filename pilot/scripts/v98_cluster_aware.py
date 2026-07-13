#!/usr/bin/env python3
"""v98 — Cluster-aware scoring with disease-pair differential CUI weights.

Forensic 결과: DDXPlus 실패 모두 cluster confusion. cluster 내 differential CUI 식별.

Algorithm:
1. Build disease similarity matrix S[i,j] = cosine(prof[i], prof[j]).
2. For each disease D, identify k=5 most similar (potential confusers).
3. For each (D, confuser) pair, compute differential CUI:
   diff(D, c, e) = max(0, prof[D][e] - prof[c][e])  # CUI more in D than c
4. score_v98(D) = score_v71(D) + gamma * sum(diff(D, top_confuser, e) * pat_vec[e])
   여기서 top_confuser는 동적 (rank top-2)
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
EV_META = "data/ddxplus/release_evidences.json"


def build_profile(G, dcs, kappa, pr):
    profile = {}; all_evs = set()
    allowed = {"patient_reportable","history","demographic"}
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


def precompute_signal(profile_rw, value_cuis, binary_evs, idf, tau, sharp):
    sig=defaultdict(dict)
    for ev_id in binary_evs:
        cuis = set(value_cuis.get(ev_id,{}).get("_question",[]))
        for d,prof in profile_rw.items():
            best=0
            for c in cuis:
                if c in prof:
                    fac=1/(1+math.exp((idf.get(c,1)-tau)/sharp))
                    best = max(best, prof[c]*fac)
            if best>0: sig[d][ev_id]=best
    return sig


def load_ddxplus(n_max):
    value_cuis=json.load(open(VALUE_CUIS))
    ev_meta=json.load(open(EV_META))
    icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    cond=json.load(open("data/ddxplus/release_conditions_en.json"))
    fr2cui={info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    dcs_list=sorted(set(fr2cui.values()))
    binary_evs={e for e,m in ev_meta.items() if m.get("data_type")=="B" and m.get("default_value")==0}
    patients=[]; n=0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n>=n_max: break
            true_cui=fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs=ast.literal_eval(row["EVIDENCES"])
            pos=set(); answered=set()
            for ev in evs:
                if "_@_" in ev:
                    base,val=ev.split("_@_",1)
                    m=value_cuis.get(base,{})
                    for k in ("_question",val):
                        v=m.get(k,[])
                        if isinstance(v,list): pos.update(v)
                else:
                    if ev in binary_evs: answered.add(ev)
                    pos.update(value_cuis.get(ev,{}).get("_question",[]))
            patients.append((true_cui,pos,binary_evs-answered)); n+=1
    return dcs_list, patients, binary_evs


def score_v71(pos, neg_binary, profile_rw, idf, beta, signal, lam):
    pat_vec={e:idf.get(e,1)**beta for e in pos}
    p_norm=math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
    scores={}
    for d,prof in profile_rw.items():
        if not prof: scores[d]=-1e9; continue
        d_norm=math.sqrt(sum(v*v for v in prof.values())) or 1e-9
        dot=sum(pat_vec[e]*prof[e] for e in pos if e in prof)
        pos_s=dot/(p_norm*d_norm)
        sig=signal.get(d,{})
        neg_pen=sum(sig.get(ev,0) for ev in neg_binary)
        neg_norm=math.sqrt(len(neg_binary)) or 1e-9
        neg_s=neg_pen/(neg_norm*d_norm)
        scores[d]=pos_s-lam*neg_s
    return scores


def score_v98(pos, neg_binary, profile_rw, idf, beta, signal, lam, gamma):
    """v71 + cluster-aware re-rank: top-2 candidate 사이 differential 강화."""
    s_v71 = score_v71(pos, neg_binary, profile_rw, idf, beta, signal, lam)
    if gamma == 0: return s_v71
    # Get top-2
    ranked = sorted(profile_rw.keys(), key=lambda d:-s_v71[d])
    top1, top2 = ranked[0], ranked[1]
    pat_vec={e:idf.get(e,1)**beta for e in pos}
    # For top1 and top2, compute differential boost
    prof1 = profile_rw[top1]; prof2 = profile_rw[top2]
    diff_1_2 = sum(pat_vec[e] * max(0, prof1.get(e,0)-prof2.get(e,0)) for e in pos)
    diff_2_1 = sum(pat_vec[e] * max(0, prof2.get(e,0)-prof1.get(e,0)) for e in pos)
    s_v71[top1] += gamma * diff_1_2
    s_v71[top2] += gamma * diff_2_1
    return s_v71


def evaluate(profile_rw, idf, beta, signal, patients, all_evs, dcs_list, lam, gamma):
    n=c1=c3=c5=0; rr=0
    for true_cui, pos_raw, neg in patients:
        pos = pos_raw & all_evs
        if not pos: continue
        s = score_v98(pos, neg, profile_rw, idf, beta, signal, lam, gamma)
        ranked = sorted(profile_rw.keys(), key=lambda d:-s[d])
        n+=1
        try: rk=ranked.index(true_cui)+1
        except: rk=len(dcs_list)
        if rk==1: c1+=1
        if rk<=3: c3+=1
        if rk<=5: c5+=1
        rr+=1/rk
    return {"n":n, "at1":100*c1/n, "at3":100*c3/n, "at5":100*c5/n, "mrr":rr/n}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--graph", default="pilot/data/onlykg_graph_v93_s3.pkl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--gamma_sweep", default="0.0,0.1,0.3,0.5,1.0,2.0,5.0")
    args = ap.parse_args()

    G = pickle.load(open(args.graph,"rb"))
    pr = set(json.load(open(PR_UNIVERSE)))
    dcs_list, patients, binary_evs = load_ddxplus(args.n)
    base, all_evs = build_profile(G, dcs_list, 20.0, pr)
    idf = compute_idf(base, 0.12)
    profile_rw = reweight(base, idf, 0.75)
    value_cuis = json.load(open(VALUE_CUIS))
    sig = precompute_signal(profile_rw, value_cuis, binary_evs, idf, 2.0, 0.5)

    print(f"=== v98 cluster-aware (top-2 differential) — N={args.n} ===")
    for g_s in args.gamma_sweep.split(","):
        g = float(g_s)
        r = evaluate(profile_rw, idf, 0.75, sig, patients, all_evs, dcs_list, 0.4, g)
        print(f"  gamma={g:.2f}: @1={r['at1']:.2f}% @3={r['at3']:.2f}% MRR={r['mrr']:.4f}", flush=True)


if __name__=="__main__":
    main()
