"""Integrate v104 attribute channels into the strong union base (best config:
base6+anatIE+hallmark, cosine+neg+disease-spreading @1 47.1/@10 94.4). final =
base_score + lambda * Σ attr_channel. Tests whether structured attributes
(esp. location) add @10 on top of the strong base. Patient alignment verified 1:1."""
import sys,math,pickle,json,itertools
from collections import defaultdict,Counter
import numpy as np
sys.path.insert(0,"pilot/scripts")
import onlykg_eval_v71_selfaware as V71
from onlykg_eval_v71_selfaware import compute_idf,reweight,precompute_signal_v71,score
dcs,pats,binary_evs=V71.load_ddxplus_full(3000)
value_cuis=json.load(open("/windows/data/medkg/kg/ddxplus_evidence_value_cuis.json"))
def L(f): return pickle.load(open(f"pilot/data/cache/{f}","rb"))
# --- union base (best config) ---
files=["v103h_exh_kg.pkl","v103ddx49_sci_kg.pkl","v103pres_ddx49_sci_kg.pkl","v103sci_ddx49_kg.pkl","v103i_clean_kg.pkl","v103j_exp_kg.pkl","v103q_anat_kg.pkl","v103r_hallmark_kg.pkl","v103r_hallmark_dec_kg.pkl"]
wt={"v103i_clean_kg.pkl":3,"v103j_exp_kg.pkl":3,"v103q_anat_kg.pkl":3,"v103r_hallmark_kg.pkl":3,"v103r_hallmark_dec_kg.pkl":4}
Praw=defaultdict(lambda:defaultdict(float))
for fn in files:
    G=L(fn)
    for d in dcs:
        if d in G:
            for _,p,e in G.out_edges(d,data=True):
                if e.get("etype")=="HAS_PHENOTYPE": Praw[d][p]+=wt.get(fn,1)*e.get("n_mentions",0.0)
P={d:dict((p,x/(x+2.0)) for p,x in Praw[d].items() if x>0) for d in dcs if Praw[d]}
all_evs=set().union(*[set(p) for p in P.values()])
idf=compute_idf(P,0.12); beta=0.75
Pw=reweight(dict(P),idf,1.0,beta)
sig=precompute_signal_v71(P,value_cuis,binary_evs,idf,3.0,1.0)
dl=list(Pw); DN={d:(math.sqrt(sum(v*v for v in Pw[d].values()))or 1e-9) for d in dl}
ddS={d:{} for d in dl}
for i,d1 in enumerate(dl):
    x=Pw[d1]
    for d2 in dl[i+1:]:
        y=Pw[d2]; aa,bb=(x,y) if len(x)<len(y) else (y,x)
        s=sum(aa[e]*bb[e] for e in aa if e in bb)/(DN[d1]*DN[d2])
        if s>0.05: ddS[d1][d2]=s; ddS[d2][d1]=s
# --- v104 attribute channels ---
D=pickle.load(open("pilot/data/cache/v104_attr_vectors.pkl","rb"))
vpats=D["patients"]; dis=D["diseases"]
ATTRS=["location","severity","onset","character","radiation","timing"]
df=defaultdict(int)
for d in dl:
    if d not in dis: continue
    toks=set()
    for c in ATTRS:
        for t in dis[d]["attr"].get(c,set()): toks.add((c,t))
    for t in toks: df[t]+=1
Na=len([d for d in dl if d in dis])
aidf={t:math.log((Na+1)/(df[t]+1))+1.0 for t in df}
dvec={}
for d in dl:
    dvec[d]={}
    for c in ATTRS:
        s=dis[d]["attr"].get(c,set()) if d in dis else set()
        dvec[d][c]={t:aidf.get((c,t),1.0) for t in s}
adnorm={d:(math.sqrt(sum(w*w for c in ATTRS for w in dvec[d][c].values()))or 1e-9) for d in dl}
def attr_contrib(pi,d,chans):
    pa=vpats[pi]["attr"]; tot=0.0
    for c in chans:
        dw=dvec[d][c]
        for t in pa.get(c,set()):
            if t in dw: tot+=aidf.get((c,t),1.0)*dw[t]
    return tot/adnorm[d]
def run(chans,lam):
    atK=Counter(); n=0
    for pi,(tc,pos,neg) in enumerate(pats):
        if tc not in Pw or not Pw[tc]: continue
        posm=pos&all_evs
        if not posm: continue
        b=score(posm,neg,Pw,idf,beta,sig,1.0)
        sc={d:b[d]+0.2*sum(ddS[d].get(d2,0)*b[d2] for d2 in ddS[d]) for d in dl}
        if lam>0 and chans:
            for d in dl: sc[d]+=lam*attr_contrib(pi,d,chans)
        rk=sorted(sc,key=lambda d:-sc[d]).index(tc)+1; n+=1
        for K in(1,5,10,20):
            if rk<=K: atK[K]+=1
    return {K:100*atK[K]/n for K in(1,5,10,20)}
r=run([],0); print(f"union base(속성0)      : @1={r[1]:.2f} @5={r[5]:.2f} @10={r[10]:.2f} @20={r[20]:.2f}")
for chans,nm in [(["location"],"+location"),(["location","severity","onset","character"],"+loc+sev+ons+char"),(ATTRS,"+all6")]:
    for lam in [0.3,0.6,1.0]:
        r=run(chans,lam); print(f"{nm:20s} lam={lam}: @1={r[1]:.2f} @5={r[5]:.2f} @10={r[10]:.2f} @20={r[20]:.2f}")
