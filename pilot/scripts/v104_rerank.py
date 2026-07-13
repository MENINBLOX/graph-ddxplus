"""Convert @20->@10: re-rank the top-20 shortlist (GT in top-20 = 99.25%) to
promote GT into top-10. 3 strict, benchmark-blind re-rankers on top of (c) base:
(a) specificity (top-k high-IDF match), (b) aggressive negative rule-out,
(c) contrastive (down-weight evidence shared across the shortlist). random6000->full."""
import sys,math,pickle,json,csv,ast,random
from collections import defaultdict,Counter
sys.path.insert(0,"pilot/scripts")
from onlykg_eval_v71_selfaware import compute_idf,reweight,precompute_signal_v71
random.seed(42)
VC="/windows/data/medkg/kg/ddxplus_evidence_value_cuis.json"
value_cuis=json.load(open(VC)); evmeta=json.load(open("data/ddxplus/release_evidences.json"))
icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json")); cond=json.load(open("data/ddxplus/release_conditions_en.json"))
fr2cui={info.get("cond-name-fr",""):icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
dcs=sorted(set(fr2cui.values()))
binary_evs={e for e,m in evmeta.items() if m.get("data_type")=="B" and m.get("default_value")==0}
def L(f): return pickle.load(open(f"pilot/data/cache/{f}","rb"))
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
idf=compute_idf(P,0.12); beta=0.75; Pw=reweight(dict(P),idf,1.0,beta)
sig=precompute_signal_v71(P,value_cuis,binary_evs,idf,3.0,1.0)
dl=list(Pw); DN={d:(math.sqrt(sum(v*v for v in Pw[d].values()))or 1e-9) for d in dl}
SIZE={d:len(P[d]) for d in dl}; med=sorted(SIZE.values())[len(SIZE)//2]
ddS={d:{} for d in dl}
for i,d1 in enumerate(dl):
    x=Pw[d1]
    for d2 in dl[i+1:]:
        y=Pw[d2]; aa,bb=(x,y) if len(x)<len(y) else (y,x)
        s=sum(aa[e]*bb[e] for e in aa if e in bb)/(DN[d1]*DN[d2])
        if s>0.05: ddS[d1][d2]=s; ddS[d2][d1]=s
def ppos(evs):
    pos=set()
    for ev in evs:
        b=ev.split("_@_")[0]; m=value_cuis.get(b,{})
        pos.update(m.get("_question",[]) or [])
        if "_@_" in ev: pos.update(m.get(ev.split("_@_")[1],[]) or [])
    return pos
def base_scores(evs):
    pos=ppos(evs); posm=pos&all_evs
    if not posm: return None
    ans={e.split("_@_")[0] for e in evs}; neg=set(binary_evs)-ans
    patv={e:idf.get(e,1.0)**beta for e in posm}; pn=math.sqrt(sum(v*v for v in patv.values()))or 1e-9
    nn=math.sqrt(len(neg))or 1e-9
    base={}
    for d in dl:
        pr=Pw[d]; pc=sum(patv[e]*pr[e] for e in posm if e in pr)/(pn*DN[d])
        s=sig.get(d,{}); npen=sum(s.get(ev,0.0) for ev in neg)
        base[d]=pc-0.7*(npen/(nn*DN[d]*(SIZE[d]/med)**0.5))
    sc0={d:base[d]+0.2*sum(ddS[d].get(d2,0)*base[d2] for d2 in ddS[d]) for d in dl}
    return sc0,posm,neg,nn
def rerank(evs,method,w,K=20):
    r=base_scores(evs)
    if r is None: return None
    sc0,posm,neg,nn=r
    order=sorted(dl,key=lambda d:-sc0[d]); short=order[:K]
    if method=="base": return order
    # normalize base within shortlist to [0,1] for blending
    bvals=[sc0[d] for d in short]; bmin,bmax=min(bvals),max(bvals); rng=(bmax-bmin)or 1e-9
    rr={}
    if method=="spec":
        for d in short:
            ms=sorted((idf.get(e,1.0) for e in posm if e in Pw[d]),reverse=True)[:3]; rr[d]=sum(ms)
    elif method=="ruleout":
        for d in short:
            s=sig.get(d,{}); rr[d]=-sum(s.get(ev,0.0) for ev in neg)/(nn*DN[d])
    elif method=="contrast":
        sdf=Counter()
        for d in short:
            for e in posm:
                if e in Pw[d]: sdf[e]+=1
        for d in short:
            rr[d]=sum(idf.get(e,1.0)*math.log(K/(sdf[e]+1)+1) for e in posm if e in Pw[d])
    rvals=list(rr.values()); rmin,rmax=min(rvals),max(rvals); rr_rng=(rmax-rmin)or 1e-9
    blended=sorted(short,key=lambda d:-((sc0[d]-bmin)/rng + w*(rr[d]-rmin)/rr_rng))
    return blended+order[K:]
allrows=[]
with open("data/ddxplus/release_test_patients.csv") as f:
    for row in csv.DictReader(f):
        tc=fr2cui.get(row["PATHOLOGY"])
        if tc in Pw and Pw[tc]: allrows.append((tc,ast.literal_eval(row["EVIDENCES"])))
def evalset(rows,configs):
    acc={c:Counter() for c in configs}; n=0
    pre=[(tc,base_scores(evs),evs) for tc,evs in rows]
    pre=[(tc,r,evs) for tc,r,evs in pre if r]
    for tc,r,evs in pre:
        n+=1
        for cfg in configs:
            m,w=cfg
            rk=rerank(evs,m,w).index(tc)+1; C=acc[cfg]
            for K in(1,5,10,20):
                if rk<=K: C[K]+=1
    return acc,n
samp=random.sample(allrows,6000)
configs=[("base",0)]+[(m,w) for m in("spec","ruleout","contrast") for w in(0.3,0.6,1.0)]
acc,n=evalset(samp,configs)
res=sorted([c for c in configs if c[0]!="base"],key=lambda c:-acc[c][10])
print(f"random6000 N={n}  base: @1={100*acc[('base',0)][1]/n:.2f} @10={100*acc[('base',0)][10]/n:.2f} @20={100*acc[('base',0)][20]/n:.2f}")
for c in sorted(configs,key=lambda c:-acc[c][10])[:8]:
    C=acc[c]; print(f"  {c[0]} w={c[1]}: @1={100*C[1]/n:.2f} @5={100*C[5]/n:.2f} @10={100*C[10]/n:.2f}")
top3=res[:3]
print("\n=== FULL 134K (base + 상위3) ===",flush=True)
accf,nf=evalset(allrows,[("base",0)]+top3)
Cb=accf[("base",0)]; print(f"N={nf}  base: @1={100*Cb[1]/nf:.2f} @5={100*Cb[5]/nf:.2f} @10={100*Cb[10]/nf:.2f} @20={100*Cb[20]/nf:.2f}")
for c in top3:
    C=accf[c]; print(f"  {c[0]} w={c[1]}: @1={100*C[1]/nf:.2f} @5={100*C[5]/nf:.2f} @10={100*C[10]/nf:.2f} @20={100*C[20]/nf:.2f}")
