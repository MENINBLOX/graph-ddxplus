"""(c) Negative-penalty profile-size normalization. Large profiles (PE=94 phen)
get over-penalized by the negative-evidence term. Test lambda + size-normalization
variants. random6000 -> top3 FULL."""
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
SIZE={d:len(P[d]) for d in dl}; medsize=sorted(SIZE.values())[len(SIZE)//2]
ddS={d:{} for d in dl}
for i,d1 in enumerate(dl):
    x=Pw[d1]
    for d2 in dl[i+1:]:
        y=Pw[d2]; aa,bb=(x,y) if len(x)<len(y) else (y,x)
        s=sum(aa[e]*bb[e] for e in aa if e in bb)/(DN[d1]*DN[d2])
        if s>0.05: ddS[d1][d2]=s; ddS[d2][d1]=s
def patient_pos(evs):
    pos=set()
    for ev in evs:
        b=ev.split("_@_")[0]; m=value_cuis.get(b,{})
        pos.update(m.get("_question",[]) or [])
        if "_@_" in ev: pos.update(m.get(ev.split("_@_")[1],[]) or [])
    return pos
def precompute(evs):
    pos=patient_pos(evs); posm=pos&all_evs
    if not posm: return None
    ans={e.split("_@_")[0] for e in evs}; neg=set(binary_evs)-ans
    patv={e:idf.get(e,1.0)**beta for e in posm}; pn=math.sqrt(sum(v*v for v in patv.values()))or 1e-9
    nn=math.sqrt(len(neg))or 1e-9
    pc={}; npen={}
    for d in dl:
        pr=Pw[d]; pc[d]=sum(patv[e]*pr[e] for e in posm if e in pr)/(pn*DN[d])
        s=sig.get(d,{}); npen[d]=sum(s.get(ev,0.0) for ev in neg)
    return pc,npen,nn
def evalrows(rows,configs):
    acc={c:Counter() for c in configs}; n=0
    pre=[]
    for tc,evs in rows:
        r=precompute(evs)
        if r is None: continue
        pre.append((tc,r))
    for tc,(pc,npen,nn) in pre:
        n+=1
        for cfg in configs:
            lam,p=cfg
            base={}
            for d in dl:
                sf=(SIZE[d]/medsize)**p
                base[d]=pc[d]-lam*(npen[d]/(nn*DN[d]*sf))
            sc={d:base[d]+0.2*sum(ddS[d].get(d2,0)*base[d2] for d2 in ddS[d]) for d in dl}
            rk=sorted(sc,key=lambda d:-sc[d]).index(tc)+1; C=acc[cfg]
            for K in(1,5,10,20):
                if rk<=K: C[K]+=1
    return acc,n
allrows=[]
with open("data/ddxplus/release_test_patients.csv") as f:
    for row in csv.DictReader(f):
        tc=fr2cui.get(row["PATHOLOGY"])
        if tc in Pw and Pw[tc]: allrows.append((tc,ast.literal_eval(row["EVIDENCES"])))
samp=random.sample(allrows,6000)
# configs: (lam, size-power). p=0 = baseline (current). p>0 downweights big-profile penalty
configs=[(1.0,0.0)]+[(lam,p) for lam in (0.7,1.0,1.3) for p in (0.5,1.0)]
acc,n=evalrows(samp,configs)
res=sorted(configs,key=lambda c:-acc[c][1])
print(f"random6000 N={n}  (medsize={medsize})")
print(f"  baseline(lam1,p0): @1={100*acc[(1.0,0.0)][1]/n:.2f} @10={100*acc[(1.0,0.0)][10]/n:.2f}")
for c in res[:5]:
    C=acc[c]; print(f"  lam={c[0]} sizep={c[1]}: @1={100*C[1]/n:.2f} @5={100*C[5]/n:.2f} @10={100*C[10]/n:.2f}")
top3=res[:3]
print("\n=== FULL 134K ===",flush=True)
accf,nf=evalrows(allrows,[(1.0,0.0)]+top3)
print(f"N={nf}")
print(f"  baseline(lam1,p0): @1={100*accf[(1.0,0.0)][1]/nf:.2f} @5={100*accf[(1.0,0.0)][5]/nf:.2f} @10={100*accf[(1.0,0.0)][10]/nf:.2f} @20={100*accf[(1.0,0.0)][20]/nf:.2f}")
for c in top3:
    C=accf[c]; print(f"  lam={c[0]} sizep={c[1]}: @1={100*C[1]/nf:.2f} @5={100*C[5]/nf:.2f} @10={100*C[10]/nf:.2f} @20={100*C[20]/nf:.2f}")
