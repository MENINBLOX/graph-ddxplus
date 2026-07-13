"""Attribute-combination ablation on the union base+(c). For each subset of the
6 DDXPlus-active attributes, add that attribute's CUI/token overlap to the score.
random6000 -> top3 full. Confirms the qualitative finding quantitatively."""
import sys,math,pickle,json,csv,ast,random,itertools
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
A=pickle.load(open("pilot/data/cache/v104b_attr.pkl","rb")); dz=A["dz"]
ACT=["location","radiation","character","severity","onset","timing"]
# attribute IDF over diseases
adf={a:defaultdict(int) for a in ACT}
for d in dl:
    for a in ACT:
        for t in dz.get(d,{}).get("attr",{}).get(a,set()): adf[a][t]+=1
Nd=len(dl)
def aidf(a,t): return math.log((Nd+1)/(adf[a].get(t,0)+1))+1.0
ATTR_SUF={"endroitducorps":"location","precis":"location","irrad":"radiation","intens":"severity","prurit":"severity","sev":"severity","soudain":"onset","carac":"character","aboy":"character","noct":"timing","nuit":"timing"}
def sufof(eid):
    for s,a in ATTR_SUF.items():
        if eid.lower().endswith("_"+s) or eid.lower()==s: return a
    return None
def parse(evs):
    base=set(); pat=defaultdict(set)
    for ev in evs:
        eid=ev.split("_@_")[0] if "_@_" in ev else ev; val=ev.split("_@_")[1] if "_@_" in ev else None
        a=sufof(eid); m=value_cuis.get(eid,{})
        if a is None:
            base|=set(m.get("_question",[]) or [])
            if val: base|=set(m.get(val,[]) or [])
        elif a in ("location","radiation","character"):
            if val: pat[a]|=set(m.get(val,[]) or [])
        elif a=="severity":
            try: x=float(val); pat["severity"]|={"sev_mild" if x<=3 else "sev_moderate" if x<=6 else "sev_severe"}
            except: pass
        elif a=="onset":
            try: x=float(val); pat["onset"]|={"onset_sudden" if x>=5 else "onset_gradual"}
            except: pass
        elif a=="timing": pat["timing"]|={"tim_nocturnal"}
    return base,pat
allrows=[]
with open("data/ddxplus/release_test_patients.csv") as f:
    for row in csv.DictReader(f):
        tc=fr2cui.get(row["PATHOLOGY"])
        if tc in Pw and Pw[tc]: allrows.append((tc,ast.literal_eval(row["EVIDENCES"])))
W=0.05
def evalset(rows,subsets):
    acc={s:Counter() for s in subsets}; n=0
    for tc,evs in rows:
        base,pat=parse(evs); posm=base&all_evs
        if not posm: continue
        ans={e.split("_@_")[0] for e in evs}; neg=set(binary_evs)-ans
        patv={e:idf.get(e,1.0)**beta for e in posm}; pn=math.sqrt(sum(v*v for v in patv.values()))or 1e-9; nn=math.sqrt(len(neg))or 1e-9
        b0={}
        for d in dl:
            pr=Pw[d]; pc=sum(patv[e]*pr[e] for e in posm if e in pr)/(pn*DN[d])
            s=sig.get(d,{}); npen=sum(s.get(ev,0.0) for ev in neg)
            b0[d]=pc-0.7*(npen/(nn*DN[d]*(SIZE[d]/med)**0.5))
        base_sc={d:b0[d]+0.2*sum(ddS[d].get(d2,0)*b0[d2] for d2 in ddS[d]) for d in dl}
        # per-attribute overlap normalized per-patient
        ach={a:{} for a in ACT}
        for a in ACT:
            pv=pat.get(a,set())
            for d in dl:
                dv=dz.get(d,{}).get("attr",{}).get(a,set())
                ach[a][d]=sum(aidf(a,t) for t in (pv&dv))
            mx=max(ach[a].values()) or 1; ach[a]={d:v/mx for d,v in ach[a].items()}
        n+=1
        for sub in subsets:
            sc={d:base_sc[d]+W*sum(ach[a][d] for a in sub) for d in dl}
            rk=sorted(sc,key=lambda d:-sc[d]).index(tc)+1; C=acc[sub]
            for K in(1,5,10,20):
                if rk<=K: C[K]+=1
    return acc,n
subsets=[frozenset(c) for k in range(0,len(ACT)+1) for c in itertools.combinations(ACT,k)]
samp=random.sample(allrows,6000)
acc,n=evalset(samp,subsets)
res=sorted(subsets,key=lambda s:-acc[s][1])
print(f"random6000 N={n}  base(공집합): @1={100*acc[frozenset()][1]/n:.2f} @10={100*acc[frozenset()][10]/n:.2f}")
print("상위6 (by @1):")
for s in res[:6]: print(f"  {sorted(s)}: @1={100*acc[s][1]/n:.2f} @5={100*acc[s][5]/n:.2f} @10={100*acc[s][10]/n:.2f}")
top3=res[:3]
print("\n=== FULL ===",flush=True)
accf,nf=evalset(allrows,[frozenset()]+top3)
print(f"base: @1={100*accf[frozenset()][1]/nf:.2f} @5={100*accf[frozenset()][5]/nf:.2f} @10={100*accf[frozenset()][10]/nf:.2f} @20={100*accf[frozenset()][20]/nf:.2f}")
for s in top3:
    print(f"  {sorted(s)}: @1={100*accf[s][1]/nf:.2f} @5={100*accf[s][5]/nf:.2f} @10={100*accf[s][10]/nf:.2f} @20={100*accf[s][20]/nf:.2f}")
