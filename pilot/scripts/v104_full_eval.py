"""Full DDXPlus eval of the best config: union base (cosine+neg+spreading) +
micro-lambda attribute refinement. Patient attributes extracted inline (no
scispaCy needed patient-side); disease attr vectors reused from v104 pkl cache."""
import sys,math,pickle,json,csv,ast,argparse
from collections import defaultdict,Counter
sys.path.insert(0,"pilot/scripts")
from onlykg_eval_v71_selfaware import compute_idf,reweight,precompute_signal_v71,score
ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=200000); ap.add_argument("--lam",type=float,default=0.02); a=ap.parse_args()
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
# disease attr vectors
Dv=pickle.load(open("pilot/data/cache/v104_attr_vectors.pkl","rb")); dis=Dv["diseases"]
ATTR_SUFFIX={"endroitducorps":"location","precis":"location","irrad":"radiation","intens":"severity","prurit":"severity","sev":"severity","soudain":"onset","carac":"character","aboy":"character","noct":"timing","nuit":"timing"}
def suffix_of(eid):
    for suf,at in ATTR_SUFFIX.items():
        if eid.lower().endswith("_"+suf) or eid.lower()==suf: return at
    return None
def patient_attr(evs):
    base=set(); chan=defaultdict(set)
    for ev in evs:
        eid=ev.split("_@_")[0] if "_@_" in ev else ev
        val=ev.split("_@_")[1] if "_@_" in ev else None
        at=suffix_of(eid); m=value_cuis.get(eid,{})
        if at is None:
            base.update(m.get("_question",[]) or [])
            if val is not None: base.update(m.get(val,[]) or [])
        elif at in ("location","radiation","character"):
            if val is not None: chan[at].update(m.get(val,[]) or [])
        elif at=="severity":
            try: x=float(val); chan["severity"].add("sev_mild" if x<=3 else("sev_moderate" if x<=6 else "sev_severe"))
            except: pass
        elif at=="onset":
            try: x=float(val); chan["onset"].add("onset_sudden" if x>=5 else "onset_gradual")
            except: pass
        elif at=="timing": chan["timing"].add("timing_nocturnal")
    return base,chan
CHANS=["location","severity","onset","character"]
# disease attr idf
df=defaultdict(int)
for d in dl:
    toks=set()
    for c in CHANS:
        for t in dis.get(d,{}).get("attr",{}).get(c,set()): toks.add((c,t))
    for t in toks: df[t]+=1
Na=len(dl); aidf={t:math.log((Na+1)/(df[t]+1))+1.0 for t in df}
atKa={"base":Counter(),"loc":Counter(),"locsoc":Counter()}; n=0
import os
with open("data/ddxplus/release_test_patients.csv") as f:
    for row in csv.DictReader(f):
        if n>=a.n: break
        tc=fr2cui.get(row["PATHOLOGY"])
        if tc not in Pw or not Pw[tc]: continue
        evs=ast.literal_eval(row["EVIDENCES"])
        base,chan=patient_attr(evs)
        posm=base&all_evs
        if not posm: continue
        b=score(posm,set(binary_evs)-{e.split("_@_")[0] for e in evs},Pw,idf,beta,sig,1.0)
        sc={d:b[d]+0.2*sum(ddS[d].get(d2,0)*b[d2] for d2 in ddS[d]) for d in dl}
        # attribute refinement (two variants)
        def refine(usechans):
            out=dict(sc)
            for c in usechans:
                cv={}
                for d in dl:
                    dd=dis.get(d,{}).get("attr",{}).get(c,set())
                    cv[d]=len(chan.get(c,set())&dd)
                mx=max(cv.values()) or 1
                for d in dl: out[d]+=a.lam*cv[d]/mx
            return out
        n+=1
        for key,uc in [("base",[]),("loc",["location"]),("locsoc",CHANS)]:
            s2=refine(uc); rk=sorted(s2,key=lambda d:-s2[d]).index(tc)+1
            for K in(1,5,10,20):
                if rk<=K: atKa[key][K]+=1
print(f"N={n}")
for key,nm in [("base","base(속성0)"),("loc","+location"),("locsoc","+loc+sev+ons+char")]:
    c=atKa[key]; print(f"{nm:22s}: @1={100*c[1]/n:.2f} @5={100*c[5]/n:.2f} @10={100*c[10]/n:.2f} @20={100*c[20]/n:.2f}")
