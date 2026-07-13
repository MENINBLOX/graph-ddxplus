"""BOUND qualified-edge eval (professor's design). Patient symptom s with bound
attrs matches disease phenotype p only when s_cui==p_cui; then per-attribute
agreement adds a bonus. Protocol: search 64 attr-subsets x 3 lambda on a RANDOM
sample, take top-3 by @1, confirm on FULL 134K. (no first-3000 bias)"""
import sys,math,pickle,json,csv,ast,random,itertools,argparse
from collections import defaultdict,Counter
sys.path.insert(0,"pilot/scripts")
from onlykg_eval_v71_selfaware import compute_idf,reweight,precompute_signal_v71,score
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
B=pickle.load(open("pilot/data/cache/v104_bound.pkl","rb")); dz=B["diseases"]
# disease bound index: per disease, per phenotype_cui -> merged attr sets
dzi={}
for d in dl:
    idx=defaultdict(lambda:defaultdict(set))
    for pc,at in dz.get(d,[]):
        for a,vs in at.items(): idx[pc][a]|=vs
    dzi[d]=idx
ATTRS=["location","radiation","character","severity","onset","timing"]
ATTR_SUF={"endroitducorps":"location","precis":"location","irrad":"radiation","intens":"severity","prurit":"severity","sev":"severity","soudain":"onset","carac":"character","aboy":"character","noct":"timing","nuit":"timing"}
def suf(eid):
    for s,a in ATTR_SUF.items():
        if eid.lower().endswith("_"+s) or eid.lower()==s: return a
    return None
def patient_parse(evs):
    base=set(); psym=defaultdict(lambda:{"cuis":set(),"attr":defaultdict(set)})
    for ev in evs:
        eid=ev.split("_@_")[0] if "_@_" in ev else ev
        val=ev.split("_@_")[1] if "_@_" in ev else None
        a=suf(eid); m=value_cuis.get(eid,{})
        if a is None:
            base.update(m.get("_question",[]) or [])
            if val is not None: base.update(m.get(val,[]) or [])
        else:
            parent=eid
            for s in ATTR_SUF:
                if eid.lower().endswith("_"+s): parent=eid[:-(len(s)+1)]; break
            pc=value_cuis.get(parent,{}).get("_question",[]) or []
            sym=psym[parent]; sym["cuis"].update(pc)
            if a in ("location","radiation","character"):
                if val is not None: sym["attr"][a].update(m.get(val,[]) or [])
            elif a=="severity":
                try: x=float(val); sym["attr"]["severity"].add("sev_mild" if x<=3 else "sev_moderate" if x<=6 else "sev_severe")
                except: pass
            elif a=="onset":
                try: x=float(val); sym["attr"]["onset"].add("onset_sudden" if x>=5 else "onset_gradual")
                except: pass
            elif a=="timing": sym["attr"]["timing"].add("timing_nocturnal")
    return base,psym
def load_rows(idxset=None,nmax=None):
    rows=[]
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i,row in enumerate(csv.DictReader(f)):
            if nmax and len(rows)>=nmax: break
            if idxset is not None and i not in idxset: continue
            tc=fr2cui.get(row["PATHOLOGY"])
            if tc not in Pw or not Pw[tc]: continue
            rows.append((tc,ast.literal_eval(row["EVIDENCES"])))
    return rows
def eval_rows(rows,configs):
    # precompute base + per-attr bound bonus per row, then score each config
    acc={c:Counter() for c in configs}; n=0
    for tc,evs in rows:
        base,psym=patient_parse(evs)
        posm=base&all_evs
        if not posm: continue
        ans={e.split("_@_")[0] for e in evs}
        b=score(posm,set(binary_evs)-ans,Pw,idf,beta,sig,1.0)
        sc0={d:b[d]+0.2*sum(ddS[d].get(d2,0)*b[d2] for d2 in ddS[d]) for d in dl}
        # bound bonus per attribute channel
        bb={a:{} for a in ATTRS}
        for a in ATTRS:
            for d in dl:
                tot=0.0; idxd=dzi[d]
                for parent,sym in psym.items():
                    sv=sym["attr"].get(a)
                    if not sv: continue
                    for pc in sym["cuis"]:
                        if pc in idxd and a in idxd[pc]:
                            tot+=len(sv & idxd[pc][a])
                bb[a][d]=tot
            mx=max(bb[a].values()) or 1
            bb[a]={d:v/mx for d,v in bb[a].items()}
        n+=1
        for (subset,lam) in configs:
            sc={d:sc0[d]+lam*sum(bb[a][d] for a in subset) for d in dl}
            rk=sorted(sc,key=lambda d:-sc[d]).index(tc)+1
            C=acc[(subset,lam)]
            for K in(1,5,10,20):
                if rk<=K: C[K]+=1
    return acc,n
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",default="search"); a=ap.parse_args()
    # all valid patient indices
    allrows=[]; 
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i,row in enumerate(csv.DictReader(f)):
            tc=fr2cui.get(row["PATHOLOGY"])
            if tc in Pw and Pw[tc]: allrows.append((i,tc,ast.literal_eval(row["EVIDENCES"])))
    print(f"total valid {len(allrows)}",flush=True)
    samp=random.sample(allrows,6000)
    rows=[(tc,evs) for _,tc,evs in samp]
    subsets=[]
    ai=ATTRS
    for k in range(1,len(ai)+1):
        for combo in itertools.combinations(ai,k): subsets.append(frozenset(combo))
    configs=[(s,lam) for s in subsets for lam in (0.02,0.05,0.1)]
    configs.append((frozenset(),0.0))
    print(f"random6000: {len(configs)} configs...",flush=True)
    acc,n=eval_rows(rows,configs)
    res=sorted(configs,key=lambda c:-acc[c][1])
    print(f"\nN={n}  base@1={100*acc[(frozenset(),0.0)][1]/n:.2f} @10={100*acc[(frozenset(),0.0)][10]/n:.2f}")
    print("=== random6000 상위10 (by @1) ===")
    for c in res[:10]:
        C=acc[c]; print(f"  {sorted(c[0])} lam={c[1]}: @1={100*C[1]/n:.2f} @5={100*C[5]/n:.2f} @10={100*C[10]/n:.2f}")
    top3=res[:3]
    # full eval top3 + base
    print("\n=== FULL 134K 상위3 + base 확정 ===",flush=True)
    fullrows=[(tc,evs) for _,tc,evs in allrows]
    accf,nf=eval_rows(fullrows,top3+[(frozenset(),0.0)])
    Cb=accf[(frozenset(),0.0)]; print(f"N={nf}")
    print(f"  base                     : @1={100*Cb[1]/nf:.2f} @5={100*Cb[5]/nf:.2f} @10={100*Cb[10]/nf:.2f} @20={100*Cb[20]/nf:.2f}")
    for c in top3:
        C=accf[c]; print(f"  {sorted(c[0])} lam={c[1]}: @1={100*C[1]/nf:.2f} @5={100*C[5]/nf:.2f} @10={100*C[10]/nf:.2f} @20={100*C[20]/nf:.2f}")
main()
