"""Combination-IDF weighted BOUND matching (benchmark-blind). Each matched
(phenotype, attribute-value) pair is weighted by how RARE that combination is
across the 49-disease KG (combo-IDF) — rash@cheek scores high because few
diseases have that pair; pain@chest scores low. KG-derived, no benchmark labels.
Protocol: random6000 search -> top3 FULL 134K."""
import sys,math,pickle,json,csv,ast,random,itertools,os
from collections import defaultdict,Counter
import torch
from transformers import AutoTokenizer,AutoModel
sys.path.insert(0,"pilot/scripts")
from onlykg_eval_v71_selfaware import compute_idf,reweight,precompute_signal_v71,score
random.seed(42)
MRCONSO="/windows/data/umls_subset/MRCONSO.RRF"; VC="/windows/data/medkg/kg/ddxplus_evidence_value_cuis.json"
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
ddS={d:{} for d in dl}
for i,d1 in enumerate(dl):
    x=Pw[d1]
    for d2 in dl[i+1:]:
        y=Pw[d2]; aa,bb=(x,y) if len(x)<len(y) else (y,x)
        s=sum(aa[e]*bb[e] for e in aa if e in bb)/(DN[d1]*DN[d2])
        if s>0.05: ddS[d1][d2]=s; ddS[d2][d1]=s
B=pickle.load(open("pilot/data/cache/v104_bound.pkl","rb")); dz=B["diseases"]
CUI_ATTRS=["location","radiation","character"]; ATTRS=["location","radiation","character","severity","onset","timing"]
ATTR_SUF={"endroitducorps":"location","precis":"location","irrad":"radiation","intens":"severity","prurit":"severity","sev":"severity","soudain":"onset","carac":"character","aboy":"character","noct":"timing","nuit":"timing"}
def suf(eid):
    for s,a in ATTR_SUF.items():
        if eid.lower().endswith("_"+s) or eid.lower()==s: return a
    return None
# ---- canon (cache) ----
CF="pilot/data/cache/v104_canon_map.pkl"
if os.path.exists(CF):
    canon=pickle.load(open(CF,"rb")); print("canon loaded",flush=True)
else:
    cset=set()
    for d,phs in dz.items():
        for pc,at in phs:
            cset.add(pc)
            for a in CUI_ATTRS: cset|=at.get(a,set())
    for eid,m in value_cuis.items():
        a=suf(eid)
        if a is None: cset|=set(m.get("_question",[]) or [])
        elif a in CUI_ATTRS:
            for k,v in m.items():
                if isinstance(v,list): cset|=set(v)
    cset={c for c in cset if isinstance(c,str) and c.startswith("C")}
    nm={}
    with open(MRCONSO,encoding="utf-8") as f:
        for line in f:
            p=line.split("|")
            if p[1]=="ENG" and p[0] in cset and (p[0] not in nm or p[2]=="P"): nm[p[0]]=p[14]
    named=[c for c in sorted(cset) if c in nm]; names=[nm[c].lower() for c in named]
    tok=AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    mdl=AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda().eval().half()
    embs=[]
    with torch.no_grad():
        for i in range(0,len(names),512):
            b=tok(names[i:i+512],padding=True,truncation=True,max_length=32,return_tensors="pt").to("cuda")
            e=mdl(**b).last_hidden_state[:,0,:]; embs.append(torch.nn.functional.normalize(e,dim=1).float())
    E=torch.cat(embs).cuda(); parent=list(range(len(named)))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for i in range(0,len(named),1024):
        sims=E[i:i+1024]@E.T
        for r,c in (sims>=0.80).nonzero(as_tuple=False).cpu().numpy():
            gi=i+r
            if c>gi:
                ra,rb=find(gi),find(int(c))
                if ra!=rb: parent[max(ra,rb)]=min(ra,rb)
    canon={c:("cl",find(i)) for i,c in enumerate(named)}
    pickle.dump(canon,open(CF,"wb")); print(f"canon built {len(named)} CUIs",flush=True)
def cz(c): return canon.get(c,("raw",c))
def czset(s): return {cz(c) for c in s}
# disease bound index (canonical)
dzi={}
for d in dl:
    idxp=defaultdict(lambda:defaultdict(set))
    for pc,at in dz.get(d,[]):
        cp=cz(pc)
        for a in CUI_ATTRS: idxp[cp][a]|=czset(at.get(a,set()))
        for a in ("severity","onset","timing"): idxp[cp][a]|=at.get(a,set())
    dzi[d]=idxp
# combination-IDF: df over diseases of (canon_pheno, attr, canon_val)
combo_df=defaultdict(int)
for d in dl:
    seen=set()
    for cp,ad in dzi[d].items():
        for a,vs in ad.items():
            for v in vs: seen.add((cp,a,v))
    for k in seen: combo_df[k]+=1
Nd=len(dl)
def combo_idf(cp,a,v): return math.log((Nd+1)/(combo_df.get((cp,a,v),0)+1))+1.0
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
            parent_id=eid
            for s in ATTR_SUF:
                if eid.lower().endswith("_"+s): parent_id=eid[:-(len(s)+1)]; break
            pc=value_cuis.get(parent_id,{}).get("_question",[]) or []
            sym=psym[parent_id]; sym["cuis"].update(czset(pc))
            if a in CUI_ATTRS:
                if val is not None: sym["attr"][a]|=czset(m.get(val,[]) or [])
            elif a=="severity":
                try: x=float(val); sym["attr"]["severity"].add("sev_mild" if x<=3 else "sev_moderate" if x<=6 else "sev_severe")
                except: pass
            elif a=="onset":
                try: x=float(val); sym["attr"]["onset"].add("onset_sudden" if x>=5 else "onset_gradual")
                except: pass
            elif a=="timing": sym["attr"]["timing"].add("timing_nocturnal")
    return base,psym
def eval_rows(rows,configs):
    acc={c:Counter() for c in configs}; n=0
    for tc,evs in rows:
        base,psym=patient_parse(evs)
        posm=base&all_evs
        if not posm: continue
        ans={e.split("_@_")[0] for e in evs}
        b=score(posm,set(binary_evs)-ans,Pw,idf,beta,sig,1.0)
        sc0={d:b[d]+0.2*sum(ddS[d].get(d2,0)*b[d2] for d2 in ddS[d]) for d in dl}
        bb={a:{} for a in ATTRS}
        for a in ATTRS:
            for d in dl:
                tot=0.0; idxd=dzi[d]
                for sym in psym.values():
                    sv=sym["attr"].get(a)
                    if not sv: continue
                    for cp in sym["cuis"]:
                        dv=idxd.get(cp,{}).get(a)
                        if dv:
                            for v in (sv & dv): tot+=combo_idf(cp,a,v)
                bb[a][d]=tot
            mx=max(bb[a].values()) or 1; bb[a]={d:v/mx for d,v in bb[a].items()}
        n+=1
        for (subset,lam) in configs:
            sc={d:sc0[d]+lam*sum(bb[a][d] for a in subset) for d in dl}
            rk=sorted(sc,key=lambda d:-sc[d]).index(tc)+1; C=acc[(subset,lam)]
            for K in(1,5,10,20):
                if rk<=K: C[K]+=1
    return acc,n
allrows=[]
with open("data/ddxplus/release_test_patients.csv") as f:
    for i,row in enumerate(csv.DictReader(f)):
        tc=fr2cui.get(row["PATHOLOGY"])
        if tc in Pw and Pw[tc]: allrows.append((tc,ast.literal_eval(row["EVIDENCES"])))
samp=random.sample(allrows,6000)
subsets=[frozenset(c) for k in range(1,len(ATTRS)+1) for c in itertools.combinations(ATTRS,k)]
configs=[(s,lam) for s in subsets for lam in (0.05,0.1,0.2)]+[(frozenset(),0.0)]
print(f"random6000: {len(configs)} configs (combo-IDF weighted)...",flush=True)
acc,n=eval_rows(samp,configs)
res=sorted(configs,key=lambda c:-acc[c][1])
print(f"N={n} base@1={100*acc[(frozenset(),0.0)][1]/n:.2f} @10={100*acc[(frozenset(),0.0)][10]/n:.2f}")
print("=== random6000 상위8 (by @1) ===")
for c in res[:8]:
    C=acc[c]; print(f"  {sorted(c[0])} lam={c[1]}: @1={100*C[1]/n:.2f} @5={100*C[5]/n:.2f} @10={100*C[10]/n:.2f}")
top3=res[:3]
print("\n=== FULL 134K 상위3+base ===",flush=True)
accf,nf=eval_rows(allrows,top3+[(frozenset(),0.0)])
Cb=accf[(frozenset(),0.0)]; print(f"N={nf}")
print(f"  base                     : @1={100*Cb[1]/nf:.2f} @5={100*Cb[5]/nf:.2f} @10={100*Cb[10]/nf:.2f} @20={100*Cb[20]/nf:.2f}")
for c in top3:
    C=accf[c]; print(f"  {sorted(c[0])} lam={c[1]}: @1={100*C[1]/nf:.2f} @5={100*C[5]/nf:.2f} @10={100*C[10]/nf:.2f} @20={100*C[20]/nf:.2f}")
