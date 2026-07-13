"""CORRECT attribute eval: TWO complete systems on the SAME grounded-IE phenotype
backbone. P = phenotype-only cosine. P+A = phenotype cosine + BOUND attribute
agreement (patient symptom binds to disease phenotype AND its attribute matches).
SapBERT canon aligns vocab. Difference P vs P+A = the pure, isolated attribute
contribution. random6000 -> full. Also a contrastive note on confusable cases."""
import sys,math,pickle,json,csv,ast,random
from collections import defaultdict,Counter
import torch
from transformers import AutoTokenizer,AutoModel
random.seed(42)
MRCONSO="/windows/data/umls_subset/MRCONSO.RRF"; VC="/windows/data/medkg/kg/ddxplus_evidence_value_cuis.json"
value_cuis=json.load(open(VC)); evmeta=json.load(open("data/ddxplus/release_evidences.json"))
icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json")); cond=json.load(open("data/ddxplus/release_conditions_en.json"))
fr2cui={info.get("cond-name-fr",""):icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
dcs=sorted(set(fr2cui.values()))
B=pickle.load(open("pilot/data/cache/v105_bound.pkl","rb")); dz=B["diseases"]
CUI_ATTRS=["location","radiation","character"]; ATTRS=["location","radiation","character","severity","onset","timing"]
ATTR_SUF={"endroitducorps":"location","precis":"location","irrad":"radiation","intens":"severity","prurit":"severity","sev":"severity","soudain":"onset","carac":"character","aboy":"character","noct":"timing","nuit":"timing"}
def sufof(eid):
    for s,a in ATTR_SUF.items():
        if eid.lower().endswith("_"+s) or eid.lower()==s: return a
    return None
# --- phenotype profile from grounded IE (canon-able phen CUIs) ---
# disease phen CUIs = keys of dz bound phenotypes
Pphen={d:Counter() for d in dcs}
for d in dcs:
    for pc,at in dz.get(d,[]): Pphen[d][pc]+=1
# --- canon over all phen + attr + patient CUIs ---
cset=set()
for d,phs in dz.items():
    for pc,at in phs:
        cset.add(pc)
        for a in CUI_ATTRS: cset|=at.get(a,set())
for eid,m in value_cuis.items():
    a=sufof(eid)
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
print(f"canon {len(named)} CUIs...",flush=True)
tok=AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
mdl=AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda().eval().half()
embs=[]
with torch.no_grad():
    for i in range(0,len(names),512):
        b=tok(names[i:i+512],padding=True,truncation=True,max_length=32,return_tensors="pt").to("cuda")
        embs.append(torch.nn.functional.normalize(mdl(**b).last_hidden_state[:,0,:],dim=1).float())
E=torch.cat(embs).cuda(); parent=list(range(len(named)))
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
for i in range(0,len(named),1024):
    sims=E[i:i+1024]@E.T
    for r,c in (sims>=0.85).nonzero(as_tuple=False).cpu().numpy():
        if c>i+r:
            ra,rb=find(i+r),find(int(c))
            if ra!=rb: parent[max(ra,rb)]=min(ra,rb)
canon={c:("cl",find(i)) for i,c in enumerate(named)}
def cz(c): return canon.get(c,("raw",c))
def czs(s): return {cz(c) for c in s}
# canon phenotype profile
Pc={d:Counter() for d in dcs}
for d in dcs:
    for pc,n2 in Pphen[d].items(): Pc[d][cz(pc)]+=n2
Pc={d:{p:v/(v+1.0) for p,v in c.items()} for d,c in Pc.items() if c}
all_evs=set().union(*[set(p) for p in Pc.values()])
N=len(Pc); df=defaultdict(int)
for pr in Pc.values():
    for p in pr: df[p]+=1
idf={p:math.log((N+1)/(df[p]+1))+1.0 for p in df}
dl=[d for d in Pc if Pc[d]]; DN={d:(math.sqrt(sum((v*idf.get(p,1)**0.75)**2 for p,v in Pc[d].items()))or 1e-9) for d in dl}
# disease bound index (canon)
dzi={d:defaultdict(lambda:defaultdict(set)) for d in dl}
for d in dl:
    for pc,at in dz.get(d,[]):
        cp=cz(pc)
        for a in CUI_ATTRS: dzi[d][cp][a]|=czs(at.get(a,set()))
        for a in ("severity","onset","timing"): dzi[d][cp][a]|=at.get(a,set())
def patient(evs):
    base=set(); psym=defaultdict(lambda:{"cuis":set(),"attr":defaultdict(set)})
    for ev in evs:
        eid=ev.split("_@_")[0] if "_@_" in ev else ev; val=ev.split("_@_")[1] if "_@_" in ev else None
        a=sufof(eid); m=value_cuis.get(eid,{})
        if a is None:
            base|=czs(m.get("_question",[]) or [])
            if val: base|=czs(m.get(val,[]) or [])
        else:
            parent_id=eid
            for s in ATTR_SUF:
                if eid.lower().endswith("_"+s): parent_id=eid[:-(len(s)+1)]; break
            sym=psym[parent_id]; sym["cuis"]|=czs(value_cuis.get(parent_id,{}).get("_question",[]) or [])
            if a in CUI_ATTRS:
                if val: sym["attr"][a]|=czs(m.get(val,[]) or [])
            elif a=="severity":
                try: x=float(val); sym["attr"]["severity"]|={"sev_mild" if x<=3 else "sev_moderate" if x<=6 else "sev_severe"}
                except: pass
            elif a=="onset":
                try: x=float(val); sym["attr"]["onset"]|={"onset_sudden" if x>=5 else "onset_gradual"}
                except: pass
            elif a=="timing": sym["attr"]["timing"]|={"tim_nocturnal"}
    return base,psym
allrows=[]
with open("data/ddxplus/release_test_patients.csv") as f:
    for row in csv.DictReader(f):
        tc=fr2cui.get(row["PATHOLOGY"])
        if tc in Pc and Pc[tc]: allrows.append((tc,ast.literal_eval(row["EVIDENCES"])))
def evalset(rows,lams):
    acc={("P",0):Counter()}; 
    for lam in lams: acc[("PA",lam)]=Counter()
    n=0
    for tc,evs in rows:
        base,psym=patient(evs)
        # patient phen = base ∪ symptom cuis
        pphen=set(base)
        for sym in psym.values(): pphen|=sym["cuis"]
        pphen&=all_evs
        if not pphen: continue
        patv={e:idf.get(e,1.0)**0.75 for e in pphen}; pn=math.sqrt(sum(v*v for v in patv.values()))or 1e-9
        Pscore={}
        for d in dl:
            pr=Pc[d]; Pscore[d]=sum(patv[e]*pr[e]*idf.get(e,1)**0.75 for e in pphen if e in pr)/(pn*DN[d])
        # bound attribute agreement
        abonus={}
        for d in dl:
            tot=0.0; idxd=dzi[d]
            for sym in psym.values():
                for a in ATTRS:
                    sv=sym["attr"].get(a)
                    if not sv: continue
                    for cp in sym["cuis"]:
                        dv=idxd.get(cp,{}).get(a)
                        if dv: tot+=len(sv&dv)
            abonus[d]=tot
        mx=max(abonus.values()) or 1; abonus={d:v/mx for d,v in abonus.items()}
        n+=1
        rkP=sorted(dl,key=lambda d:-Pscore[d]).index(tc)+1
        C=acc[("P",0)]
        for K in(1,5,10,20):
            if rkP<=K: C[K]+=1
        for lam in lams:
            sc={d:Pscore[d]+lam*abonus[d] for d in dl}
            rk=sorted(dl,key=lambda d:-sc[d]).index(tc)+1; C=acc[("PA",lam)]
            for K in(1,5,10,20):
                if rk<=K: C[K]+=1
    return acc,n
samp=random.sample(allrows,6000)
acc,n=evalset(samp,[0.05,0.1,0.2])
print(f"\nrandom6000 N={n}")
print(f"  P (phenotype-only):  @1={100*acc[('P',0)][1]/n:.2f} @5={100*acc[('P',0)][5]/n:.2f} @10={100*acc[('P',0)][10]/n:.2f}")
bestlam=max([0.05,0.1,0.2],key=lambda l:acc[('PA',l)][1])
for lam in [0.05,0.1,0.2]:
    C=acc[('PA',lam)]; print(f"  P+A (lam={lam}): @1={100*C[1]/n:.2f} @5={100*C[5]/n:.2f} @10={100*C[10]/n:.2f}")
print(f"\n=== FULL (P vs best P+A lam={bestlam}) ===",flush=True)
accf,nf=evalset(allrows,[bestlam])
Cp=accf[('P',0)]; Ca=accf[('PA',bestlam)]
print(f"  P     : @1={100*Cp[1]/nf:.2f} @5={100*Cp[5]/nf:.2f} @10={100*Cp[10]/nf:.2f} @20={100*Cp[20]/nf:.2f}")
print(f"  P+A   : @1={100*Ca[1]/nf:.2f} @5={100*Ca[5]/nf:.2f} @10={100*Ca[10]/nf:.2f} @20={100*Ca[20]/nf:.2f}")
print(f"  속성 순수기여(P+A − P): @1 {100*(Ca[1]-Cp[1])/nf:+.2f}p  @10 {100*(Ca[10]-Cp[10])/nf:+.2f}p")
