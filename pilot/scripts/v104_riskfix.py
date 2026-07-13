"""(a) Risk-factor mapping fix: the 27 evidences with empty value_cuis (surgery,
hormones, immobilization, smoking, IV drugs, travel...) are wasted. Map each
evidence's question_en -> UMLS CUI via scispaCy (eval-adapter: benchmark code ->
standard, professor-endorsed). Add to patient pos. Applied on top of (c) neg-fix
(lam0.7,sizep0.5). random6000 -> top3 FULL."""
import sys,math,pickle,json,csv,ast,random,re
from collections import defaultdict,Counter
import spacy
from scispacy.linking import EntityLinker
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
# --- map empty-value_cuis evidences via scispaCy on question text ---
empty=[e for e in evmeta if not (value_cuis.get(e,{}).get("_question"))]
# clean phrase from question (strip "Do you / Have you ...")
def clean_q(q):
    q=q.lower()
    q=re.sub(r"^(do|did|have|are|is|were|was|has)\s+(you|your|there|the person)\s*","",q)
    q=re.sub(r"[?\.]","",q)
    return q.strip()
nlp=spacy.load("en_core_sci_lg")
nlp.add_pipe("scispacy_linker",config={"resolve_abbreviations":True,"linker_name":"umls","k":3,"threshold":0.85,"max_entities_per_mention":1})
phrases=[clean_q(evmeta[e].get("question_en","")) for e in empty]
ev2cuis={}
for e,doc in zip(empty,nlp.pipe(phrases,batch_size=64)):
    cs=set()
    for ent in doc.ents:
        for cui,sc in ent._.kb_ents[:1]: cs.add(cui)
    cs&=all_evs   # only keep if some disease has it (else useless)
    if cs: ev2cuis[e]=cs
print(f"매핑된 미매핑 evidence: {len(ev2cuis)}/{len(empty)} (all_evs와 교집합 있는 것)",flush=True)
for e in list(ev2cuis)[:10]: print(f"  {e}: {ev2cuis[e]} <- {evmeta[e].get('question_en','')[:45]}")
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
def patient_pos(evs,use_rf):
    pos=set()
    for ev in evs:
        b=ev.split("_@_")[0]; m=value_cuis.get(b,{})
        pos.update(m.get("_question",[]) or [])
        if "_@_" in ev: pos.update(m.get(ev.split("_@_")[1],[]) or [])
        if use_rf and b in ev2cuis: pos|=ev2cuis[b]
    return pos
LAM=0.7; SP=0.5
def evalrows(rows,use_rf):
    atK=Counter(); n=0
    for tc,evs in rows:
        pos=patient_pos(evs,use_rf); posm=pos&all_evs
        if not posm: continue
        ans={e.split("_@_")[0] for e in evs}; neg=set(binary_evs)-ans
        patv={e:idf.get(e,1.0)**beta for e in posm}; pn=math.sqrt(sum(v*v for v in patv.values()))or 1e-9
        nn=math.sqrt(len(neg))or 1e-9
        base={}
        for d in dl:
            pr=Pw[d]; pc=sum(patv[e]*pr[e] for e in posm if e in pr)/(pn*DN[d])
            s=sig.get(d,{}); npen=sum(s.get(ev,0.0) for ev in neg)
            base[d]=pc-LAM*(npen/(nn*DN[d]*(SIZE[d]/medsize)**SP))
        sc={d:base[d]+0.2*sum(ddS[d].get(d2,0)*base[d2] for d2 in ddS[d]) for d in dl}
        rk=sorted(sc,key=lambda d:-sc[d]).index(tc)+1; n+=1
        for K in(1,5,10,20):
            if rk<=K: atK[K]+=1
    return atK,n
allrows=[]
with open("data/ddxplus/release_test_patients.csv") as f:
    for row in csv.DictReader(f):
        tc=fr2cui.get(row["PATHOLOGY"])
        if tc in Pw and Pw[tc]: allrows.append((tc,ast.literal_eval(row["EVIDENCES"])))
samp=random.sample(allrows,6000)
for lab,rf in [("(c)만",False),("(c)+riskfactor",True)]:
    a,n=evalrows(samp,rf); print(f"random6000 {lab}: @1={100*a[1]/n:.2f} @5={100*a[5]/n:.2f} @10={100*a[10]/n:.2f}",flush=True)
print("=== FULL ===",flush=True)
for lab,rf in [("(c)만",False),("(c)+riskfactor",True)]:
    a,n=evalrows(allrows,rf); print(f"FULL {lab}: @1={100*a[1]/n:.2f} @5={100*a[5]/n:.2f} @10={100*a[10]/n:.2f} @20={100*a[20]/n:.2f}",flush=True)
