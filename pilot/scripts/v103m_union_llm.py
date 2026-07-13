"""@10→100 via KG-top5 ∪ LLM-differential-top5 (user-authorized #4 relaxation).
LLM gets patient findings + the 49 candidate disease names (names extractable
per #3), returns its ranked differential. Union with KG top-5 → ≤10 candidates.
Measure whether GT lands in the union (recall@union)."""
import sys,math,pickle,json,csv,ast,re
from collections import defaultdict,Counter
sys.path.insert(0,"pilot/scripts")
import onlykg_eval_v71_selfaware as V71
from onlykg_eval_v71_selfaware import compute_idf,reweight,precompute_signal_v71,score
N=1500
dcs,pats,binary_evs=V71.load_ddxplus_full(N)
value_cuis=json.load(open("/windows/data/medkg/kg/ddxplus_evidence_value_cuis.json"))
icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
cui2name={info["cui"]:dn for dn,info in icd.items() if "cui" in info}
evmeta=json.load(open("data/ddxplus/release_evidences.json"))
def L(f): return pickle.load(open(f"pilot/data/cache/{f}","rb"))
files=["v103h_exh_kg.pkl","v103ddx49_sci_kg.pkl","v103pres_ddx49_sci_kg.pkl","v103sci_ddx49_kg.pkl","v103i_clean_kg.pkl","v103j_exp_kg.pkl"]
wt={"v103i_clean_kg.pkl":3,"v103j_exp_kg.pkl":3}
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
disease_names=sorted({cui2name[d] for d in Pw if d in cui2name})
# patient findings render via csv (need raw evidences)
def render(evs):
    out=[]
    for ev in evs:
        if "_@_" in ev:
            base,val=ev.split("_@_",1); m=evmeta.get(base,{})
            vm=m.get("value_meaning",{}).get(val,{}); en=vm.get("en",val) if isinstance(vm,dict) else val
            if en and en!="nowhere": out.append(f"{m.get('question_en','')}: {en}")
        else: out.append(evmeta.get(ev,{}).get("question_en",ev))
    return list(dict.fromkeys(out))
raws=[]
with open("data/ddxplus/release_test_patients.csv") as f:
    for row in csv.DictReader(f):
        if len(raws)>=N: break
        raws.append(ast.literal_eval(row["EVIDENCES"]))
# KG top5 per patient + build prompts
tasks=[]
namelist="\n".join(f"- {n}" for n in disease_names)
for (tc,pos,neg),evs in zip(pats,raws):
    if tc not in Pw or not Pw[tc]: continue
    if not (pos&all_evs): continue
    sc=score(pos&all_evs,neg,Pw,idf,beta,sig,1.0)
    kg=sorted(sc,key=lambda d:-sc[d])
    kg10=kg[:10]; kg5=kg[:5]
    fl="\n".join(f"- {x}" for x in render(evs))
    prompt=(f"A patient presents with:\n{fl}\n\nFrom this candidate disease list:\n{namelist}\n\n"
            f"List the 5 MOST LIKELY diagnoses, most likely first, one per line as 'N. <exact disease name>'.")
    tasks.append((tc,kg10,kg5,prompt))
print(f"{len(tasks)} patients. KG-only @10 baseline...",flush=True)
base10=sum(1 for tc,kg10,_,_ in tasks if tc in kg10)
from vllm import LLM,SamplingParams
llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
outs=llm.chat([[{"role":"user","content":p}] for _,_,_,p in tasks],SamplingParams(temperature=0.0,max_tokens=256),use_tqdm=True)
name2cui={n:d for d in Pw if d in cui2name for n in [cui2name[d]]}
union_hit=0; llm_hit=0
for (tc,kg10,kg5,_),o in zip(tasks,outs):
    txt=o.outputs[0].text
    picks=[]
    for line in txt.splitlines():
        for n in disease_names:
            if n.lower() in line.lower() and name2cui[n] not in picks:
                picks.append(name2cui[n]); break
        if len(picks)>=5: break
    llm5=picks[:5]
    union=set(kg5)|set(llm5)
    if tc in union: union_hit+=1
    if tc in llm5: llm_hit+=1
n=len(tasks)
print(f"\nN={n}")
print(f"KG-only @10        : {100*base10/n:.2f}")
print(f"LLM-diff @5        : {100*llm_hit/n:.2f}")
print(f"KG5 ∪ LLM5 (≤10)   : {100*union_hit/n:.2f}")
