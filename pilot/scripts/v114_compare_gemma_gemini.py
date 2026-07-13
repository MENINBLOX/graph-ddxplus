"""Matched comparison gemma-4-E4B vs gemini-3.1-pro-preview on the SAME diseases
Gemini finished (14/49). Identical prompt/source/post-processing — only MODEL differs.
Per-attribute faithfulness (NLI source-entailment) + fill stats + severity-binding
check. Tests whether the severity mis-binding (gemma 18%) is model-capacity or prompt."""
import json, glob, os, re, torch, statistics
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

GEM_DIR="pilot/data/cache/v106_gemini_ie"
GMA_DIR="pilot/data/cache/v106_grounded_ie"
CUIS=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{GEM_DIR}/*.json"))  # the 14 gemini did
CORE=["location","severity","onset","character"]

MID="FacebookAI/roberta-large-mnli"
tok=AutoTokenizer.from_pretrained(MID,use_fast=False)
mdl=AutoModelForSequenceClassification.from_pretrained(MID,dtype=torch.float16).to("cuda").eval()
ENT=[i for i,l in mdl.config.id2label.items() if l.lower().startswith("entail")][0]
def hyp(name,a,v):
    T={"location":f"The {name} is located in the {v}.","onset":f"The {name} has a {v} onset.",
       "severity":f"The {name} is {v}.","character":f"The {name} is {v}."}
    return T.get(a,f"The {name} is {v}.")
def sents(t):
    t=re.sub(r'\s+',' ',t); return [s.strip() for s in re.split(r'(?<=[.;])\s+',t) if len(s.strip())>15][:30]
SRC={fp.split("/")[-1][:-4]:sents(open(fp).read()[:2200]) for fp in glob.glob("pilot/data/cache/v105_sources/*.txt")}

@torch.no_grad()
def ent(prems,hyps,bs=128):
    out=[]
    for i in range(0,len(prems),bs):
        enc=tok(prems[i:i+bs],hyps[i:i+bs],return_tensors="pt",padding="max_length",truncation=True,max_length=96).to("cuda")
        out+=[float(x) for x in torch.softmax(mdl(**enc).logits.float(),dim=1)[:,ENT]]
    return out

def score_dir(d):
    P=[];H=[];cid=[];kind=[];n=0; nf=0; fill=defaultdict(int)
    for c in CUIS:
        fp=f"{d}/{c}.json"
        if not os.path.exists(fp): continue
        o=json.load(open(fp)); ss=SRC.get(c)
        for f in o.get("findings",[]):
            nf+=1
            for a in CORE:
                v=str(f.get(a,"")).strip()
                if not v: continue
                fill[a]+=1
                if ss:
                    for s in ss: P.append(s);H.append(hyp(f["name"],a,v));cid.append(n)
                    kind.append(a); n+=1
    sc=ent(P,H) if P else []
    mx=[0.0]*n
    for c,x in zip(cid,sc):
        if x>mx[c]: mx[c]=x
    by=defaultdict(list)
    for i in range(n): by[kind[i]].append(mx[i])
    return nf,fill,by

print(f"Matched diseases (both models): {len(CUIS)}\n")
for label,d in [("gemma-4-E4B",GMA_DIR),("gemini-3.1-pro-preview",GEM_DIR)]:
    nf,fill,by=score_dir(d)
    print(f"=== {label} ===  findings={nf}")
    print(f"  {'attr':<10}{'fill':>5}{'faithful%':>11}{'mean':>7}")
    for a in CORE:
        arr=by[a]; pct=100*sum(1 for x in arr if x>0.5)/max(len(arr),1)
        print(f"  {a:<10}{fill[a]:>5}{pct:>10.0f}%{(statistics.mean(arr) if arr else 0):>7.2f}")
    print()
