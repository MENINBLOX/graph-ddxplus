"""Per-attribute faithfulness (groundedness) of v106 IE — NLI sentence-level
max-entailment (recognized hallucination metric). Breaks the aggregate attr
faithfulness down by the 4 core attributes (location/severity/onset/character),
so attributes WITHOUT a recognized gold slot (onset, character) still get a
recognized intrinsic number: does the source entail the extracted value?
Faithfulness = PRECISION/groundedness only (not recall, not value-correctness)."""
import json, glob, re, torch, statistics
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MID="FacebookAI/roberta-large-mnli"
tok=AutoTokenizer.from_pretrained(MID)
mdl=AutoModelForSequenceClassification.from_pretrained(MID,dtype=torch.float16).to("cuda").eval()
ENT=[i for i,l in mdl.config.id2label.items() if l.lower().startswith("entail")][0]
CORE=["location","severity","onset","character"]
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

P=[];H=[];cid=[];kind=[]; n=0
for fp in glob.glob("pilot/data/cache/v106_grounded_ie/*.json"):
    o=json.load(open(fp)); ss=SRC.get(o["cui"])
    if not ss: continue
    for f in o["findings"]:
        for a in CORE:
            v=str(f.get(a,"")).strip()
            if not v: continue
            for s in ss: P.append(s);H.append(hyp(f["name"],a,v));cid.append(n)
            kind.append(a); n+=1
sc=ent(P,H)
mx=[0.0]*n
for c,x in zip(cid,sc):
    if x>mx[c]: mx[c]=x
by=defaultdict(list)
for i in range(n): by[kind[i]].append(mx[i])
print("=== v106 per-attribute faithfulness (NLI, source-entailment, groundedness only) ===")
print(f"{'attribute':<12}{'n':>5}{'faithful%(>0.5)':>16}{'mean':>8}")
for a in CORE:
    arr=by[a]; pct=100*sum(1 for x in arr if x>0.5)/max(len(arr),1)
    print(f"{a:<12}{len(arr):>5}{pct:>15.0f}%{statistics.mean(arr) if arr else 0:>8.2f}")
allv=[x for a in CORE for x in by[a]]
print(f"{'ALL 4':<12}{len(allv):>5}{100*sum(1 for x in allv if x>0.5)/max(len(allv),1):>15.0f}%{statistics.mean(allv):>8.2f}")
