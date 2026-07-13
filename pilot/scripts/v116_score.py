"""Score severity prompt-variants vs v106 baseline by per-attribute NLI faithfulness.
Key: severity must IMPROVE faithful% WITHOUT collapsing fill to ~0 (suppressing all
severity trivially inflates %). Track fill / faithful% / faithful_n. Also watch that
finding count + location/character faithfulness are not damaged (collateral)."""
import json, glob, re, torch, statistics
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DIRS=["v106_grounded_ie","v115_R2","v116_R4","v116_R5"]
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

def score(d):
    P=[];H=[];cid=[];kind=[];n=0;nf=0;fill=defaultdict(int)
    for fp in glob.glob(f"pilot/data/cache/{d}/*.json"):
        o=json.load(open(fp)); ss=SRC.get(o["cui"])
        if not ss: continue
        for f in o.get("findings",[]):
            nf+=1
            for a in CORE:
                v=str(f.get(a,"")).strip()
                if not v: continue
                fill[a]+=1
                for s in ss: P.append(s);H.append(hyp(f["name"],a,v));cid.append(n)
                kind.append(a); n+=1
    mx=[0.0]*n
    for c,x in zip(cid,ent(P,H) if P else []):
        if x>mx[c]: mx[c]=x
    by=defaultdict(list)
    for i in range(n): by[kind[i]].append(mx[i])
    return nf,fill,by

print(f"{'variant':<16}{'find':>5}{'sev_fill':>9}{'sev_f%':>7}{'sev_fN':>7}{'loc_f%':>7}{'char_f%':>7}")
for d in DIRS:
    if not glob.glob(f"pilot/data/cache/{d}/*.json"): print(f"{d:<16} (missing)"); continue
    nf,fill,by=score(d)
    def fp(a):
        arr=by[a]; return (100*sum(1 for x in arr if x>0.5)/max(len(arr),1), sum(1 for x in arr if x>0.5))
    sevp,sevn=fp("severity"); locp,_=fp("location"); chp,_=fp("character")
    print(f"{d:<16}{nf:>5}{fill['severity']:>9}{sevp:>6.0f}%{sevn:>7}{locp:>6.0f}%{chp:>6.0f}%")
