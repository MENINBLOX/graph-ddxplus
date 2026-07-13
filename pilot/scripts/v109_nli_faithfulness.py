"""Standard NLI faithfulness eval of IE outputs vs source. For each extracted claim
(finding + each attribute), compute max P(entailment) over source sentences using a
standard MNLI model (sentence-level granularity = standard faithfulness practice).
Faithful if max-entailment > 0.5. Compares v105/v106/v107 — replaces ad-hoc regex
with an accepted metric (NLI entailment = faithfulness, per hallucination surveys)."""
import json, glob, re, torch, statistics
from transformers import AutoModelForSequenceClassification, AutoTokenizer
MID="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
print("loading NLI...",flush=True)
tok=AutoTokenizer.from_pretrained(MID)
mdl=AutoModelForSequenceClassification.from_pretrained(MID).to("cuda").eval()
ENT=[i for i,l in mdl.config.id2label.items() if l.lower().startswith("entail")][0]
def split_sents(t):
    t=re.sub(r'\s+',' ',t)
    return [s.strip() for s in re.split(r'(?<=[.;])\s+',t) if len(s.strip())>15][:40]
def hyp(name,attr,val):
    T={"location":f"The {name} is located in the {val}.","onset":f"The {name} has a {val} onset.",
       "severity":f"The {name} is {val}.","character":f"The {name} is {val}.","radiation":f"The {name} radiates {val}.",
       "timing":f"The {name} occurs {val}.","aggravating":f"The {name} is aggravated by {val}.","relieving":f"The {name} is relieved by {val}.",
       "duration":f"The {name} lasts {val}.","associated":f"The {name} is associated with {val}.","context":f"The {name} occurs in the context of {val}."}
    return T.get(attr,f"The {name} is {val}.")
@torch.no_grad()
def max_entail(prem_sents,hypo):
    # batch premise sentences vs single hypothesis
    enc=tok(prem_sents,[hypo]*len(prem_sents),return_tensors="pt",padding=True,truncation=True,max_length=256).to("cuda")
    p=torch.softmax(mdl(**enc).logits,dim=1)[:,ENT]
    return float(p.max())
SRC={fp.split("/")[-1][:-4]:open(fp).read() for fp in glob.glob("pilot/data/cache/v105_sources/*.txt")}
for ver in ["v105_grounded_ie","v106_grounded_ie","v107_grounded_ie"]:
    files=glob.glob(f"pilot/data/cache/{ver}/*.json")
    if not files: print(ver,"없음"); continue
    fnd=[]; att=[]
    for fp in files:
        o=json.load(open(fp)); s=SRC.get(o["cui"],"")
        if not s: continue
        sents=split_sents(s)
        for f in o["findings"]:
            fnd.append(max_entail(sents,f"Patients with this condition have {f['name']}."))
            for k,v in f.items():
                if k=="name" or not v: continue
                att.append(max_entail(sents,hyp(f["name"],k,v)))
    alls=fnd+att
    def pct(a,t=0.5): return 100*sum(1 for x in a if x>t)/max(len(a),1)
    print(f"\n{ver}: claims={len(alls)}")
    print(f"  finding entail: mean={statistics.mean(fnd):.3f} faithful%(>0.5)={pct(fnd):.0f}% n={len(fnd)}")
    print(f"  attr    entail: mean={statistics.mean(att):.3f} faithful%(>0.5)={pct(att):.0f}% n={len(att)}")
    print(f"  전체    entail: mean={statistics.mean(alls):.3f} faithful%={pct(alls):.0f}%")
