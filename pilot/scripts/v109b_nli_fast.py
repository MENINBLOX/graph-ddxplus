import json, glob, re, torch, statistics
from transformers import AutoModelForSequenceClassification, AutoTokenizer
MID="FacebookAI/roberta-large-mnli"
tok=AutoTokenizer.from_pretrained(MID)
mdl=AutoModelForSequenceClassification.from_pretrained(MID,dtype=torch.float16).to("cuda").eval()
ENT=[i for i,l in mdl.config.id2label.items() if l.lower().startswith("entail")][0]
def hyp(name,a,v):
    T={"location":f"The {name} is located in the {v}.","onset":f"The {name} has a {v} onset.","severity":f"The {name} is {v}.",
       "character":f"The {name} is {v}.","radiation":f"The {name} radiates {v}.","timing":f"The {name} occurs {v}.",
       "aggravating":f"The {name} is aggravated by {v}.","relieving":f"The {name} is relieved by {v}.","duration":f"The {name} lasts {v}.",
       "associated":f"The {name} is associated with {v}.","context":f"The {name} occurs in the context of {v}."}
    return T.get(a,f"The {name} is {v}.")
SRC={fp.split("/")[-1][:-4]:open(fp).read()[:2000] for fp in glob.glob("pilot/data/cache/v105_sources/*.txt")}
@torch.no_grad()
def ent_batch(prems,hyps,bs=32):
    out=[]
    for i in range(0,len(prems),bs):
        enc=tok(prems[i:i+bs],hyps[i:i+bs],return_tensors="pt",padding="max_length",truncation=True,max_length=512).to("cuda")
        p=torch.softmax(mdl(**enc).logits.float(),dim=1)[:,ENT]
        out+=[float(x) for x in p]
    return out
for ver in ["v105_grounded_ie","v106_grounded_ie","v107_grounded_ie"]:
    files=glob.glob(f"pilot/data/cache/{ver}/*.json")
    if not files: print(ver,"none"); continue
    prems=[];hyps=[];kinds=[]
    for fp in files:
        o=json.load(open(fp)); s=SRC.get(o["cui"])
        if not s: continue
        for f in o["findings"]:
            prems.append(s);hyps.append(f"Patients with this condition have {f['name']}.");kinds.append("f")
            for k,v in f.items():
                if k=="name" or not v: continue
                prems.append(s);hyps.append(hyp(f["name"],k,v));kinds.append("a")
    sc=ent_batch(prems,hyps)
    fnd=[x for x,k in zip(sc,kinds) if k=="f"];att=[x for x,k in zip(sc,kinds) if k=="a"]
    def pct(a): return 100*sum(1 for x in a if x>0.5)/max(len(a),1)
    print(f"{ver}: claims={len(sc)} | finding faithful%={pct(fnd):.0f}(mean{statistics.mean(fnd):.2f},n{len(fnd)}) | attr faithful%={pct(att):.0f}(mean{statistics.mean(att):.2f},n{len(att)}) | overall={pct(sc):.0f}%",flush=True)
