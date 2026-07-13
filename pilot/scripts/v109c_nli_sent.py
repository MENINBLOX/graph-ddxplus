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
for ver in ["v105_grounded_ie","v106_grounded_ie","v107_grounded_ie"]:
    files=glob.glob(f"pilot/data/cache/{ver}/*.json")
    if not files: print(ver,"none"); continue
    # build flat pairs with claim ids
    P=[];H=[];cid=[];kind=[]; n=0
    for fp in files:
        o=json.load(open(fp)); ss=SRC.get(o["cui"])
        if not ss: continue
        for f in o["findings"]:
            for s in ss: P.append(s);H.append(f"Patients with this condition have {f['name']}.");cid.append(n)
            kind.append("f"); n+=1
            for k,v in f.items():
                if k=="name" or not v: continue
                for s in ss: P.append(s);H.append(hyp(f["name"],k,v));cid.append(n)
                kind.append("a"); n+=1
    sc=ent(P,H)
    # max per claim
    mx=[0.0]*n
    for c,x in zip(cid,sc):
        if x>mx[c]: mx[c]=x
    fnd=[mx[i] for i in range(n) if kind[i]=="f"]; att=[mx[i] for i in range(n) if kind[i]=="a"]
    def pct(a): return 100*sum(1 for x in a if x>0.5)/max(len(a),1)
    print(f"{ver}: claims={n} | finding faithful%={pct(fnd):.0f}(mean{statistics.mean(fnd):.2f}) | attr faithful%={pct(att):.0f}(mean{statistics.mean(att):.2f}) | overall faithful%={pct(mx):.0f}%",flush=True)
