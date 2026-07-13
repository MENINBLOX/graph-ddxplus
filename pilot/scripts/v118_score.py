"""Score severity-binding on DEV vs held-out TEST split separately.
Tuning decisions use DEV only; the winner's TEST number is the reportable result."""
import json, glob, re
from collections import defaultdict

ANN="pilot/data/cache/maccrobat/brat"
SPLIT=json.load(open("pilot/data/cache/maccrobat/split.json"))
STOP=set("the a an of to in on with and or for is are be may can at as by from his her their patient left right".split())
def toks(s): return {w for w in re.findall(r'[a-z]+',s.lower()) if w not in STOP and len(w)>2}
def sym_match(a,b):
    ta,tb=toks(a),toks(b); return bool(ta and tb and (a.lower() in b.lower() or b.lower() in a.lower() or ta&tb))

def parse_gold():
    g=defaultdict(list)
    for ann in glob.glob(f"{ANN}/*.ann"):
        pmid=ann.split("/")[-1][:-4]; L=open(ann).read().splitlines(); ents={}; evs={}
        for line in L:
            p=line.split("\t")
            if p[0].startswith("T") and len(p)>=3: ents[p[0]]=(p[1].split()[0].lower(),p[2])
            elif p[0].startswith("E") and len(p)>=2:
                t,tid=p[1].split()[0].split(":"); evs[p[0]]=(t.lower(),tid)
        def tt(a):
            if a in ents: return ents[a]
            if a in evs: return (evs[a][0],ents.get(evs[a][1],("",""))[1])
            return ("","")
        for line in L:
            p=line.split("\t")
            if p[0].startswith("R") and len(p)>=2 and p[1].split()[0]=="MODIFY":
                m=re.findall(r'Arg\d:(\w+)',p[1])
                if len(m)==2 and ents.get(m[0],("",""))[0]=="severity":
                    typ,sym=tt(m[1])
                    if typ=="sign_symptom" and sym: g[pmid].append((sym.lower(),ents[m[0]][1].lower()))
    return g

GOLD=parse_gold()
docs=json.load(open("pilot/data/cache/maccrobat/MACCROBAT2020-V2.json"))["data"]
txt2pmid={open(t).read().strip()[:120]:t.split("/")[-1][:-4] for t in glob.glob(f"{ANN}/*.txt")}
idx2pmid=[txt2pmid.get(d["full_text"].strip()[:120]) for d in docs]

def score(pred_file, docset):
    preds=json.load(open(pred_file)); TPb=TPv=NP=NG=0
    for i,doc in enumerate(preds):
        pmid=idx2pmid[i]
        if not pmid or pmid not in docset: continue
        gp=GOLD.get(pmid,[]); pp=[(x["name"],x["severity"]) for x in doc if x["severity"]]
        NP+=len(pp); NG+=len(gp); ub=[False]*len(gp); uv=[False]*len(gp)
        for pn,pv in pp:
            for j,(gn,gv) in enumerate(gp):
                if not ub[j] and sym_match(pn,gn): ub[j]=True; TPb+=1; break
            for j,(gn,gv) in enumerate(gp):
                if not uv[j] and sym_match(pn,gn) and (pv in gv or gv in pv or toks(pv)&toks(gv)): uv[j]=True; TPv+=1; break
    def prf(tp): P=tp/NP if NP else 0; R=tp/NG if NG else 0; return P,R,(2*P*R/(P+R) if P+R else 0)
    return prf(TPb), prf(TPv), NP

VARS=["v117_base","v117_R2","v118_M1","v118_M2","v118_M3"]
for name,ds in [("DEV",set(SPLIT["dev"])),("TEST(held-out)",set(SPLIT["test"]))]:
    print(f"\n=== {name} ===  {'variant':<12}{'pred':>5}{'  binding P/R/F1':>20}{'  value P/R/F1':>20}")
    for v in VARS:
        f=f"pilot/data/cache/maccrobat/{v}_pred.json"
        if not glob.glob(f): continue
        (Pb,Rb,Fb),(Pv,Rv,Fv),NP=score(f,ds)
        print(f"{'':17}{v:<12}{NP:>5}   {Pb:.2f}/{Rb:.2f}/{Fb:.2f}     {Pv:.2f}/{Rv:.2f}/{Fv:.2f}")
