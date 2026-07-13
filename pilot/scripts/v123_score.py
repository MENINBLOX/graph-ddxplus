"""Diagnose MACCROBAT gold relation-incompleteness from GOLD ITSELF (no external
judgment), and build a proximity-reconstructed complete severity->finding gold from
gold entity offsets. Re-measure variants against it for a fairer F1.

(1) Of all gold Severity ENTITIES, what % have a MODIFY->finding relation? The rest
    = relations the annotators left out -> proves relation-incompleteness.
(2) Proximity gold: pair each Severity entity with the nearest Sign_symptom/
    Disease_disorder entity by character offset (uses only gold spans)."""
import json, glob, re
from collections import defaultdict

ANN="pilot/data/cache/maccrobat/brat"
SPLIT=json.load(open("pilot/data/cache/maccrobat/split.json"))
FIND={"sign_symptom","disease_disorder"}

def parse_full(ann):
    L=open(ann).read().splitlines(); ents={}; evs={}
    for line in L:
        p=line.split("\t")
        if p[0].startswith("T") and len(p)>=3:
            head=p[1].split(); typ=head[0].lower()
            try: s=int(head[1]); e=int(head[-1])
            except: s=e=-1
            ents[p[0]]=(typ,p[2],s,e)
        elif p[0].startswith("E") and len(p)>=2:
            t,tid=p[1].split()[0].split(":"); evs[p[0]]=(t.lower(),tid)
    def resolve(a):
        if a in ents: return ents[a]
        if a in evs:
            tid=evs[a][1]; return ents.get(tid,(evs[a][0],"",-1,-1))
        return ("","",-1,-1)
    rels=[]
    for line in L:
        p=line.split("\t")
        if p[0].startswith("R") and len(p)>=2 and p[1].split()[0]=="MODIFY":
            m=re.findall(r'Arg\d:(\w+)',p[1])
            if len(m)==2: rels.append((m[0],m[1]))
    return ents,evs,resolve,rels

# (1) relation-incompleteness
def analyze(docset):
    sev_total=sev_with_rel=0
    for ann in glob.glob(f"{ANN}/*.ann"):
        pmid=ann.split("/")[-1][:-4]
        if pmid not in docset: continue
        ents,evs,resolve,rels=parse_full(ann)
        sev_ids=[t for t,v in ents.items() if v[0]=="severity"]
        linked=set()
        for a1,a2 in rels:
            if ents.get(a1,("",))[0]=="severity" and resolve(a2)[0] in FIND: linked.add(a1)
        sev_total+=len(sev_ids); sev_with_rel+=len(linked)
    return sev_total,sev_with_rel

# (2) proximity-reconstructed gold per pmid
def proximity_gold(ann):
    ents,evs,resolve,rels=parse_full(ann)
    sevs=[(v[1],v[2]) for v in ents.values() if v[0]=="severity" and v[2]>=0]
    finds=[(v[1],v[2],v[3]) for v in ents.values() if v[0] in FIND and v[2]>=0]
    pairs=[]
    for sv,sp in sevs:
        if not finds: continue
        nf=min(finds,key=lambda f:min(abs(sp-f[1]),abs(sp-f[2])))
        pairs.append((nf[0].lower(),sv.lower()))
    return pairs

for name,ds in [("DEV",set(SPLIT["dev"])),("TEST",set(SPLIT["test"]))]:
    tot,wr=analyze(ds)
    print(f"[{name}] gold Severity entities={tot}, with MODIFY->finding relation={wr} "
          f"({100*wr/max(tot,1):.0f}%) -> relation-incompleteness={100*(tot-wr)/max(tot,1):.0f}%")

# write proximity gold
docs=json.load(open("pilot/data/cache/maccrobat/MACCROBAT2020-V2.json"))["data"]
txt2pmid={open(t).read().strip()[:120]:t.split("/")[-1][:-4] for t in glob.glob(f"{ANN}/*.txt")}
idx2pmid=[txt2pmid.get(d["full_text"].strip()[:120]) for d in docs]
PG={}
for ann in glob.glob(f"{ANN}/*.ann"):
    PG[ann.split("/")[-1][:-4]]=proximity_gold(ann)
json.dump(PG,open("pilot/data/cache/maccrobat/proximity_gold.json","w"))
ntest=sum(len(PG[p]) for p in SPLIT["test"]); print(f"proximity gold TEST pairs={ntest}")

# (3) re-score variants vs proximity gold
STOP=set("the a an of to in on with and or for is are be may can at as by from his her their patient left right".split())
def toks(s): return {w for w in re.findall(r'[a-z]+',s.lower()) if w not in STOP and len(w)>2}
def sym_match(a,b):
    ta,tb=toks(a),toks(b); return bool(ta and tb and (a.lower() in b.lower() or b.lower() in a.lower() or ta&tb))
def score(predf,docset):
    preds=json.load(open(predf)); TPb=TPv=NP=NG=0
    for i,doc in enumerate(preds):
        pmid=idx2pmid[i]
        if not pmid or pmid not in docset: continue
        gp=PG.get(pmid,[]); pp=[(x["name"],x["severity"]) for x in doc if x["severity"]]
        NP+=len(pp); NG+=len(gp); ub=[False]*len(gp); uv=[False]*len(gp)
        for pn,pv in pp:
            for j,(gn,gv) in enumerate(gp):
                if not ub[j] and sym_match(pn,gn): ub[j]=True; TPb+=1; break
            for j,(gn,gv) in enumerate(gp):
                if not uv[j] and sym_match(pn,gn) and (pv in gv or gv in pv or toks(pv)&toks(gv)): uv[j]=True; TPv+=1; break
    def prf(tp): P=tp/NP if NP else 0; R=tp/NG if NG else 0; return P,R,(2*P*R/(P+R) if P+R else 0)
    return prf(TPb),prf(TPv),NP
print(f"\n=== vs PROXIMITY gold (TEST held-out) ===  {'variant':<12}{'pred':>5}{'  binding P/R/F1':>20}{'  value P/R/F1':>18}")
for v in ["v117_base","v118_M2","v119_M4","v123_M6","v123_M7"]:
    f=f"pilot/data/cache/maccrobat/{v}_pred.json"
    if not glob.glob(f): continue
    (Pb,Rb,Fb),(Pv,Rv,Fv),NP=score(f,set(SPLIT["test"]))
    print(f"{'':17}{v:<12}{NP:>5}   {Pb:.2f}/{Rb:.2f}/{Fb:.2f}    {Pv:.2f}/{Rv:.2f}/{Fv:.2f}")
