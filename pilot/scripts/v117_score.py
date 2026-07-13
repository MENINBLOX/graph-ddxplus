"""Score severity-binding vs MACCROBAT MODIFY(Severity->Sign_symptom) gold.
Recognized relation-level P/R/F1: did we attach a severity to the RIGHT symptom?
Primary metric = binding detection (symptom token-overlap match), which directly
measures the defect baseline over-attaches case-level severity. Secondary =
value-aware (also require severity word overlap)."""
import json, glob, re
from collections import defaultdict

ANN_DIR="pilot/data/cache/maccrobat/brat"
STOP=set("the a an of to in on with and or for is are be may can at as by from his her their patient left right".split())
def toks(s): return {w for w in re.findall(r'[a-z]+',s.lower()) if w not in STOP and len(w)>2}
def sym_match(a,b):
    ta,tb=toks(a),toks(b)
    return bool(ta and tb and (a.lower() in b.lower() or b.lower() in a.lower() or ta&tb))

def parse_gold():
    """returns per-pmid: list of (symptom_text, severity_text)."""
    docid2pairs=defaultdict(list)
    for ann in glob.glob(f"{ANN_DIR}/*.ann"):
        pmid=ann.split("/")[-1][:-4]
        L=open(ann).read().splitlines()
        ents={}; evs={}
        for line in L:
            p=line.split("\t")
            if p[0].startswith("T") and len(p)>=3: ents[p[0]]=(p[1].split()[0].lower(),p[2])
            elif p[0].startswith("E") and len(p)>=2:
                typ,tid=p[1].split()[0].split(":"); evs[p[0]]=(typ.lower(),tid)
        def text_type(arg):
            if arg in ents: return ents[arg]
            if arg in evs:
                t=evs[arg][1]; return (evs[arg][0], ents.get(t,("",""))[1])
            return ("","")
        for line in L:
            p=line.split("\t")
            if p[0].startswith("R") and len(p)>=2 and p[1].split()[0]=="MODIFY":
                m=re.findall(r'Arg\d:(\w+)',p[1])
                if len(m)!=2: continue
                if ents.get(m[0],("",""))[0]=="severity":
                    typ,symtext=text_type(m[1])
                    if typ=="sign_symptom" and symtext:
                        docid2pairs[pmid].append((symtext.lower(), ents[m[0]][1].lower()))
    return docid2pairs

def main():
    gold=parse_gold()
    docs=json.load(open("pilot/data/cache/maccrobat/MACCROBAT2020-V2.json"))["data"]
    # map doc index -> pmid via full_text match against .txt
    txt2pmid={}
    for t in glob.glob(f"{ANN_DIR}/*.txt"):
        txt2pmid[open(t).read().strip()[:120]]=t.split("/")[-1][:-4]
    idx2pmid=[txt2pmid.get(d["full_text"].strip()[:120]) for d in docs]
    matched=sum(1 for x in idx2pmid if x)
    print(f"docs aligned to brat pmid: {matched}/{len(docs)}")
    ng=sum(len(v) for v in gold.values())
    print(f"gold severity-binding pairs total: {ng}\n")

    print(f"{'variant':<12}{'pred_pairs':>11}{'  binding P/R/F1':>22}{'  value-aware P/R/F1':>24}")
    for v in ["v117_base","v117_R2"]:
        preds=json.load(open(f"pilot/data/cache/maccrobat/{v}_pred.json"))
        TPb=TPv=NP=NG=0
        for d_i,doc in enumerate(preds):
            pmid=idx2pmid[d_i]
            if not pmid: continue
            gpairs=gold.get(pmid,[])
            ppairs=[(x["name"],x["severity"]) for x in doc if x["severity"]]
            NP+=len(ppairs); NG+=len(gpairs)
            usedb=[False]*len(gpairs); usedv=[False]*len(gpairs)
            for pn,pv in ppairs:
                for j,(gn,gv) in enumerate(gpairs):
                    if not usedb[j] and sym_match(pn,gn): usedb[j]=True; TPb+=1; break
                for j,(gn,gv) in enumerate(gpairs):
                    if not usedv[j] and sym_match(pn,gn) and (pv in gv or gv in pv or toks(pv)&toks(gv)):
                        usedv[j]=True; TPv+=1; break
        def prf(tp):
            P=tp/NP if NP else 0; R=tp/NG if NG else 0; return P,R,(2*P*R/(P+R) if P+R else 0)
        Pb,Rb,Fb=prf(TPb); Pv,Rv,Fv=prf(TPv)
        print(f"{v:<12}{NP:>11}   {Pb:.2f}/{Rb:.2f}/{Fb:.2f}        {Pv:.2f}/{Rv:.2f}/{Fv:.2f}")

if __name__=="__main__": main()
