"""Re-validate the unified 4-attribute prompt's severity (R2) and location (L1) on
MACCROBAT relation gold, AND compare single-pass vs multi-step CoT extraction.
- Gold: MACCROBAT MODIFY(Severity->Sign_symptom) and MODIFY(Biological_structure->Sign_symptom).
- Modes: 'single' = one prompt extracts findings + 4 attrs (current production shape);
         'cot'    = 2-stage: pass1 list findings, pass2 assign 4 attrs per finding w/ reasoning.
- Metric: binding P/R/F1 (right symptom got an attr) + value-aware P/R/F1, per the recognized
  relation-level scoring (v117_score). Compares to standalone R2 sev F1~0.42 / L1 loc F1~0.55.
gemma-4-E4B, temp=0, source-grounded, NO few-shot. Reuses validated rules from v143."""
import json, glob, re, argparse
from collections import defaultdict
from v143_unified_4attr import SEV_R2, LOC_L1, CHAR, ONSET

ANN_DIR="pilot/data/cache/maccrobat/brat"
ONSET_RULE=ONSET["O2"]

HEAD_SINGLE='''You are a clinical information extractor. Read the clinical CASE TEXT and extract, USING ONLY facts explicitly stated in it, every abnormal clinical finding the patient has, with its attributes.

CASE TEXT:
"""{src}"""

STEP 1 — Reasoning: list the patient's abnormal findings (symptoms, signs, lesions, abnormal conditions). EXCLUDE tests/procedures, medications, normal findings, headings.
STEP 2 — For each finding output its attributes (fill ONLY if the CASE TEXT states it; otherwise ""):
{onset}
{severity}
{location}
{character}

Output exactly:
JSON: {{"findings":[{{"name":"","onset":"","severity":"","location":"","character":""}}]}}'''

COT1='''You are a clinical information extractor. Read the clinical CASE TEXT and list every abnormal clinical finding the patient has (symptom, sign, lesion, abnormal condition). EXCLUDE tests/procedures, medications, normal findings, headings.

CASE TEXT:
"""{src}"""

Output ONLY JSON: {{"findings":["...","..."]}}'''

COT2='''You are a clinical information extractor. For EACH listed finding, decide its attributes USING ONLY facts explicitly stated in the CASE TEXT. Think step by step about which words in the text directly modify each finding before answering.

CASE TEXT:
"""{src}"""

FINDINGS: {finds}

For each finding fill attributes ONLY if the CASE TEXT states it for THAT finding; otherwise "":
{onset}
{severity}
{location}
{character}

Output exactly:
JSON: {{"findings":[{{"name":"","onset":"","severity":"","location":"","character":""}}]}}'''

STOP=set("the a an of to in on with and or for is are be may can at as by from his her their patient left right".split())
def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
def toks(s): return {w for w in re.findall(r'[a-z]+',s.lower()) if w not in STOP and len(w)>2}
def sym_match(a,b):
    ta,tb=toks(a),toks(b); return bool(ta and tb and (a.lower() in b.lower() or b.lower() in a.lower() or ta&tb))

def parse_gold(src_type):
    """MODIFY(src_type -> sign_symptom) gold pairs per pmid: (symptom_text, attr_text)."""
    g=defaultdict(list)
    for ann in glob.glob(f"{ANN_DIR}/*.ann"):
        pmid=ann.split("/")[-1][:-4]; L=open(ann).read().splitlines(); ents={}; evs={}
        for line in L:
            p=line.split("\t")
            if p[0].startswith("T") and len(p)>=3: ents[p[0]]=(p[1].split()[0].lower(),p[2])
            elif p[0].startswith("E") and len(p)>=2:
                t=p[1].split()[0].split(":"); evs[p[0]]=(t[0].lower(),t[1])
        def tt(arg):
            if arg in ents: return ents[arg]
            if arg in evs: return (evs[arg][0], ents.get(evs[arg][1],("",""))[1])
            return ("","")
        for line in L:
            p=line.split("\t")
            if p[0].startswith("R") and len(p)>=2 and p[1].split()[0]=="MODIFY":
                m=re.findall(r'Arg\d:(\w+)',p[1])
                if len(m)!=2: continue
                if ents.get(m[0],("",""))[0]==src_type:
                    typ,symtext=tt(m[1])
                    if typ=="sign_symptom" and symtext: g[pmid].append((symtext.lower(),ents[m[0]][1].lower()))
    return g

def score(preds_by_pmid, attr):
    src_type="severity" if attr=="severity" else "biological_structure"
    gold=parse_gold(src_type); TPb=TPv=NP=NG=0
    for pmid,doc in preds_by_pmid.items():
        gp=gold.get(pmid,[]); pp=[(x["name"],x[attr]) for x in doc if x.get(attr)]
        NP+=len(pp); NG+=len(gp); ub=[False]*len(gp); uv=[False]*len(gp)
        for pn,pv in pp:
            for j,(gn,gv) in enumerate(gp):
                if not ub[j] and sym_match(pn,gn): ub[j]=True; TPb+=1; break
            for j,(gn,gv) in enumerate(gp):
                if not uv[j] and sym_match(pn,gn) and (pv in gv or gv in pv or toks(pv)&toks(gv)): uv[j]=True; TPv+=1; break
    def prf(tp): P=tp/NP if NP else 0; R=tp/NG if NG else 0; return P,R,(2*P*R/(P+R) if P+R else 0)
    return prf(TPb),prf(TPv),NP,NG

def parse_json(txt):
    m=re.search(r'(\{.*\})',txt,re.DOTALL)
    if not m: return []
    try: return json.loads(m.group(1)).get("findings",[])
    except: return []

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=200); a=ap.parse_args()
    txts=sorted(glob.glob(f"{ANN_DIR}/*.txt"))[:a.n]
    docs=[(t.split("/")[-1][:-4], open(t).read()) for t in txts]
    single_tmpl=HEAD_SINGLE.replace("{onset}",ONSET_RULE).replace("{severity}",SEV_R2).replace("{location}",LOC_L1).replace("{character}",CHAR)
    cot2_tmpl=COT2.replace("{onset}",ONSET_RULE).replace("{severity}",SEV_R2).replace("{location}",LOC_L1).replace("{character}",CHAR)
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=8192,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    SP=SamplingParams(temperature=0.0,max_tokens=4096)

    def clean(finds,srcl):
        out=[]
        for f in finds:
            if not isinstance(f,dict): continue
            nm=str(f.get("name","")).strip().lower()
            if len(nm)<2 or not in_src(nm,srcl): continue
            g={"name":nm}
            for at in ("onset","severity","location","character"):
                v=str(f.get(at,"")).strip().lower()
                if at=="onset": v=v if v in ("sudden","gradual") else ""
                elif v and not in_src(v,srcl): v=""
                g[at]=v
            out.append(g)
        return out

    # SINGLE
    outs=llm.chat([[{"role":"user","content":single_tmpl.format(src=s)}] for _,s in docs],SP,use_tqdm=True)
    single={pmid:clean(parse_json(o.outputs[0].text),s.lower()) for (pmid,s),o in zip(docs,outs)}

    # COT stage1: findings
    o1=llm.chat([[{"role":"user","content":COT1.format(src=s)}] for _,s in docs],SP,use_tqdm=True)
    finds1=[]
    for o in o1:
        m=re.search(r'(\{.*\})',o.outputs[0].text,re.DOTALL); fl=[]
        if m:
            try: fl=[str(x) for x in json.loads(m.group(1)).get("findings",[]) if isinstance(x,str)]
            except: pass
        finds1.append(fl[:40])
    # COT stage2: assign attrs
    o2=llm.chat([[{"role":"user","content":cot2_tmpl.format(src=s,finds=json.dumps(fl))}] for (_,s),fl in zip(docs,finds1)],SP,use_tqdm=True)
    cot={pmid:clean(parse_json(o.outputs[0].text),s.lower()) for (pmid,s),o in zip(docs,o2)}

    for name,preds in (("single",single),("cot",cot)):
        json.dump(preds,open(f"pilot/data/cache/maccrobat/v144_{name}_pred.json","w"))
        for attr in ("severity","location"):
            (Pb,Rb,Fb),(Pv,Rv,Fv),NP,NG=score(preds,attr)
            print(f"[{name:6}] {attr:8} pred={NP:4} gold={NG:4}  binding P/R/F1={Pb:.2f}/{Rb:.2f}/{Fb:.2f}  value={Pv:.2f}/{Rv:.2f}/{Fv:.2f}",flush=True)

if __name__=="__main__": main()
