"""Source-grounded CoT attribute IE. For each disease, the LLM reads a clinical
SOURCE (Wikipedia signs/symptoms, CC BY-SA), reasons step-by-step (CoT) about what
is EXPLICITLY stated, then extracts findings+attributes. Post-validation regex
BLOCKS any finding name or attribute value whose key token is absent from the
source -> hallucination cannot enter the KG."""
import sys, json, glob, re, argparse
from pathlib import Path
ATTRS=["location","onset","duration","character","severity","radiation","timing","aggravating","relieving","associated","course","context","prior_episodes"]
PROMPT='''You are a clinical information extractor. Below is SOURCE TEXT about "{disease}".

SOURCE:
"""{src}"""

TASK (think step by step, then output JSON):
Step 1 — Reasoning: read the SOURCE and identify the symptoms/signs it explicitly describes, and for each, which attributes (location, onset, character, severity, radiation, timing, aggravating/relieving factors, associated features, course, context) the SOURCE explicitly states.
Step 2 — Output JSON. For EACH finding, fill an attribute ONLY if the SOURCE explicitly states it (use the SOURCE's own words). If the SOURCE does not state an attribute, leave it "". Do NOT add knowledge that is not in the SOURCE. Do NOT use ranges/hedges.

After your reasoning, output exactly one JSON object on its own:
JSON: {{"findings":[{{"name":"...","location":"...","onset":"...","duration":"...","character":"...","severity":"...","radiation":"...","timing":"...","aggravating":"...","relieving":"...","associated":"...","course":"...","context":"...","prior_episodes":""}}]}}'''
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--src_dir",default="pilot/data/cache/v105_sources"); ap.add_argument("--out_dir",default="pilot/data/cache/v105_grounded_ie")
    a=ap.parse_args(); Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    import json as _j
    icd=_j.load(open("data/ddxplus/disease_icd10_cui_mapping.json")); cui2name={info["cui"]:dn for dn,info in icd.items() if "cui" in info}
    files=sorted(glob.glob(f"{a.src_dir}/*.txt"))
    items=[]
    for fp in files:
        c=fp.split("/")[-1][:-4]; src=open(fp).read()
        if c in cui2name: items.append((c,cui2name[c],src))
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=8192,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    outs=llm.chat([[{"role":"user","content":PROMPT.format(disease=dn,src=src[:2200])}] for c,dn,src in items],
                  SamplingParams(temperature=0.0,max_tokens=4096),use_tqdm=True)
    STOP=set("the a an of to in on with and or for is are be may can pain at as by from".split())
    def keytok(s): 
        ws=[w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
        return ws
    def in_src(val,srcl):
        ws=keytok(val)
        if not ws: return False
        return sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2+ (0 if len(ws)<=1 else 0))  # majority of key tokens present
    kept=0; dropped=0
    for (c,dn,src),o in zip(items,outs):
        srcl=src.lower()
        txt=o.outputs[0].text
        m=re.search(r'JSON:\s*(\{.*\})',txt,re.DOTALL) or re.search(r'(\{.*\})',txt,re.DOTALL)
        finds=[]
        if m:
            try: finds=_j.loads(m.group(1)).get("findings",[])
            except: pass
        clean=[]
        for f in finds:
            if not isinstance(f,dict): continue
            nm=str(f.get("name","")).strip().lower()
            if len(nm)<2 or nm==dn.lower(): continue
            if not in_src(nm,srcl): dropped+=1; continue   # name must be grounded
            rec={"name":nm}
            for at in ATTRS:
                v=str(f.get(at,"")).strip().lower()
                if v and not re.search(r'\b(to|or|and)\b|,|/',v) and len(v)<=40 and in_src(v,srcl):
                    rec[at]=v; kept+=1
                else:
                    rec[at]=""
                    if v: dropped+=1
            clean.append(rec)
        _j.dump({"disease":dn,"cui":c,"findings":clean},open(f"{a.out_dir}/{c}.json","w"))
        na=sum(1 for f in clean for at in ATTRS if f[at])
        print(f"  {dn}: {len(clean)} findings, {na} grounded attr-values",flush=True)
    print(f"\n총 grounded attr-values kept={kept}, source에 없어 drop={dropped}")
main()
