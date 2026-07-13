"""STRICT attribute IE (anti-hallucination). Fill an attribute ONLY when it is a
CHARACTERISTIC/DISCRIMINATIVE feature for that finding in that disease. No hedged
ranges ('mild to severe', 'constant or intermittent') — leave empty. Most findings
should have only 1-3 attributes. Post-validation blanks any range/hedge value."""
import sys, json, argparse, re
from pathlib import Path
ATTRS=["location","onset","duration","character","severity","radiation","timing","aggravating","relieving","associated","course","context","prior_episodes"]
PROMPT='''You are an expert clinician. For a patient with "{disease}", list the symptoms and signs they present with (be reasonably comprehensive).

For EACH finding, fill a clinical attribute ONLY when that value is CHARACTERISTIC and DISCRIMINATIVE for "{disease}" — a specific detail a clinician would cite to DISTINGUISH this disease from others. 

STRICT RULES:
- If an attribute is variable, nonspecific, or not characteristic for this finding → leave it "" (empty).
- NEVER write ranges or hedges. Do NOT write "mild to severe", "sudden or gradual", "constant or intermittent", "first episode or recurrent". If it varies, leave "".
- Most findings should have only 1-3 attributes filled. Empty is the default.
- location should be a precise body site only when characteristic (e.g. malar rash -> "cheeks"; groin bulge -> "groin").

Attributes: location, onset(sudden/gradual), duration, character(sharp/dull/burning/throbbing/pruritic/productive...), severity(mild/moderate/severe ONE value), radiation, timing, aggravating, relieving, associated, course, context, prior_episodes.

Output JSON: {{"findings":[{{"name":"...","location":"...","onset":"...","duration":"...","character":"...","severity":"...","radiation":"...","timing":"...","aggravating":"...","relieving":"...","associated":"...","course":"...","context":"...","prior_episodes":"..."}}]}}'''
HEDGE=re.compile(r'\b(to|or|and|/)\b|,')
def clean(v):
    v=str(v).strip()
    if not v: return ""
    if HEDGE.search(v.lower()): return ""        # drop hedged ranges
    if len(v)>40: return ""                        # drop verbose
    if v.lower() in ("variable","nonspecific","n/a","none","any","unknown",""): return ""
    return v.lower()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--shard_file",required=True); ap.add_argument("--out_dir",required=True)
    a=ap.parse_args(); Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    ds=[l.strip().split("\t") for l in open(a.shard_file) if "\t" in l]
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=8192,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    outs=llm.chat([[{"role":"user","content":PROMPT.format(disease=dn)}] for c,dn in ds],SamplingParams(temperature=0.0,max_tokens=5120),use_tqdm=True)
    for (c,dn),o in zip(ds,outs):
        m=re.search(r'\{.*\}',o.outputs[0].text,re.DOTALL); finds=[]
        if m:
            try: finds=json.loads(m.group(0)).get("findings",[])
            except: pass
        clean_f=[]
        for f in finds:
            if not isinstance(f,dict): continue
            nm=str(f.get("name","")).strip().lower()
            if len(nm)<2 or nm==dn.lower(): continue
            rec={"name":nm}
            for at in ATTRS: rec[at]=clean(f.get(at,""))
            clean_f.append(rec)
        json.dump({"disease":dn,"cui":c,"findings":clean_f},open(f"{a.out_dir}/{c}.json","w"))
        nf=sum(1 for f in clean_f for at in ATTRS if f[at])
        print(f"  {dn}: {len(clean_f)} findings, {nf} attr-values ({nf/max(len(clean_f),1):.1f}/finding)",flush=True)
main()
