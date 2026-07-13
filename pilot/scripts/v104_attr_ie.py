"""Attribute-rich IE: extract phenotypes each WITH the maximal 13-attribute set
(union of OLDCARTS/OPQRST/SOCRATES/NLICE/LOCATES/CPX). Benchmark-blind (disease
name only). Run ONCE; the all-subset ablation (13→4) reuses output in-memory."""
import sys, json, argparse, re
from pathlib import Path
ATTRS=["location","onset","duration","character","severity","radiation","timing",
       "aggravating","relieving","associated","course","context","prior_episodes"]
PROMPT='''You are an expert clinician characterizing how a patient with "{disease}" presents.

List EXHAUSTIVELY every symptom and sign this patient could report or show at presentation — be comprehensive, do not limit the count, include common and characteristic findings. For EACH finding, give its clinical attributes using the full history-taking framework (OLDCARTS/OPQRST/SOCRATES). Leave an attribute as "" if not applicable to that finding.

Attributes per finding:
- name: symptom/sign in concise lay/clinical terms ("pain","rash","swelling","cough")
- location: body location(s) as plain anatomical words ("groin","cheeks","lower abdomen","chest")
- onset: mode of onset ("sudden"/"gradual"/"acute"/"insidious")
- duration: how long it lasts ("minutes","hours","days","weeks","chronic")
- character: quality/nature ("sharp","dull","burning","throbbing","cramping","pruritic","productive")
- severity: intensity ("mild"/"moderate"/"severe")
- radiation: where it spreads ("to the left arm","to the scrotum","" if none)
- timing: chronology/frequency ("constant","intermittent","nocturnal","post-prandial","morning")
- aggravating: what worsens it ("exertion","lying down","deep breathing")
- relieving: what relieves it ("rest","sitting up","antacids")
- associated: features accompanying this finding (short)
- course: progression ("worsening","improving","relapsing-remitting","stable")
- context: trigger/exposure/setting that provoked it ("after shellfish","during exercise","seasonal","recent travel")
- prior_episodes: recurrence ("first episode","recurrent","chronic history")

For "{disease}", be precise about the characteristic attribute values that distinguish it.
Output JSON: {{"findings": [{{"name":"...","location":"...","onset":"...","duration":"...","character":"...","severity":"...","radiation":"...","timing":"...","aggravating":"...","relieving":"...","associated":"...","course":"...","context":"...","prior_episodes":"..."}}]}}'''
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--shard_file",required=True); ap.add_argument("--out_dir",required=True)
    a=ap.parse_args(); Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    ds=[l.strip().split("\t") for l in open(a.shard_file) if "\t" in l]
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=8192,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    sp=SamplingParams(temperature=0.0,max_tokens=6144)
    outs=llm.chat([[{"role":"user","content":PROMPT.format(disease=dn)}] for c,dn in ds],sp,use_tqdm=True)
    for (c,dn),o in zip(ds,outs):
        m=re.search(r'\{.*\}',o.outputs[0].text,re.DOTALL); finds=[]
        if m:
            try: finds=json.loads(m.group(0)).get("findings",[])
            except: pass
        clean=[]
        for f in finds:
            if not isinstance(f,dict): continue
            nm=str(f.get("name","")).strip().lower()
            if len(nm)<2 or nm==dn.lower(): continue
            rec={"name":nm}
            for at in ATTRS:
                v=f.get(at,""); rec[at]=str(v).strip().lower() if v else ""
            clean.append(rec)
        json.dump({"disease":dn,"cui":c,"findings":clean},open(f"{a.out_dir}/{c}.json","w"))
        print(f"  {dn}: {len(clean)} findings",flush=True)
main()
