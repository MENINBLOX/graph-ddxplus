"""@10-targeting: presentation-focused exhaustive IE. Forces the ACUTE CLINICAL
PRESENTATION (what brings the patient in), not molecular/lab/chronic content.
Benchmark-blind (full disease name only). Goal = GT always in top-10 (recall)."""
import sys, json, argparse, re
from pathlib import Path
PROMPT='''You are an experienced clinician. A patient comes to the clinic and is later diagnosed with: "{disease}".

List EXHAUSTIVELY the symptoms and signs THIS patient would actually REPORT or SHOW AT PRESENTATION — the complaints that brought them in and the findings on exam. Think like a clinical vignette / chief complaints.
- Focus on the PATIENT-PRESENTING clinical picture (pain, fever, swelling, location, rash, cough, etc.).
- For "{disease}", include the typical acute presentation specifically.
- EXCLUDE molecular biology, lab assay names, viral load markers, pathophysiology, epidemiology, treatment.
- Use concise lay/clinical terms patients and doctors use.
Output JSON: {{"findings": ["finding1", ...]}}'''
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--shard_file",required=True); ap.add_argument("--out_dir",required=True)
    a=ap.parse_args(); Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    ds=[l.strip().split("\t") for l in open(a.shard_file) if "\t" in l]
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    sp=SamplingParams(temperature=0.0,max_tokens=2048)
    outs=llm.chat([[{"role":"user","content":PROMPT.format(disease=dn)}] for c,dn in ds],sp,use_tqdm=True)
    for (c,dn),o in zip(ds,outs):
        m=re.search(r'\{.*\}',o.outputs[0].text,re.DOTALL); finds=[]
        if m:
            try: finds=json.loads(m.group(0)).get("findings",[])
            except: pass
        agg={f.strip().lower():{"n_mentions":1,"frequency_in_abstracts":1.0,"location_dist":{},"severity_dist":{},"onset_dist":{},"character_dist":{}} for f in finds if isinstance(f,str) and len(f.strip())>2 and f.strip().lower()!=dn.lower()}
        json.dump({"disease":dn,"cui":c,"aggregated":agg},open(f"{a.out_dir}/{c}.json","w"))
        print(f"  {dn}: {len(agg)}",flush=True)
main()
