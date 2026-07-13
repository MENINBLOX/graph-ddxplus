"""Coverage-max test: exhaustive LLM self-knowledge IE (benchmark-blind, disease
name only — 원칙 #5/#6). Quantifies coverage->@1 vs the verifiable strict IE.
Each disease: LLM lists ALL presenting findings. No source text (= unverifiable;
this measures the coverage ceiling, not a strict result)."""
import os, sys, json, argparse, re
from pathlib import Path
PROMPT = '''List the clinical findings (symptoms, signs, and how the patient presents) of the disease "{disease}".
Be EXHAUSTIVE — include common and characteristic presenting features a patient would report or a clinician would find. Use concise standard clinical terms. Exclude treatment and the disease name itself.
Output JSON: {{"findings": ["finding1", "finding2", ...]}}'''
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--shard_file",required=True); ap.add_argument("--out_dir",required=True)
    a=ap.parse_args(); Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    ds=[l.strip().split("\t") for l in open(a.shard_file) if "\t" in l]
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    sp=SamplingParams(temperature=0.0,max_tokens=2048)
    convs=[[{"role":"user","content":PROMPT.format(disease=dn)}] for c,dn in ds]
    outs=llm.chat(convs,sp,use_tqdm=True)
    for (c,dn),o in zip(ds,outs):
        m=re.search(r'\{.*\}',o.outputs[0].text,re.DOTALL); finds=[]
        if m:
            try: finds=json.loads(m.group(0)).get("findings",[])
            except: pass
        agg={f.strip().lower():{"n_mentions":1,"frequency_in_abstracts":1.0,"location_dist":{},"severity_dist":{},"onset_dist":{},"character_dist":{}} for f in finds if isinstance(f,str) and len(f.strip())>2 and f.strip().lower()!=dn.lower()}
        json.dump({"disease":dn,"cui":c,"aggregated":agg},open(f"{a.out_dir}/{c}.json","w"))
        print(f"  {dn}: {len(agg)}",flush=True)
main()
