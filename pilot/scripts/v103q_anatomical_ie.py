"""Anatomical-location-separated IE. Forces SEPARATE body-location findings
(groin, scrotum, lower abdomen, flank...) so they map to anatomy CUIs that
match DDXPlus location evidence (patient reports location as its own evidence).
Past anatomical IE gave large jumps (task #176/177). Benchmark-blind (disease
name only). Self-knowledge, no corpus."""
import sys, json, argparse, re
from pathlib import Path
PROMPT='''You are an experienced clinician describing how a patient with "{disease}" presents.

Output TWO lists for THIS patient's typical presentation:
1. "findings": EXHAUSTIVE symptoms and signs the patient REPORTS or SHOWS (pain, fever, swelling, bulge, cough, rash, shortness of breath, etc.). Concise lay/clinical terms.
2. "locations": EXHAUSTIVE list of BODY LOCATIONS, listed SEPARATELY as standalone anatomical terms, where the patient feels pain / swelling / symptoms or where signs appear. Use the SAME plain anatomical words a patient or triage form uses (e.g., "groin", "scrotum", "lower abdomen", "right lower quadrant", "flank", "chest", "throat", "calf"). One location per entry, no description.

For "{disease}" specifically, be precise about WHERE the patient localizes the complaint.
EXCLUDE molecular biology, lab assays, pathophysiology, epidemiology, treatment.
Output JSON: {{"findings": ["..."], "locations": ["..."]}}'''
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--shard_file",required=True); ap.add_argument("--out_dir",required=True)
    a=ap.parse_args(); Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    ds=[l.strip().split("\t") for l in open(a.shard_file) if "\t" in l]
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    sp=SamplingParams(temperature=0.0,max_tokens=2048)
    outs=llm.chat([[{"role":"user","content":PROMPT.format(disease=dn)}] for c,dn in ds],sp,use_tqdm=True)
    for (c,dn),o in zip(ds,outs):
        m=re.search(r'\{.*\}',o.outputs[0].text,re.DOTALL); finds=[]; locs=[]
        if m:
            try:
                j=json.loads(m.group(0)); finds=j.get("findings",[]); locs=j.get("locations",[])
            except: pass
        agg={}
        for f in finds:
            if isinstance(f,str) and len(f.strip())>2 and f.strip().lower()!=dn.lower():
                agg[f.strip().lower()]={"n_mentions":1}
        for l in locs:
            if isinstance(l,str) and len(l.strip())>1:
                agg[l.strip().lower()]={"n_mentions":2}  # weight locations higher
        json.dump({"disease":dn,"cui":c,"aggregated":agg},open(f"{a.out_dir}/{c}.json","w"))
        print(f"  {dn}: {len(agg)} ({len(locs)} loc)",flush=True)
main()
