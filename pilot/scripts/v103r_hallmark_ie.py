"""Hallmark / pathognomonic feature IE. Extracts the MOST SPECIFIC findings that
strongly point to THIS disease over common mimics (malar rash→SLE, groin bulge
→hernia, target lesion→Lyme...) PLUS their body location. High-weighted so the
rare specific match dominates generic competitors. Benchmark-blind (disease name
only); medical knowledge of disease specificity, not benchmark mappings."""
import sys, json, argparse, re
from pathlib import Path
PROMPT='''You are an expert diagnostician. For a patient diagnosed with "{disease}", list the HALLMARK findings — the most SPECIFIC and DISTINGUISHING symptoms, signs, and their body locations that point to "{disease}" rather than to other diseases that present similarly.

- Give the pathognomonic / classic / highly specific features a patient would report or show (NOT generic features like "fatigue" or "fever" shared by many diseases).
- Include the characteristic BODY LOCATION as separate plain terms (e.g., "cheeks", "groin", "calf", "lower back").
- Use concise lay/clinical terms patients and clinicians use.
Output JSON: {{"hallmarks": ["...", "..."]}}'''
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--shard_file",required=True); ap.add_argument("--out_dir",required=True)
    a=ap.parse_args(); Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    ds=[l.strip().split("\t") for l in open(a.shard_file) if "\t" in l]
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    sp=SamplingParams(temperature=0.0,max_tokens=1536)
    outs=llm.chat([[{"role":"user","content":PROMPT.format(disease=dn)}] for c,dn in ds],sp,use_tqdm=True)
    for (c,dn),o in zip(ds,outs):
        m=re.search(r'\{.*\}',o.outputs[0].text,re.DOTALL); hm=[]
        if m:
            try: hm=json.loads(m.group(0)).get("hallmarks",[])
            except: pass
        agg={h.strip().lower():{"n_mentions":1} for h in hm if isinstance(h,str) and len(h.strip())>2 and h.strip().lower()!=dn.lower()}
        json.dump({"disease":dn,"cui":c,"aggregated":agg},open(f"{a.out_dir}/{c}.json","w"))
        print(f"  {dn}: {len(agg)}",flush=True)
main()
