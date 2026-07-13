"""Exposure/risk-factor IE: the discriminator for generic-presentation diseases
(HIV acute = flu-like, but exposure=sexual/IV distinguishes it). Benchmark-blind
(disease name only). Concise terms (avoid verbose phrases that mis-link)."""
import sys, json, argparse, re
from pathlib import Path
PROMPT='''For the disease "{disease}", list the characteristic RISK FACTORS, EXPOSURES, and relevant PATIENT HISTORY that point toward this diagnosis — the things a clinician asks about in the history that raise suspicion for "{disease}".
Examples of the KIND of items (not for a specific disease): unprotected sexual contact, intravenous drug use, recent travel, smoking, animal/insect bite, contaminated food, family history, recent surgery, known allergy, occupational exposure.
Use SHORT standard terms (2-4 words each, no parenthetical explanations). Exclude symptoms and treatment.
Output JSON: {{"findings": ["risk1", "risk2", ...]}}'''
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--shard_file",required=True); ap.add_argument("--out_dir",required=True)
    a=ap.parse_args(); Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    ds=[l.strip().split("\t") for l in open(a.shard_file) if "\t" in l]
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    outs=llm.chat([[{"role":"user","content":PROMPT.format(disease=dn)}] for c,dn in ds],SamplingParams(temperature=0.0,max_tokens=1024),use_tqdm=True)
    for (c,dn),o in zip(ds,outs):
        m=re.search(r'\{.*\}',o.outputs[0].text,re.DOTALL); finds=[]
        if m:
            try: finds=json.loads(m.group(0)).get("findings",[])
            except: pass
        agg={re.split(r'[\(,]',f)[0].strip().lower():{"n_mentions":1,"frequency_in_abstracts":1.0,"location_dist":{},"severity_dist":{},"onset_dist":{},"character_dist":{}} for f in finds if isinstance(f,str) and len(f.strip())>2}
        json.dump({"disease":dn,"cui":c,"aggregated":agg},open(f"{a.out_dir}/{c}.json","w"))
        print(f"  {dn}: {len(agg)}",flush=True)
main()
