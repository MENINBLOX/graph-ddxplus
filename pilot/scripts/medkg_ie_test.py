#!/usr/bin/env python3
"""Test IE on first 50 sections from sections.jsonl. Confirm vLLM pipeline works."""
import os, json, re, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/home/max/Graph-DDXPlus/pilot/scripts")
from medkg_ie_multi_source import IE_PROMPT, parse_phenotypes

from pathlib import Path
SEC = Path("/home/max/Graph-DDXPlus/data/medkg/processed/sections.jsonl")


def main():
    sections = []
    with SEC.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            sections.append(json.loads(line))
    print(f"Total {len(sections)} sections; testing first 50")
    sample = sections[:50]

    from vllm import LLM, SamplingParams
    print("Loading gemma-4-E4B-it...")
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.95,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sp = SamplingParams(temperature=0, max_tokens=600)
    convs = [[{"role": "user", "content": IE_PROMPT.format(
        disease=s["disease"], source_type=s["source"],
        section_name=s.get("section_name","unknown"), text=s["text"][:3000])}] for s in sample]
    outs = llm.chat(convs, sp)
    n_findings = 0
    for s, o in zip(sample, outs):
        text = o.outputs[0].text
        f = parse_phenotypes(text)
        n_findings += len(f)
        if len(f) > 0 and len(f) < 20:  # show realistic results
            print(f"  [{s['source']:12s}] {s['disease'][:30]:30s} {s.get('section_name','')[:25]:25s} → {len(f)} phenotypes: {f[:5]}")
    print(f"\nTotal phenotypes: {n_findings} from {len(sample)} sections (avg {n_findings/len(sample):.1f}/section)")


if __name__ == "__main__":
    main()
