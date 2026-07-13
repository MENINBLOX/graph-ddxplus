#!/usr/bin/env python3
"""Compare categorized IE on URTI vs Influenza vs Pneumonia history sections.

Critical confusion cluster — we want IE to clearly differentiate these.
"""
from __future__ import annotations
import os, sys, json, re
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from medkg_ie_categorized import IE_PROMPT, parse_categorized

SEC_PATH = MEDKG_ROOT / "processed" / "sections.jsonl"
TARGET = ["URTI", "Influenza", "Pneumonia"]
TARGET_SN = {"epidemiology", "history and physical", "signs and symptoms"}


def main():
    by_disease = {d: [] for d in TARGET}
    with SEC_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: s = json.loads(line)
            except: continue
            d = s.get("disease", "")
            sn = (s.get("section_name") or "").lower().strip()
            if d in by_disease and sn in TARGET_SN:
                t = (s.get("text") or "").strip()
                if len(t) < 200: continue
                s["text"] = t[:3000]
                by_disease[d].append(s)
    sections = []
    for d in TARGET:
        sections.extend(by_disease[d][:3])
    print(f"Loaded {len(sections)} sections")

    print("Loading vLLM...")
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.95,
              enforce_eager=True, limit_mm_per_prompt={"image":0,"audio":0})
    sampling = SamplingParams(temperature=0, max_tokens=500)

    convs = [[{"role":"user","content":IE_PROMPT.format(
        disease=s["disease"], section_name=s.get("section_name",""),
        text=s["text"]
    )}] for s in sections]
    outs = llm.chat(convs, sampling)

    print("\n" + "="*80)
    for s, o in zip(sections, outs):
        try: text = o.outputs[0].text
        except: text = ""
        print(f"\n--- {s['source']}/{s['disease']} [{s['section_name']}] ---")
        print(text)


if __name__ == "__main__":
    main()
