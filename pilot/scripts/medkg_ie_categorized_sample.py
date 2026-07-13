#!/usr/bin/env python3
"""Sample (small N) Categorized IE for validation before full run.

Picks 20-30 sections covering DDXPlus-relevant diseases (URTI, Influenza,
Pneumonia, Bronchitis, Pharyngitis, Sarcoidosis, Pericarditis, Myocarditis)
and runs the categorized IE prompt. Prints input → output pairs for human
inspection.
"""
from __future__ import annotations
import os, sys, json, re, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from medkg_ie_categorized import IE_PROMPT, parse_categorized, TARGET_SECTIONS

SEC_PATH = MEDKG_ROOT / "processed" / "sections.jsonl"

# Diseases to sample
TARGET_DISEASES = {"URTI","Influenza","Pneumonia","Bronchitis","Viral pharyngitis",
                   "Sarcoidosis","Pericarditis","Myocarditis","Spontaneous pneumothorax",
                   "Acute laryngitis","Acute rhinosinusitis"}


def main():
    sections = []
    with SEC_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: s = json.loads(line)
            except: continue
            sn = (s.get("section_name") or "").lower().strip()
            if sn not in TARGET_SECTIONS: continue
            disease = s.get("disease", "")
            if disease not in TARGET_DISEASES: continue
            t = (s.get("text") or "").strip()
            if len(t) < 200: continue
            s["text"] = t[:3000]
            sections.append(s)
            if len(sections) >= 30: break
    print(f"Sample sections: {len(sections)}")

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
        print(f"INPUT (first 300 chars): {s['text'][:300]}...")
        print(f"OUTPUT:")
        print(text)
        feats = parse_categorized(text)
        print(f"PARSED: {len(feats)} features")
        from collections import Counter
        c = Counter(f[0] for f in feats)
        print(f"  Categories: {dict(c)}")


if __name__ == "__main__":
    main()
