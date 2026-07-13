#!/usr/bin/env python3
"""Stage 2: run vLLM IE on Wikipedia records using pre-built scispaCy cache.

Input: /windows/data/medkg/processed/wikipedia_cui_cache.jsonl
  (produced by medkg_link_wikipedia.py — scispaCy runs in a separate process)

Output: /windows/data/medkg/processed/edges_wikipedia_ie.jsonl

Universal: Wikipedia canonical title as disease anchor (post-redirect),
mapped to UMLS CUI offline.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path

# Default to GPU 1 so we don't collide with the Phase D PubMed IE on GPU 0
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from medkg_ie_pubmed import IE_PROMPT, parse_phenotypes, log

CACHE_PATH = MEDKG_ROOT / "processed" / "wikipedia_cui_cache.jsonl"
OUT_PATH = MEDKG_ROOT / "processed" / "edges_wikipedia_ie.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    if not CACHE_PATH.exists():
        log(f"Cache not found: {CACHE_PATH}. Run medkg_link_wikipedia.py first.")
        sys.exit(1)

    records = []
    with CACHE_PATH.open() as f:
        for line in f:
            try: records.append(json.loads(line))
            except: pass
    log(f"Loaded {len(records)} cached records")
    if not records:
        log("Nothing to do."); return

    t0 = time.time()
    log("Loading vLLM (gemma-4-E4B-it)...")
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.95,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=350)

    BATCH = 256
    n_edges = 0; n_done = 0
    with OUT_PATH.open("w") as out:
        for chunk_start in range(0, len(records), BATCH):
            chunk = records[chunk_start:chunk_start + BATCH]
            convs = [[{"role": "user", "content": IE_PROMPT.format(
                disease=r["disease"], title=r["disease"], text=r["text"]
            )}] for r in chunk]
            outputs = llm.chat(convs, sampling)
            for r, o in zip(chunk, outputs):
                try: text = o.outputs[0].text
                except: text = ""
                for p in parse_phenotypes(text):
                    edge = {"disease": r["disease"], "umls_cui": r["umls_cui"],
                            "phenotype": p, "source": "wikipedia",
                            "source_id": r["source_file"], "section_name": "intro",
                            "section_id": r["page_id"],
                            "extracted_by": "gemma-4-E4B"}
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
                n_done += 1
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            log(f"  {n_done}/{len(records)}  edges={n_edges:,}  rate={rate:.1f}/s")
    log(f"Done. {n_edges:,} edges written to {OUT_PATH}")


if __name__ == "__main__":
    main()
