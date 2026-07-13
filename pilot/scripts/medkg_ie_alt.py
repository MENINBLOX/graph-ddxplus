#!/usr/bin/env python3
"""IE on alt-search PubMed abstracts (DDXPlus 49 deep coverage)."""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from medkg_ie_pubmed import IE_PROMPT, parse_phenotypes, log

ALT_DIR = MEDKG_ROOT / "pubmed_alt"
OUT = MEDKG_ROOT / "processed" / "edges_pubmed_alt_ie.jsonl"


def main():
    records = []
    for fp in ALT_DIR.glob("*.jsonl"):
        if fp.stat().st_size == 0: continue
        with fp.open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: e = json.loads(line)
                except: continue
                ab = (e.get("abstract") or "").strip()
                if len(ab) < 50: continue
                records.append({"disease": e.get("disease_name", ""),
                               "cui": e.get("cui", fp.stem),
                               "pmid": e.get("pmid", ""),
                               "title": e.get("title", ""),
                               "abstract": ab[:2500]})
    log(f"Loaded {len(records):,} alt abstracts")

    log("Loading vLLM...")
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.95,
              enforce_eager=True, limit_mm_per_prompt={"image":0,"audio":0})
    sampling = SamplingParams(temperature=0, max_tokens=400)

    n_edges = 0
    BATCH = 1024
    with OUT.open("w") as out:
        for cs in range(0, len(records), BATCH):
            chunk = records[cs:cs+BATCH]
            convs = [[{"role":"user","content":IE_PROMPT.format(
                disease=r["disease"], title=r["title"], text=r["abstract"]
            )}] for r in chunk]
            outs = llm.chat(convs, sampling)
            for r, o in zip(chunk, outs):
                try: text = o.outputs[0].text
                except: text = ""
                for p in parse_phenotypes(text):
                    edge = {"disease": r["disease"], "umls_cui": r["cui"],
                            "phenotype": p, "source": "pubmed_alt",
                            "source_id": r["pmid"], "section_name": "abstract",
                            "section_id": r["pmid"], "title": r["title"],
                            "extracted_by": "gemma-4-E4B", "pmid": r["pmid"]}
                    out.write(json.dumps(edge, ensure_ascii=False)+"\n")
                    n_edges += 1
            log(f"  {cs+len(chunk)}/{len(records)} edges={n_edges:,}")
    log(f"Done. Alt IE edges: {n_edges:,}")


if __name__ == "__main__":
    main()
