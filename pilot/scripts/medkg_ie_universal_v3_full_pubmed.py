#!/usr/bin/env python3
"""Run v3 6-way categorized IE on the FULL existing PubMed corpus.

Reads all .jsonl files in /mnt/medkg/pubmed/ (38K+ files, ~734K abstracts).
Each file = 1 disease CUI. Filters out already-processed CUIs.

Output: /mnt/medkg/processed/edges_universal_v3_pubmed_full_part{N}.jsonl

Usage (3-GPU parallel):
  CUDA_VISIBLE_DEVICES=0 python ...v3_full_pubmed.py --part 0 --of 3
  CUDA_VISIBLE_DEVICES=1 python ...v3_full_pubmed.py --part 1 --of 3
  CUDA_VISIBLE_DEVICES=2 python ...v3_full_pubmed.py --part 2 --of 3
"""
from __future__ import annotations
import os, sys, json, re, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from medkg_ie_universal_v3 import IE_PROMPT, parse_categorized_v3

PUBMED_DIR = MEDKG_ROOT / "pubmed"
PUBMED_ALT_DIR = MEDKG_ROOT / "pubmed_alt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_abstracts_per_cui", type=int, default=20)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--skip_cuis", help="JSON list of CUIs to skip (already processed)")
    args = ap.parse_args()

    # Skip already-processed CUIs
    skip = set()
    if args.skip_cuis:
        skip = set(json.load(open(args.skip_cuis)))

    # Enumerate CUI files
    cui_files = sorted([f for f in PUBMED_DIR.glob("*.jsonl") if f.stem not in skip])
    # Partition
    cui_files = [f for i, f in enumerate(cui_files) if i % args.of == args.part]
    print(f"[part {args.part}/{args.of}] {len(cui_files)} CUI files to process", flush=True)

    # Load abstracts
    records = []
    for fp in cui_files:
        if fp.stat().st_size == 0: continue
        cui = fp.stem
        with fp.open() as f:
            cnt = 0
            for line in f:
                line = line.strip()
                if not line: continue
                try: e = json.loads(line)
                except: continue
                ab = (e.get("abstract") or "").strip()
                if len(ab) < 50: continue
                records.append({
                    "disease": e.get("disease_name", cui),
                    "cui": cui,
                    "pmid": e.get("pmid", ""),
                    "title": e.get("title", ""),
                    "abstract": ab[:2500],
                })
                cnt += 1
                if cnt >= args.max_abstracts_per_cui: break
    print(f"  loaded {len(records)} abstracts", flush=True)

    if not records:
        print("  nothing to process", flush=True)
        return

    print("Loading vLLM...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem,
              tensor_parallel_size=args.tp, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=500)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_edges = 0
    BATCH = 512
    t0 = time.time()
    with open(args.out, "w") as out:
        for cs in range(0, len(records), BATCH):
            chunk = records[cs:cs+BATCH]
            convs = [[{"role": "user", "content": IE_PROMPT.format(
                disease=r["disease"], section="abstract", text=r["abstract"]
            )}] for r in chunk]
            outs = llm.chat(convs, sampling)
            for r, o in zip(chunk, outs):
                try: text = o.outputs[0].text
                except: text = ""
                for cat, feat in parse_categorized_v3(text):
                    edge = {"disease": r["disease"], "umls_cui": r["cui"],
                            "phenotype": feat, "category": cat,
                            "source": "pubmed_full", "source_id": r["pmid"],
                            "pmid": r["pmid"], "title": r["title"],
                            "extracted_by": "gemma-4-E4B-v3"}
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
            elapsed = time.time() - t0
            rate = (cs + len(chunk)) / max(elapsed, 1)
            eta = (len(records) - cs - len(chunk)) / max(rate, 1)
            print(f"  [part {args.part}] {cs+len(chunk)}/{len(records)} "
                  f"edges={n_edges:,} ETA={eta/60:.0f}min", flush=True)
    print(f"Done. Edges: {n_edges:,} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
