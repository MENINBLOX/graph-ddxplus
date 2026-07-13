#!/usr/bin/env python3
"""Test Gemma-4-E4B-it MTP speedup vs baseline on small IE batch.

vLLM speculative config:
  --speculative_config method=mtp model=google/gemma-4-E4B-it-assistant num_speculative_tokens=5
"""
from __future__ import annotations
import os, sys, json, time, argparse
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from medkg_ie_universal_v3 import IE_PROMPT, parse_categorized_v3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--mtp_model", default="google/gemma-4-E4B-it-assistant",
                    help="speculative MTP assistant model")
    ap.add_argument("--num_spec", type=int, default=4)
    args = ap.parse_args()

    # Load some abstracts
    pub_dir = MEDKG_ROOT / "pubmed"
    files = sorted(pub_dir.glob("*.jsonl"))[:10]
    records = []
    for fp in files:
        if fp.stat().st_size == 0: continue
        with fp.open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: e = json.loads(line)
                except: continue
                ab = (e.get("abstract") or "").strip()
                if len(ab) < 50: continue
                records.append({
                    "disease": e.get("disease_name", fp.stem),
                    "cui": fp.stem,
                    "abstract": ab[:2500]
                })
                if len(records) >= args.n_prompts: break
        if len(records) >= args.n_prompts: break
    print(f"Loaded {len(records)} abstracts for benchmark", flush=True)

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model="google/gemma-4-E4B-it", dtype="bfloat16",
        max_model_len=4096, gpu_memory_utilization=0.85,
        enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0},
    )
    if args.mtp:
        llm_kwargs["speculative_config"] = {
            "model": args.mtp_model,
            "method": "mtp",
            "num_speculative_tokens": args.num_spec,
        }
        print(f"MTP enabled: {args.mtp_model}, num_spec={args.num_spec}", flush=True)
    else:
        print("MTP disabled (baseline)", flush=True)

    print("Loading vLLM...", flush=True)
    t_load = time.time()
    llm = LLM(**llm_kwargs)
    t_load = time.time() - t_load
    print(f"vLLM loaded in {t_load:.0f}s", flush=True)

    sampling = SamplingParams(temperature=0, max_tokens=500)
    convs = [[{"role": "user", "content": IE_PROMPT.format(
        disease=r["disease"], section="abstract", text=r["abstract"]
    )}] for r in records]

    # Warm up
    print("Warmup (10 prompts)...", flush=True)
    _ = llm.chat(convs[:10], sampling)

    # Benchmark
    print(f"Benchmark on {len(convs)} prompts...", flush=True)
    t0 = time.time()
    outs = llm.chat(convs, sampling)
    elapsed = time.time() - t0
    n_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"\n=== Result ===", flush=True)
    print(f"  Mode: {'MTP' if args.mtp else 'baseline'}", flush=True)
    print(f"  Prompts: {len(convs)}", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    print(f"  Tokens out: {n_tokens:,}", flush=True)
    print(f"  Throughput: {n_tokens/elapsed:.0f} tok/s", flush=True)
    print(f"  Prompts/sec: {len(convs)/elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
