#!/usr/bin/env python3
"""v103 shard runner — single-GPU process multiple diseases sequentially."""
import os, sys, json, argparse, time, re
from pathlib import Path
from collections import defaultdict, Counter

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")

sys.path.insert(0, str(Path(__file__).parent))
from v103_grounded_ie import IEOutputGrounded
from v103_batch_ie import PROMPT_TPL, post_validate, aggregate_disease


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pubmed_dir", default="/mnt/medkg/pubmed")
    ap.add_argument("--max_abstracts", type=int, default=20)
    ap.add_argument("--greedy", action="store_true",
                    help="temperature=0 deterministic decoding (reproducible)")
    args = ap.parse_args()

    diseases = []
    with open(args.shard_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2: diseases.append((parts[0], parts[1]))  # cui, name (ignore extra cols)
    print(f"Shard: {len(diseases)} diseases", flush=True)

    # Load vLLM ONCE
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    print("Loading vLLM...", flush=True)
    schema = IEOutputGrounded.model_json_schema()
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=8192, gpu_memory_utilization=0.85,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    if args.greedy:
        sampling = SamplingParams(
            temperature=0.0, max_tokens=3072,
            structured_outputs=StructuredOutputsParams(json=schema)
        )
        print("  Greedy decoding (temperature=0, reproducible)", flush=True)
    else:
        sampling = SamplingParams(
            temperature=0.2, max_tokens=3072, top_p=0.9,
            structured_outputs=StructuredOutputsParams(json=schema)
        )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for i, (cui, dn) in enumerate(diseases):
        out_path = f"{args.out_dir}/{cui}.json"
        if os.path.exists(out_path):
            print(f"  [{i+1}/{len(diseases)}] {dn} ({cui}): skip (exists)", flush=True)
            continue

        pm_path = f"{args.pubmed_dir}/{cui}.jsonl"
        if not os.path.exists(pm_path):
            print(f"  [{i+1}/{len(diseases)}] {dn} ({cui}): no pubmed file", flush=True)
            continue

        abstracts = []
        with open(pm_path) as f:
            for line in f:
                try:
                    a = json.loads(line)
                    if a.get("abstract"):
                        abstracts.append(a)
                        if len(abstracts) >= args.max_abstracts: break
                except: continue
        if not abstracts:
            print(f"  [{i+1}/{len(diseases)}] {dn}: 0 abstracts", flush=True)
            continue

        prompts = [PROMPT_TPL.format(source_text=a["abstract"][:2500], disease=dn)
                   for a in abstracts]
        convs = [[{"role":"user", "content": p}] for p in prompts]
        t0 = time.time()
        outs = llm.chat(convs, sampling)
        elapsed = time.time() - t0

        per_abs_results = []
        n_parsed = 0; n_dropped = 0
        for out in outs:
            try:
                parsed = json.loads(out.outputs[0].text)
                validated = IEOutputGrounded(**parsed)
                n_parsed += 1
                for p in validated.phenotypes: n_dropped += post_validate(p)
                per_abs_results.append(validated.phenotypes)
            except: continue

        aggregated = aggregate_disease(per_abs_results)
        with open(out_path, "w") as f:
            json.dump({
                "disease": dn, "cui": cui,
                "n_abstracts": len(abstracts), "n_parsed": n_parsed,
                "n_attrs_dropped": n_dropped,
                "aggregated": aggregated
            }, f, indent=2)
        print(f"  [{i+1}/{len(diseases)}] {dn}: {len(abstracts)}abs → {n_parsed}parsed, "
              f"{len(aggregated)}phens, {n_dropped}dropped ({elapsed:.1f}s)", flush=True)

    total = time.time() - t_start
    print(f"\nShard done: {total:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
