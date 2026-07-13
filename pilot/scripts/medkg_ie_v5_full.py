#!/usr/bin/env python3
"""v5 IE for full corpus, partitioned for parallel GPU use."""
from __future__ import annotations
import os, sys, json, re, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from medkg_ie_v5_prevalence import IE_PROMPT, parse_prevalence_v5

PUB_DIR = MEDKG_ROOT / "pubmed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuis_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_abstracts_per_cui", type=int, default=10)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args()

    target_cuis = []
    cui_to_name = {}
    with open(args.cuis_file) as f:
        for i, line in enumerate(f):
            if i % args.of != args.part: continue
            e = json.loads(line)
            target_cuis.append(e["cui"])
            cui_to_name[e["cui"]] = e.get("primary_name", e.get("name", ""))
    print(f"[part {args.part}/{args.of}] Target CUIs: {len(target_cuis)}", flush=True)

    records = []
    for cui in target_cuis:
        fp = PUB_DIR / f"{cui}.jsonl"
        if not fp.exists() or fp.stat().st_size == 0: continue
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
                    "disease": e.get("disease_name", cui_to_name.get(cui, cui)),
                    "cui": cui, "pmid": e.get("pmid", ""),
                    "title": e.get("title", ""), "abstract": ab[:2500],
                })
                cnt += 1
                if cnt >= args.max_abstracts_per_cui: break
    print(f"Loaded {len(records)} abstracts", flush=True)
    if not records: return

    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    print("Loading vLLM 0.21...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=600)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_edges = 0
    BATCH = 256
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
                for l1, l2, feat, freq_q, prev in parse_prevalence_v5(text):
                    edge = {"disease": r["disease"], "umls_cui": r["cui"],
                            "phenotype": feat, "level1": l1, "level2": l2,
                            "category": l2, "freq_qualifier": freq_q,
                            "prevalence": prev, "source": "pubmed_v5",
                            "pmid": r["pmid"], "extracted_by": "gemma-4-E4B-v5"}
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
            elapsed = time.time() - t0
            rate = (cs + len(chunk)) / max(elapsed, 1)
            eta = (len(records) - cs - len(chunk)) / max(rate, 1)
            print(f"  [part {args.part}] {cs+len(chunk)}/{len(records)} edges={n_edges:,} ETA={eta/60:.0f}min", flush=True)
    print(f"Done. v5 edges: {n_edges:,} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
