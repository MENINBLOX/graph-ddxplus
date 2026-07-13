#!/usr/bin/env python3
"""준비된 프롬프트로 KG 구축 (vLLM batch만 실행).

prepare_kg_prompts.py로 준비된 프롬프트 파일을 읽어서 vLLM 실행 후 KG 저장.
"""
from __future__ import annotations
import json, os, re, sys, time
from collections import Counter
from pathlib import Path
from vllm import LLM, SamplingParams

RESULTS_DIR = Path("pilot/results")


def main():
    if len(sys.argv) < 2:
        print("Usage: run_prepared_kg.py [bench]", flush=True); return
    bench = sys.argv[1]

    prompts_file = RESULTS_DIR / f"kg_{bench}_prompts.json"
    output = RESULTS_DIR / f"kg_{bench}_cache.json"

    if not prompts_file.exists():
        print(f"프롬프트 파일 없음: {prompts_file}", flush=True); return

    print(f"[{bench}] 로드: {prompts_file}", flush=True)
    with open(prompts_file) as f:
        data = json.load(f)
    tasks = data["tasks"]
    diseases = data["diseases"]
    parent_map = {k: set(v) for k, v in data["parent_map"].items()}
    synonym_map = {k: set(v) for k, v in data["synonym_map"].items()}
    print(f"  질환: {len(diseases)}, 프롬프트: {len(tasks):,}", flush=True)

    # vLLM
    print("\nvLLM batch...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=4096)
    convs = [[{"role": "user", "content": t["prompt"]}] for t in tasks]
    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    print(f"  완료: {time.time()-t0:.0f}초 ({len(outputs)/max(time.time()-t0,1):.1f}/s)", flush=True)

    # Parse
    all_rels = []
    for task, out in zip(tasks, outputs):
        text = out.outputs[0].text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"```json\s*", "", text); text = re.sub(r"```\s*$", "", text)
        m = re.search(r"\[[\s\S]*?\]", text)
        if not m: continue
        try: items = json.loads(m.group())
        except: continue
        for item in items:
            if not isinstance(item, dict): continue
            cui = item.get("cui", ""); rel = item.get("relation", "")
            if cui and rel and rel != "manifestation-of":
                dc = task["dc"]
                if cui not in synonym_map.get(dc, set()) and cui not in parent_map.get(dc, set()) and dc not in parent_map.get(cui, set()):
                    all_rels.append({"dc": dc, "cui": cui})

    pair_counts = Counter(tuple(sorted([r["dc"], r["cui"]])) for r in all_rels)
    print(f"관계: {len(all_rels):,}, 고유 쌍: {len(pair_counts):,}", flush=True)

    cache_data = {
        "pair_counts": [[list(k), v] for k, v in pair_counts.most_common()],
        "diseases": diseases,
        "stats": {"prompts": len(tasks), "n_diseases": len(diseases)},
    }
    with open(output, "w") as f: json.dump(cache_data, f)
    print(f"저장: {output}", flush=True)


if __name__ == "__main__":
    main()
