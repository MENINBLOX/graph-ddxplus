#!/usr/bin/env python3
"""환자 evidence 프랑스어 이름 → 영어 의학 용어 번역.

release_evidences.json 사용하지 않음.
환자 데이터(release_test_patients.csv)에서 evidence 이름만 추출하여
LLM으로 일반적인 프랑스어→영어 의학 번역 수행.
"""
import ast
import csv
import json
import os
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams

RESULTS_DIR = Path("pilot/results")
TRANSLATE_CACHE = RESULTS_DIR / "evidence_fr_to_en.json"

PROMPT = """Translate these French medical abbreviations/terms to English medical terms.
These are symptom or medical history codes. Give the most likely English medical term for each.

{terms}

JSON format: {{"term1": "English medical term", "term2": "English medical term", ...}}"""


def main():
    print("=" * 80, flush=True)
    print("Evidence 프랑스어→영어 번역 (release_evidences.json 미사용)", flush=True)
    print("=" * 80, flush=True)

    if TRANSLATE_CACHE.exists():
        print("캐시 존재, 건너뜀", flush=True)
        with open(TRANSLATE_CACHE) as f:
            translations = json.load(f)
        print(f"  {len(translations)}개 번역", flush=True)
        return

    # 환자 데이터에서 evidence 이름 추출
    ev_names = set()
    for split in ["release_test_patients.csv", "release_train_patients.csv"]:
        path = f"data/ddxplus/{split}"
        try:
            with open(path) as f:
                for i, row in enumerate(csv.DictReader(f)):
                    if i >= 50000: break
                    for ev in ast.literal_eval(row["EVIDENCES"]):
                        ev_names.add(ev.split("_@_")[0])
        except FileNotFoundError:
            pass

    # value도 추출
    ev_values = set()
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= 50000: break
            for ev in ast.literal_eval(row["EVIDENCES"]):
                parts = ev.split("_@_")
                if len(parts) > 1:
                    ev_values.add(parts[1])

    fr_terms = sorted(ev_names)
    fr_values = sorted(ev_values)[:100]  # top 100 values
    print(f"Evidence base 이름: {len(fr_terms)}", flush=True)
    print(f"Evidence values (top 100): {len(fr_values)}", flush=True)

    # LLM 번역 (청크로)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=4096)

    # Base names 번역
    CHUNK = 30
    all_translations = {}

    chunks = [fr_terms[i:i+CHUNK] for i in range(0, len(fr_terms), CHUNK)]
    prompts = []
    for chunk in chunks:
        terms_str = "\n".join(f"- {t}" for t in chunk)
        prompts.append(PROMPT.format(terms=terms_str))

    # Values도 번역
    val_chunks = [fr_values[i:i+CHUNK] for i in range(0, len(fr_values), CHUNK)]
    for chunk in val_chunks:
        terms_str = "\n".join(f"- {t}" for t in chunk)
        prompts.append(PROMPT.format(terms=terms_str))

    print(f"\nLLM 번역 ({len(prompts)} 프롬프트)...", flush=True)
    convs = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    print(f"  완료: {time.time()-t0:.0f}초", flush=True)

    # 파싱
    for out in outputs:
        text = out.outputs[0].text.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                d = json.loads(m.group())
                all_translations.update(d)
            except Exception:
                pass

    print(f"  번역 완료: {len(all_translations)}개", flush=True)

    with open(TRANSLATE_CACHE, "w") as f:
        json.dump(all_translations, f, indent=2, ensure_ascii=False)
    print(f"  저장: {TRANSLATE_CACHE}", flush=True)

    # 샘플 출력
    print("\n번역 샘플:", flush=True)
    for k in sorted(all_translations)[:20]:
        print(f"  {k:<35} → {all_translations[k]}", flush=True)


if __name__ == "__main__":
    main()
