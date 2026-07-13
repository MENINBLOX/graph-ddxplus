#!/usr/bin/env python3
"""KG 노이즈 제거: LLM 기반 쌍 검증.

각 disease-symptom 쌍을 LLM이 검증:
"Is X a typical CLINICAL SYMPTOM that patients with Y would PRESENT WITH?"
Yes/No 답변으로 KG 정제.
"""
from __future__ import annotations
import json, os, re, time
from collections import Counter, defaultdict
from pathlib import Path
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")
INPUT_KG = Path("pilot/results/kg_v3_cache.json")
OUTPUT_KG = Path("pilot/results/kg_v3_verified.json")

PROMPT = """For each disease-symptom pair, answer if the symptom is a TYPICAL clinical symptom that patients with the disease commonly present with.

Pairs:
{pairs}

For each, answer YES or NO (one per line, in order):"""


def main():
    print("="*80, flush=True)
    print("KG 검증: LLM 기반 노이즈 제거", flush=True)
    print("="*80, flush=True)

    # CUI names
    print("CUI names...", flush=True)
    cp = {}
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG" and p[2] == "P" and p[0] not in cp:
                cp[p[0]] = p[14].strip()

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open(INPUT_KG) as f: cache = json.load(f)

    pc = Counter()
    for k, v in cache["pair_counts"]: pc[tuple(k)] = v

    dcs = set()
    cui2name = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]; dcs.add(dc); cui2name[dc] = dn

    # 검증 대상: count >= 3인 모든 쌍 (너무 작은 건 신뢰도 낮음)
    pairs_to_verify = []
    for (a, b), cnt in pc.items():
        if cnt < 3: continue
        # Disease와 symptom 식별
        dc = a if a in dcs else (b if b in dcs else None)
        if not dc: continue
        sym = b if a == dc else a
        if sym in dcs: continue  # disease-disease 쌍은 제외 (이미 KG에서 처리)
        pairs_to_verify.append((dc, sym, cnt))

    print(f"검증 대상: {len(pairs_to_verify):,} 쌍 (count >= 3)", flush=True)

    # 청크로 나누기 (한 프롬프트당 20쌍)
    CHUNK = 20
    chunks = [pairs_to_verify[i:i+CHUNK] for i in range(0, len(pairs_to_verify), CHUNK)]
    print(f"  청크: {len(chunks):,}개", flush=True)

    prompts = []
    for chunk in chunks:
        pair_lines = []
        for i, (dc, sym, cnt) in enumerate(chunk):
            dname = cui2name.get(dc, cp.get(dc, dc))
            sname = cp.get(sym, sym)
            pair_lines.append(f"{i+1}. {sname} (symptom of {dname})")
        prompts.append(PROMPT.format(pairs="\n".join(pair_lines)))

    # vLLM
    print("\nvLLM batch...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=128)
    convs = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    print(f"  완료: {time.time()-t0:.0f}초", flush=True)

    # 파싱
    verified = []
    for chunk, out in zip(chunks, outputs):
        text = out.outputs[0].text.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # YES/NO 추출
        answers = re.findall(r"(?:^|\s)(\d+)[.\):]\s*(YES|NO|yes|no)", text)
        # answers를 dict로
        ans_map = {int(num): ans.upper() == "YES" for num, ans in answers}
        for i, (dc, sym, cnt) in enumerate(chunk):
            if ans_map.get(i+1, False):  # YES인 것만
                verified.append((dc, sym, cnt))

    print(f"\n검증 통과: {len(verified):,} / {len(pairs_to_verify):,} ({100*len(verified)/len(pairs_to_verify):.0f}%)", flush=True)

    # 저장
    pair_counts_dict = Counter()
    for dc, sym, cnt in verified:
        pair_counts_dict[tuple(sorted([dc, sym]))] = cnt

    save_data = {
        "pair_counts": [[list(k), v] for k, v in pair_counts_dict.most_common()],
        "diseases": cache.get("diseases", {}),
        "stats": {"original_pairs": len(pc), "verified_pairs": len(verified)},
    }
    with open(OUTPUT_KG, "w") as f: json.dump(save_data, f)
    print(f"저장: {OUTPUT_KG}", flush=True)


if __name__ == "__main__":
    main()
