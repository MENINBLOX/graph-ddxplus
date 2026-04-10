#!/usr/bin/env python3
"""Regex 유무에 따른 답변 비교 테스트.

동일한 프롬프트에 대해:
1. regex 제약 없이 숫자 출력
2. regex 제약으로 숫자 강제
두 결과가 동일한지 비교.
"""

import json
import os
import random
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from src.data_loader import DDXPlusLoader


def run_regex_comparison(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    n_samples: int = 100,
    top_n: int = 5,
):
    """Regex 유무 비교 테스트."""
    print("=" * 70)
    print("Regex 유무 비교 테스트")
    print(f"모델: {model}")
    print(f"샘플 수: {n_samples}")
    print("=" * 70)

    # vLLM 엔진 초기화
    print("\nvLLM 엔진 로딩...", flush=True)
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )

    # 파라미터 설정
    # 1. Regex 없음
    params_no_regex = SamplingParams(
        temperature=0.0,
        max_tokens=10,  # 숫자만 출력하도록 짧게
    )

    # 2. Regex 있음
    structured_params = StructuredOutputsParams(regex=f"[1-{top_n}]")
    params_with_regex = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        structured_outputs=structured_params,
    )

    # 데이터 로드
    print("데이터 로딩...", flush=True)
    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test", n_samples=n_samples, severity=2)
    print(f"로드된 환자: {len(patients)}")

    # 테스트용 후보 증상 풀
    symptom_pool = [
        ("C0015967", "Fever", 0.15),
        ("C0010200", "Cough", 0.12),
        ("C0015672", "Fatigue", 0.18),
        ("C0018681", "Headache", 0.10),
        ("C0008031", "Chest Pain", 0.20),
        ("C0013404", "Dyspnea", 0.15),
        ("C0027497", "Nausea", 0.08),
        ("C0042963", "Vomiting", 0.07),
        ("C0011991", "Diarrhea", 0.06),
        ("C0000737", "Abdominal Pain", 0.14),
        ("C0231528", "Myalgia", 0.09),
        ("C0003862", "Arthralgia", 0.08),
        ("C0037763", "Sore Throat", 0.11),
        ("C0035222", "Rhinorrhea", 0.07),
        ("C0036974", "Chills", 0.10),
    ]

    results = []
    match_count = 0
    mismatch_count = 0

    # 분포 추적
    dist_no_regex = {str(i): 0 for i in range(1, top_n + 1)}
    dist_with_regex = {str(i): 0 for i in range(1, top_n + 1)}

    print(f"\n{'='*70}")
    print("테스트 시작")
    print(f"{'='*70}\n")

    for i, patient in enumerate(patients):
        # 환자별 후보 선택
        random.seed(i)
        candidates = random.sample(symptom_pool, min(top_n * 2, len(symptom_pool)))
        random.shuffle(candidates)
        candidates = candidates[:top_n]

        total_score = sum(score for _, _, score in candidates) or 1
        candidate_list = "\n".join([
            f"{j+1}. {name} ({score/total_score:.0%})"
            for j, (cui, name, score) in enumerate(candidates)
        ])

        # 프롬프트: 숫자만 출력하도록
        prompt = f"""You are a medical diagnostic assistant. Select the most informative symptom to ask next.

Patient: {patient.sex}, {patient.age} years old
Chief complaint: {patient.initial_evidence}

Candidate symptoms:
{candidate_list}

Select the best symptom number (1-{top_n}). Output ONLY the number, nothing else:"""

        # 1. Regex 없이 생성
        output_no_regex = llm.generate([prompt], params_no_regex)
        response_no_regex = output_no_regex[0].outputs[0].text.strip()

        # 숫자 추출
        numbers = re.findall(r'[1-5]', response_no_regex)
        answer_no_regex = int(numbers[0]) if numbers else None

        # 2. Regex로 생성
        try:
            output_with_regex = llm.generate([prompt], params_with_regex)
            answer_with_regex = int(output_with_regex[0].outputs[0].text.strip())
        except Exception as e:
            print(f"  [{i+1}] Regex 에러: {e}")
            answer_with_regex = 1

        # 비교
        is_match = answer_no_regex == answer_with_regex
        if is_match:
            match_count += 1
        else:
            mismatch_count += 1

        # 분포 업데이트
        if answer_no_regex and 1 <= answer_no_regex <= top_n:
            dist_no_regex[str(answer_no_regex)] += 1
        if 1 <= answer_with_regex <= top_n:
            dist_with_regex[str(answer_with_regex)] += 1

        result = {
            "patient_id": i,
            "prompt": prompt,
            "response_no_regex": response_no_regex,
            "answer_no_regex": answer_no_regex,
            "answer_with_regex": answer_with_regex,
            "is_match": is_match,
        }
        results.append(result)

        # 출력
        status = "✅" if is_match else "❌ MISMATCH"
        print(f"[{i+1:3d}/{n_samples}] No-Regex: {answer_no_regex} | With-Regex: {answer_with_regex} {status}")

        if not is_match:
            print(f"       Raw output: '{response_no_regex}'")

    # 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    print(f"\n[일치/불일치]")
    print(f"  일치: {match_count}/{n_samples} ({100*match_count/n_samples:.1f}%)")
    print(f"  불일치: {mismatch_count}/{n_samples} ({100*mismatch_count/n_samples:.1f}%)")

    print(f"\n[Selection 분포 - Regex 없음]")
    for sel in sorted(dist_no_regex.keys()):
        count = dist_no_regex[sel]
        pct = 100 * count / n_samples
        bar = "█" * int(pct / 5)
        print(f"  {sel}번: {count:3d} ({pct:5.1f}%) {bar}")

    print(f"\n[Selection 분포 - Regex 있음]")
    for sel in sorted(dist_with_regex.keys()):
        count = dist_with_regex[sel]
        pct = 100 * count / n_samples
        bar = "█" * int(pct / 5)
        print(f"  {sel}번: {count:3d} ({pct:5.1f}%) {bar}")

    # 결과 저장
    model_short = model.split("/")[-1].replace("-", "_").lower()
    output_path = Path(f"results/regex_comparison_{model_short}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": model,
            "n_samples": n_samples,
            "match_rate": match_count / n_samples,
            "dist_no_regex": dist_no_regex,
            "dist_with_regex": dist_with_regex,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {output_path}")

    return results


if __name__ == "__main__":
    run_regex_comparison()
