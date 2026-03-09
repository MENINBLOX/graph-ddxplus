#!/usr/bin/env python3
"""무제한 토큰 Two-Stage 벤치마크.

100개 케이스에서 max_tokens 제한 없이 추론하고, 평균 토큰 수를 계산합니다.
"""

import json
import os
import random
import re
import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from src.data_loader import DDXPlusLoader, Patient


def run_unlimited_benchmark(
    model: str = "Qwen/Qwen3-4B-Thinking-2507",
    n_samples: int = 100,
    shuffle: bool = True,
):
    """무제한 토큰 벤치마크 실행."""
    print("=" * 70)
    print("무제한 토큰 Two-Stage 벤치마크")
    print(f"모델: {model}")
    print(f"샘플 수: {n_samples}")
    print(f"셔플: {shuffle}")
    print("=" * 70)

    # 토크나이저 로드
    print("\n토크나이저 로딩...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    # vLLM 엔진 초기화 (무제한 토큰을 위해 큰 max_model_len)
    print("vLLM 엔진 로딩...", flush=True)
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=32768,  # 충분히 큰 값
        gpu_memory_utilization=0.9,
    )

    # Stage 1: 무제한 추론 (stop token으로 자연스럽게 종료)
    sampling_params_reason = SamplingParams(
        temperature=0.0,
        max_tokens=16384,  # 매우 큰 값 (실제로는 stop token에서 멈춤)
    )

    # Stage 2: 숫자 선택 (structured_outputs 사용)
    structured_params = StructuredOutputsParams(regex="[1-5]")
    sampling_params_select = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        structured_outputs=structured_params,
    )

    # 데이터 로드
    print("데이터 로딩...", flush=True)
    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test", n_samples=n_samples, severity=2)
    print(f"로드된 환자: {len(patients)}")

    # 결과 저장
    results = []
    token_counts = []
    stage1_stage2_mismatches = 0
    total_analyzed = 0
    selection_distribution = {str(i): 0 for i in range(1, 6)}

    print(f"\n{'='*70}")
    print("벤치마크 시작")
    print(f"{'='*70}\n")

    for i, patient in enumerate(patients):
        print(f"\n[{i+1}/{len(patients)}] 환자 처리 중...", flush=True)

        # 간단한 프롬프트 생성 (첫 번째 라운드 - 증상 선택)
        # 환자 초기 증상
        initial_symptoms = [patient.initial_evidence] if patient.initial_evidence else []
        age = patient.age
        sex = patient.sex

        # 가상의 후보 증상 생성 (실제 벤치마크에서는 KG에서 가져옴)
        # 여기서는 간단히 테스트용 후보 사용
        candidate_symptoms = ["Fever", "Cough", "Fatigue", "Headache", "Chest Pain"]
        if shuffle:
            random.shuffle(candidate_symptoms)

        # 프롬프트 생성
        prompt = f"""You are a medical diagnostic assistant. A patient presents with the following information:

Age: {age}
Sex: {sex}
Chief complaint: {patient.initial_evidence}

Candidate symptoms to inquire (select the most informative for differential diagnosis):
1. {candidate_symptoms[0]} (20%)
2. {candidate_symptoms[1]} (20%)
3. {candidate_symptoms[2]} (20%)
4. {candidate_symptoms[3]} (20%)
5. {candidate_symptoms[4]} (20%)

Based on clinical reasoning, select the most diagnostically valuable symptom.
Brief reason for your selection:"""

        # Stage 1: 무제한 추론
        stage1_output = llm.generate([prompt], sampling_params_reason)
        reason = stage1_output[0].outputs[0].text.strip()

        # 토큰 수 계산
        reason_tokens = len(tokenizer.encode(reason))
        token_counts.append(reason_tokens)

        # Stage 2: 숫자 선택
        stage2_content = f"""{prompt}

Your reasoning: {reason}

Based on your reasoning, respond with just the number (1-5):"""

        try:
            stage2_output = llm.generate([stage2_content], sampling_params_select)
            selection = stage2_output[0].outputs[0].text.strip()
        except Exception as e:
            print(f"  Stage 2 에러: {e}")
            # Fallback: regex 없이 생성
            fallback_params = SamplingParams(temperature=0.0, max_tokens=5)
            stage2_output = llm.generate([stage2_content], fallback_params)
            selection_text = stage2_output[0].outputs[0].text.strip()
            numbers = re.findall(r"[1-5]", selection_text)
            selection = numbers[0] if numbers else "1"

        # Selection 분포 업데이트
        if selection in selection_distribution:
            selection_distribution[selection] += 1

        # Stage 1에서 언급된 번호 추출
        mentioned = re.findall(
            r'(?:option|select|choice|answer|number|choose|pick|recommend)\s*(?:is\s*)?(\d)',
            reason.lower()
        )
        inferred = mentioned[-1] if mentioned else None

        # 불일치 확인
        total_analyzed += 1
        mismatch = inferred and inferred != selection
        if mismatch:
            stage1_stage2_mismatches += 1

        # 결과 저장
        result = {
            "patient_id": i,
            "candidates": candidate_symptoms,
            "reason": reason,
            "reason_tokens": reason_tokens,
            "selection": selection,
            "inferred_from_reason": inferred,
            "mismatch": mismatch,
        }
        results.append(result)

        # 출력
        print(f"  후보: {candidate_symptoms}")
        print(f"  추론 토큰: {reason_tokens}")
        print(f"  추론 (처음 300자): {reason[:300]}...")
        print(f"  추론 (마지막 200자): ...{reason[-200:]}")
        print(f"  Stage 1 추론 결론: {inferred}")
        print(f"  Stage 2 선택: {selection}")
        print(f"  불일치: {'❌ YES' if mismatch else '✅ NO'}")

    # 통계 계산
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0

    print(f"\n[추론 토큰 통계]")
    print(f"  평균: {avg_tokens:.1f} tokens")
    print(f"  최소: {min_tokens} tokens")
    print(f"  최대: {max_tokens} tokens")

    print(f"\n[Stage 2 Selection 분포]")
    for sel in sorted(selection_distribution.keys()):
        count = selection_distribution[sel]
        pct = 100 * count / len(results) if results else 0
        print(f"  {sel}번: {count} ({pct:.1f}%)")

    print(f"\n[Stage 1-2 불일치]")
    print(f"  불일치 케이스: {stage1_stage2_mismatches}/{total_analyzed}")
    if total_analyzed > 0:
        print(f"  불일치율: {100 * stage1_stage2_mismatches / total_analyzed:.1f}%")

    # 결과 저장
    output_path = Path("results/unlimited_tokens_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": model,
            "n_samples": len(results),
            "shuffle": shuffle,
            "avg_tokens": avg_tokens,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "mismatch_rate": stage1_stage2_mismatches / total_analyzed if total_analyzed else 0,
            "selection_distribution": selection_distribution,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {output_path}")

    return results


if __name__ == "__main__":
    run_unlimited_benchmark()
