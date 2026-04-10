#!/usr/bin/env python3
"""Instruct 모델 Two-Stage GTPA@1 벤치마크.

Thinking 모델과 동일한 조건에서 Instruct 모델의 GTPA@1을 측정.
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

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from src.data_loader import DDXPlusLoader, Patient


def build_prompt(
    patient: Patient,
    candidates: list[tuple[str, str, float]],
    top_n: int = 5,
) -> str:
    """프롬프트 생성."""
    total_score = sum(score for _, _, score in candidates) or 1

    candidate_list = "\n".join([
        f"{i+1}. {name} ({score/total_score:.0%})"
        for i, (cui, name, score) in enumerate(candidates)
    ])

    prompt = f"""You are a medical diagnostic assistant. Select the most informative symptom to ask next.

Patient: {patient.sex}, {patient.age} years old
Chief complaint: {patient.initial_evidence}
Confirmed symptoms: None
Denied symptoms: None

Candidate symptoms to inquire:
{candidate_list}

Analyze step by step:
1. Consider the patient's demographics and chief complaint
2. Evaluate each candidate's diagnostic value for differential diagnosis
3. Select the single most informative symptom

Respond in this EXACT format:
REASONING: [Your step-by-step clinical reasoning in 2-3 sentences]
ANSWER: [Single number only, 1-{top_n}]"""

    return prompt


def extract_answer(response: str, top_n: int = 5) -> int | None:
    """응답에서 ANSWER 추출."""
    # ANSWER 추출
    answer_match = re.search(r'ANSWER:\s*(\d+)', response, re.IGNORECASE)
    if answer_match:
        num = int(answer_match.group(1))
        if 1 <= num <= top_n:
            return num

    # Fallback: 마지막 숫자
    numbers = re.findall(r'\b([1-5])\b', response[-100:])
    if numbers:
        num = int(numbers[-1])
        if 1 <= num <= top_n:
            return num

    return None


def run_instruct_gtpa_benchmark(
    model: str = "openai/gpt-oss-20b",
    n_samples: int = 100,
    shuffle: bool = True,
    top_n: int = 5,
):
    """Instruct 모델 GTPA@1 벤치마크."""
    print("=" * 70)
    print("Instruct 모델 Two-Stage GTPA@1 벤치마크")
    print(f"모델: {model}")
    print(f"샘플 수: {n_samples}")
    print(f"셔플: {shuffle}")
    print(f"Top-N: {top_n}")
    print("=" * 70)

    # 토크나이저 로드
    print("\n토크나이저 로딩...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    # vLLM 엔진 초기화
    print("vLLM 엔진 로딩...", flush=True)
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )

    # Stage 1: 추론 생성
    sampling_params_reason = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )

    # Stage 2: 숫자 선택
    structured_params = StructuredOutputsParams(regex=f"[1-{top_n}]")
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

    # 테스트용 후보 증상 풀 (Thinking 모델과 동일)
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

    # 결과 저장
    results = []
    token_counts = []
    correct_count = 0
    selection_distribution = {str(i): 0 for i in range(1, top_n + 1)}
    stage1_stage2_match = 0
    total_with_answer = 0

    print(f"\n{'='*70}")
    print("벤치마크 시작")
    print(f"{'='*70}\n")

    for i, patient in enumerate(patients):
        print(f"\n[{i+1}/{len(patients)}] 환자: {patient.sex}, {patient.age}세, 주증상: {patient.initial_evidence}")

        # 환자별 후보 선택 (Thinking 모델과 동일한 시드)
        random.seed(i)
        candidates = random.sample(symptom_pool, min(top_n * 2, len(symptom_pool)))

        if shuffle:
            candidates = list(candidates)
            random.shuffle(candidates)

        candidates = candidates[:top_n]

        # 정답 설정 (Thinking 모델과 동일)
        ground_truth_idx = random.randint(0, top_n - 1)
        ground_truth_symptom = candidates[ground_truth_idx][1]

        # 프롬프트 생성
        prompt = build_prompt(
            patient=patient,
            candidates=candidates,
            top_n=top_n,
        )

        # Stage 1: 추론 생성
        stage1_output = llm.generate([prompt], sampling_params_reason)
        response = stage1_output[0].outputs[0].text.strip()

        # 토큰 수
        reason_tokens = len(tokenizer.encode(response))
        token_counts.append(reason_tokens)

        # Stage 1 답변 추출
        answer_from_stage1 = extract_answer(response, top_n)

        # Stage 2: 숫자 선택
        stage2_content = f"""{prompt}

{response}

Based on your reasoning above, confirm your final answer with just the number (1-{top_n}):"""

        try:
            stage2_output = llm.generate([stage2_content], sampling_params_select)
            stage2_selection = int(stage2_output[0].outputs[0].text.strip())
        except Exception as e:
            print(f"  Stage 2 에러: {e}")
            stage2_selection = 1

        # 정확도 계산
        is_correct = stage2_selection == (ground_truth_idx + 1)
        if is_correct:
            correct_count += 1

        # 선택 분포
        selection_distribution[str(stage2_selection)] += 1

        # Stage 1-2 일치
        if answer_from_stage1:
            total_with_answer += 1
            if answer_from_stage1 == stage2_selection:
                stage1_stage2_match += 1

        # 결과 저장
        result = {
            "patient_id": i,
            "candidates": [(c[0], c[1], float(c[2])) for c in candidates],
            "ground_truth_idx": ground_truth_idx + 1,
            "ground_truth_symptom": ground_truth_symptom,
            "response": response,
            "answer_from_stage1": answer_from_stage1,
            "stage2_selection": stage2_selection,
            "is_correct": is_correct,
            "reason_tokens": reason_tokens,
        }
        results.append(result)

        # 출력
        print(f"  후보: {[c[1] for c in candidates]}")
        print(f"  정답: {ground_truth_idx + 1}번 ({ground_truth_symptom})")
        print(f"  Stage 1 답변: {answer_from_stage1}")
        print(f"  Stage 2 답변: {stage2_selection}")
        print(f"  정답 여부: {'✅ CORRECT' if is_correct else '❌ WRONG'}")
        print(f"  토큰: {reason_tokens}")

    # 통계 계산
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    std_tokens = (sum((x - avg_tokens) ** 2 for x in token_counts) / len(token_counts)) ** 0.5 if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0

    gtpa_at_1 = correct_count / len(results) if results else 0

    print(f"\n[토큰 통계]")
    print(f"  평균: {avg_tokens:.1f} ± {std_tokens:.1f}")
    print(f"  범위: {min_tokens} ~ {max_tokens}")

    print(f"\n[정확도]")
    print(f"  GTPA@1: {correct_count}/{len(results)} ({100*gtpa_at_1:.1f}%)")

    print(f"\n[Stage 1-2 일치율]")
    if total_with_answer > 0:
        print(f"  일치: {stage1_stage2_match}/{total_with_answer} ({100*stage1_stage2_match/total_with_answer:.1f}%)")

    print(f"\n[Selection 분포]")
    for sel in sorted(selection_distribution.keys()):
        count = selection_distribution[sel]
        pct = 100 * count / len(results) if results else 0
        bar = "█" * int(pct / 5)
        print(f"  {sel}번: {count:3d} ({pct:5.1f}%) {bar}")

    random_baseline = 100 / top_n
    print(f"\n[기준선 비교]")
    print(f"  랜덤 기준선: {random_baseline:.1f}%")
    print(f"  GTPA@1: {100*gtpa_at_1:.1f}%")
    print(f"  개선: {100*gtpa_at_1 - random_baseline:+.1f}%p")

    # 결과 저장
    model_short = model.split("/")[-1].replace("-", "_").lower()
    output_path = Path(f"results/instruct_gtpa_{model_short}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": model,
            "n_samples": len(results),
            "shuffle": shuffle,
            "top_n": top_n,
            "tokens": {
                "avg": avg_tokens,
                "std": std_tokens,
                "min": min_tokens,
                "max": max_tokens,
            },
            "gtpa_at_1": gtpa_at_1,
            "stage1_stage2_match_rate": stage1_stage2_match / total_with_answer if total_with_answer else 0,
            "selection_distribution": selection_distribution,
            "random_baseline": random_baseline / 100,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {output_path}")

    return results


if __name__ == "__main__":
    run_instruct_gtpa_benchmark()
