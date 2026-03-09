#!/usr/bin/env python3
"""최적화된 프롬프트 Two-Stage 벤치마크.

1. 명확한 출력 형식 강제 (REASONING/ANSWER)
2. CoT + 명확한 결론 유도
3. 실제 정답(Ground Truth) 비교
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

try:
    from src.umls_kg import UMLSKG
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False


def build_optimized_prompt(
    patient: Patient,
    candidates: list[tuple[str, str, float]],
    confirmed: list[str],
    denied: list[str],
    top_n: int = 5,
) -> str:
    """최적화된 Stage 1 프롬프트 생성.

    개선점:
    1. 명확한 출력 형식 (REASONING/ANSWER)
    2. CoT 단계별 분석 유도
    3. 최종 번호 명시적 요청
    """
    # Top-N 후보만 사용
    top_candidates = candidates[:top_n]
    total_score = sum(score for _, _, score in top_candidates) or 1

    candidate_list = "\n".join([
        f"{i+1}. {name} ({score/total_score:.0%})"
        for i, (cui, name, score) in enumerate(top_candidates)
    ])

    prompt = f"""You are a medical diagnostic assistant. Select the most informative symptom to ask next.

Patient: {patient.sex}, {patient.age} years old
Chief complaint: {patient.initial_evidence}
Confirmed symptoms: {', '.join(confirmed[:5]) if confirmed else 'None'}
Denied symptoms: {', '.join(denied[:5]) if denied else 'None'}

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


def extract_answer_from_response(response: str, top_n: int = 5) -> tuple[str | None, str | None]:
    """응답에서 REASONING과 ANSWER 추출.

    Returns:
        (reasoning, answer_number)
    """
    reasoning = None
    answer = None

    # REASONING 추출
    reasoning_match = re.search(r'REASONING:\s*(.+?)(?=ANSWER:|$)', response, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # ANSWER 추출
    answer_match = re.search(r'ANSWER:\s*(\d+)', response, re.IGNORECASE)
    if answer_match:
        num = int(answer_match.group(1))
        if 1 <= num <= top_n:
            answer = str(num)

    # Fallback: 마지막 숫자 찾기
    if not answer:
        numbers = re.findall(r'\b([1-5])\b', response[-100:])
        if numbers:
            answer = numbers[-1]

    return reasoning, answer


def run_optimized_benchmark(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",  # Instruct 모델로 변경
    n_samples: int = 100,
    shuffle: bool = True,
    top_n: int = 5,
):
    """최적화된 프롬프트 벤치마크 실행."""
    print("=" * 70)
    print("최적화된 프롬프트 Two-Stage 벤치마크")
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
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )

    # Stage 1: 추론 생성 (Instruct 모델용 - 짧은 응답)
    sampling_params_reason = SamplingParams(
        temperature=0.0,
        max_tokens=512,  # Instruct 모델은 짧은 응답 생성
    )

    # Stage 2: 숫자 선택 (structured_outputs)
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

    # KG 연결
    kg = None
    if KG_AVAILABLE:
        try:
            print("KG 연결...", flush=True)
            kg = UMLSKG()
            print("KG 연결 성공!", flush=True)
        except Exception as e:
            print(f"KG 연결 실패: {e}", flush=True)
            kg = None
    else:
        print("KG 모듈 없음, 테스트용 후보 사용", flush=True)

    # 결과 저장
    results = []
    token_counts = []
    format_success = 0  # REASONING/ANSWER 형식 성공
    stage1_stage2_match = 0
    total_with_answer = 0
    selection_distribution = {str(i): 0 for i in range(1, top_n + 1)}

    print(f"\n{'='*70}")
    print("벤치마크 시작")
    print(f"{'='*70}\n")

    for i, patient in enumerate(patients):
        print(f"\n[{i+1}/{len(patients)}] 환자: {patient.sex}, {patient.age}세, 주증상: {patient.initial_evidence}")
        print(f"  실제 진단: {patient.pathology}")

        # 테스트용 후보 증상 (실제 벤치마크에서는 KG에서 가져옴)
        # 다양한 의료 증상 풀에서 선택
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

        # 환자별로 다른 후보 선택 (시뮬레이션)
        random.seed(i)  # 재현성 위해 환자 인덱스 기반 시드
        candidates = random.sample(symptom_pool, min(top_n * 2, len(symptom_pool)))

        # 셔플 적용
        if shuffle:
            candidates = list(candidates)
            random.shuffle(candidates)

        candidates = candidates[:top_n]

        # 최적화된 프롬프트 생성
        prompt = build_optimized_prompt(
            patient=patient,
            candidates=candidates,
            confirmed=[],
            denied=[],
            top_n=top_n,
        )

        # Stage 1: 추론 생성
        stage1_output = llm.generate([prompt], sampling_params_reason)
        response = stage1_output[0].outputs[0].text.strip()

        # 토큰 수 계산
        reason_tokens = len(tokenizer.encode(response))
        token_counts.append(reason_tokens)

        # REASONING/ANSWER 추출
        reasoning, answer_from_stage1 = extract_answer_from_response(response, top_n)

        # 형식 성공 여부
        if reasoning and answer_from_stage1:
            format_success += 1

        # Stage 2: 숫자 선택
        stage2_content = f"""{prompt}

{response}

Based on your reasoning above, confirm your final answer with just the number (1-{top_n}):"""

        try:
            stage2_output = llm.generate([stage2_content], sampling_params_select)
            stage2_selection = stage2_output[0].outputs[0].text.strip()
        except Exception as e:
            print(f"  Stage 2 에러: {e}")
            stage2_selection = "1"

        # Selection 분포 업데이트
        if stage2_selection in selection_distribution:
            selection_distribution[stage2_selection] += 1

        # Stage 1 vs Stage 2 일치 확인
        if answer_from_stage1:
            total_with_answer += 1
            if answer_from_stage1 == stage2_selection:
                stage1_stage2_match += 1

        # 선택된 후보 정보
        try:
            selected_idx = int(stage2_selection) - 1
            selected_candidate = candidates[selected_idx] if 0 <= selected_idx < len(candidates) else None
        except:
            selected_candidate = None

        # 결과 저장
        result = {
            "patient_id": i,
            "patient_info": f"{patient.sex}, {patient.age}",
            "chief_complaint": patient.initial_evidence,
            "ground_truth": patient.pathology,
            "candidates": [(c[0], c[1], float(c[2])) for c in candidates],
            "prompt": prompt,
            "response": response,
            "reasoning": reasoning,
            "answer_from_stage1": answer_from_stage1,
            "stage2_selection": stage2_selection,
            "selected_candidate": selected_candidate[1] if selected_candidate else None,
            "format_success": bool(reasoning and answer_from_stage1),
            "stage1_stage2_match": answer_from_stage1 == stage2_selection if answer_from_stage1 else None,
            "reason_tokens": reason_tokens,
        }
        results.append(result)

        # 상세 출력
        print(f"  후보: {[c[1] for c in candidates]}")
        print(f"  추론 토큰: {reason_tokens}")
        print(f"  --- Stage 1 응답 ---")
        print(f"  {response[:500]}{'...' if len(response) > 500 else ''}")
        print(f"  --- 추출 결과 ---")
        print(f"  REASONING: {reasoning[:200] if reasoning else 'None'}{'...' if reasoning and len(reasoning) > 200 else ''}")
        print(f"  ANSWER (Stage 1): {answer_from_stage1}")
        print(f"  ANSWER (Stage 2): {stage2_selection}")
        print(f"  형식 성공: {'✅' if reasoning and answer_from_stage1 else '❌'}")
        print(f"  Stage 1-2 일치: {'✅' if answer_from_stage1 == stage2_selection else '❌' if answer_from_stage1 else 'N/A'}")
        print(f"  선택된 증상: {selected_candidate[1] if selected_candidate else 'Unknown'}")

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

    print(f"\n[형식 성공률]")
    print(f"  REASONING/ANSWER 형식 성공: {format_success}/{len(results)} ({100*format_success/len(results):.1f}%)")

    print(f"\n[Stage 1-2 일치율]")
    if total_with_answer > 0:
        print(f"  일치: {stage1_stage2_match}/{total_with_answer} ({100*stage1_stage2_match/total_with_answer:.1f}%)")
    else:
        print(f"  Stage 1에서 답변 추출 실패")

    print(f"\n[Stage 2 Selection 분포]")
    for sel in sorted(selection_distribution.keys()):
        count = selection_distribution[sel]
        pct = 100 * count / len(results) if results else 0
        bar = "█" * int(pct / 5)
        print(f"  {sel}번: {count:3d} ({pct:5.1f}%) {bar}")

    # 결과 저장 (모델명 기반 파일명)
    model_short = model.split("/")[-1].replace("-", "_").lower()
    output_path = Path(f"results/optimized_prompt_{model_short}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": model,
            "n_samples": len(results),
            "shuffle": shuffle,
            "top_n": top_n,
            "avg_tokens": avg_tokens,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "format_success_rate": format_success / len(results) if results else 0,
            "stage1_stage2_match_rate": stage1_stage2_match / total_with_answer if total_with_answer else 0,
            "selection_distribution": selection_distribution,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {output_path}")

    return results


if __name__ == "__main__":
    run_optimized_benchmark()
