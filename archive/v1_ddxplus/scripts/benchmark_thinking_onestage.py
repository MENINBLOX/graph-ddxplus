#!/usr/bin/env python3
"""Thinking 모델 One-Stage 벤치마크.

Thinking 모델은 <think> 블록에서 추론 후 답변을 생성하므로
Two-Stage가 필요 없음. One-Stage로 직접 결과를 받고 정확도 측정.
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

from src.data_loader import DDXPlusLoader, Patient

try:
    from src.umls_kg import UMLSKG
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False


def build_onestage_prompt(
    patient: Patient,
    candidates: list[tuple[str, str, float]],
    confirmed: list[str],
    denied: list[str],
    top_n: int = 5,
) -> str:
    """One-Stage 프롬프트 생성."""
    total_score = sum(score for _, _, score in candidates) or 1

    candidate_list = "\n".join([
        f"{i+1}. {name} ({score/total_score:.0%})"
        for i, (cui, name, score) in enumerate(candidates)
    ])

    prompt = f"""You are a medical diagnostic assistant. Select the most informative symptom to ask next.

Patient: {patient.sex}, {patient.age} years old
Chief complaint: {patient.initial_evidence}
Confirmed symptoms: {', '.join(confirmed[:5]) if confirmed else 'None'}
Denied symptoms: {', '.join(denied[:5]) if denied else 'None'}

Candidate symptoms to inquire:
{candidate_list}

Think step by step about which symptom would be most valuable for differential diagnosis, then provide your final answer as a single number (1-{top_n})."""

    return prompt


def extract_answer_from_thinking(response: str, top_n: int = 5) -> tuple[int | None, int, int]:
    """Thinking 모델 응답에서 답변 추출.

    Returns:
        (answer_number, thinking_tokens, answer_tokens)
    """
    # </think> 또는 </thinking> 태그로 분리
    thinking_part = ""
    answer_part = response

    for tag in ['</thinking>', '</think>']:
        if tag in response:
            parts = response.split(tag)
            thinking_part = parts[0]
            answer_part = parts[-1].strip() if len(parts) > 1 else ""
            break

    # 토큰 수 계산 (대략적으로 단어 수 * 1.3)
    thinking_tokens = len(thinking_part.split())
    answer_tokens = len(answer_part.split())

    # 답변에서 숫자 추출
    answer = None

    # 먼저 answer_part에서 숫자 찾기 (태그 이후의 최종 답변)
    if answer_part:
        # 마지막 숫자가 최종 답변일 가능성 높음
        numbers = re.findall(r'\b([1-9])\b', answer_part)
        for num in numbers:
            if 1 <= int(num) <= top_n:
                answer = int(num)
                break

    # answer_part에서 못 찾으면 thinking_part 마지막에서 찾기
    if answer is None and thinking_part:
        # 마지막 500자에서 결론 찾기
        last_part = thinking_part[-500:]
        patterns = [
            r'final\s+answer[:\s]+(\d)',
            r'(?:select|choose|pick|answer|option)\s*(?:is\s*)?[:\s]*(\d)',
            r'(?:number|#)\s*(\d)',
            r'\b(\d)\s*(?:is\s+the\s+(?:best|most))',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, last_part, re.IGNORECASE)
            if matches:
                num = int(matches[-1])
                if 1 <= num <= top_n:
                    answer = num
                    break

        # 패턴 매칭 실패시 마지막 숫자
        if answer is None:
            numbers = re.findall(r'\b([1-9])\b', last_part[-100:])
            if numbers:
                num = int(numbers[-1])
                if 1 <= num <= top_n:
                    answer = num

    return answer, thinking_tokens, answer_tokens


def run_thinking_onestage_benchmark(
    model: str = "Qwen/Qwen3-4B-Thinking-2507",
    n_samples: int = 100,
    shuffle: bool = True,
    top_n: int = 5,
):
    """Thinking 모델 One-Stage 벤치마크 실행."""
    print("=" * 70)
    print("Thinking 모델 One-Stage 벤치마크")
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

    # One-Stage: 충분한 토큰으로 thinking + 답변 생성
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,  # thinking + 답변
    )

    # 데이터 로드
    print("데이터 로딩...", flush=True)
    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test", n_samples=n_samples, severity=2)
    print(f"로드된 환자: {len(patients)}")

    # KG 연결 (실제 후보 증상용)
    kg = None
    if KG_AVAILABLE:
        try:
            print("KG 연결...", flush=True)
            kg = UMLSKG()
            print("KG 연결 성공!", flush=True)
        except Exception as e:
            print(f"KG 연결 실패: {e}", flush=True)

    # Ground Truth 증상 매핑 (테스트용)
    # 실제로는 환자의 evidences에서 정답을 가져와야 함

    # 결과 저장
    results = []
    thinking_token_counts = []
    answer_token_counts = []
    total_token_counts = []
    correct_count = 0
    selection_distribution = {str(i): 0 for i in range(1, top_n + 1)}
    extraction_failures = 0

    print(f"\n{'='*70}")
    print("벤치마크 시작")
    print(f"{'='*70}\n")

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

    for i, patient in enumerate(patients):
        print(f"\n[{i+1}/{len(patients)}] 환자: {patient.sex}, {patient.age}세, 주증상: {patient.initial_evidence}")
        print(f"  실제 진단: {patient.pathology}")

        # 환자별 후보 선택
        random.seed(i)
        candidates = random.sample(symptom_pool, min(top_n * 2, len(symptom_pool)))

        if shuffle:
            candidates = list(candidates)
            random.shuffle(candidates)

        candidates = candidates[:top_n]

        # 정답 설정 (환자의 실제 증상 중 하나를 후보에 삽입)
        # 실제 벤치마크에서는 KG에서 관련 증상을 가져옴
        # 여기서는 랜덤하게 정답 위치 설정 (테스트용)
        ground_truth_idx = random.randint(0, top_n - 1)
        ground_truth_symptom = candidates[ground_truth_idx][1]

        # 프롬프트 생성
        prompt = build_onestage_prompt(
            patient=patient,
            candidates=candidates,
            confirmed=[],
            denied=[],
            top_n=top_n,
        )

        # One-Stage 생성
        output = llm.generate([prompt], sampling_params)
        response = output[0].outputs[0].text.strip()

        # 전체 토큰 수 계산
        total_tokens = len(tokenizer.encode(response))
        total_token_counts.append(total_tokens)

        # 답변 추출
        answer, thinking_tokens, answer_tokens = extract_answer_from_thinking(response, top_n)

        # 실제 토큰 수로 업데이트 (tokenizer 사용)
        thinking_tag = None
        for tag in ['</thinking>', '</think>']:
            if tag in response:
                thinking_tag = tag
                break

        if thinking_tag:
            parts = response.split(thinking_tag)
            thinking_part = parts[0]
            answer_part = parts[-1].strip() if len(parts) > 1 else ""
            thinking_tokens = len(tokenizer.encode(thinking_part))
            answer_tokens = len(tokenizer.encode(answer_part))
        else:
            thinking_tokens = 0
            answer_tokens = total_tokens

        thinking_token_counts.append(thinking_tokens)
        answer_token_counts.append(answer_tokens)

        # 정확도 계산
        is_correct = answer == (ground_truth_idx + 1) if answer else False
        if is_correct:
            correct_count += 1

        # 선택 분포
        if answer:
            selection_distribution[str(answer)] += 1
        else:
            extraction_failures += 1

        # 선택된 후보 정보
        selected_candidate = candidates[answer - 1] if answer and 1 <= answer <= len(candidates) else None

        # 결과 저장
        result = {
            "patient_id": i,
            "patient_info": f"{patient.sex}, {patient.age}",
            "chief_complaint": patient.initial_evidence,
            "ground_truth_pathology": patient.pathology,
            "candidates": [(c[0], c[1], float(c[2])) for c in candidates],
            "ground_truth_idx": ground_truth_idx + 1,
            "ground_truth_symptom": ground_truth_symptom,
            "response": response,
            "extracted_answer": answer,
            "is_correct": is_correct,
            "thinking_tokens": thinking_tokens,
            "answer_tokens": answer_tokens,
            "total_tokens": total_tokens,
            "selected_symptom": selected_candidate[1] if selected_candidate else None,
        }
        results.append(result)

        # 상세 출력
        print(f"  후보: {[c[1] for c in candidates]}")
        print(f"  정답: {ground_truth_idx + 1}번 ({ground_truth_symptom})")
        print(f"  --- 응답 (처음 300자) ---")
        print(f"  {response[:300]}...")
        print(f"  --- 응답 (마지막 200자) ---")
        print(f"  ...{response[-200:]}")
        print(f"  --- 추출 결과 ---")
        print(f"  Thinking 토큰: {thinking_tokens}")
        print(f"  Answer 토큰: {answer_tokens}")
        print(f"  전체 토큰: {total_tokens}")
        print(f"  추출된 답변: {answer}")
        print(f"  정답 여부: {'✅ CORRECT' if is_correct else '❌ WRONG'}")
        print(f"  선택된 증상: {selected_candidate[1] if selected_candidate else 'None'}")

    # 통계 계산
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    # 토큰 통계
    avg_thinking = sum(thinking_token_counts) / len(thinking_token_counts) if thinking_token_counts else 0
    std_thinking = (sum((x - avg_thinking) ** 2 for x in thinking_token_counts) / len(thinking_token_counts)) ** 0.5 if thinking_token_counts else 0
    min_thinking = min(thinking_token_counts) if thinking_token_counts else 0
    max_thinking = max(thinking_token_counts) if thinking_token_counts else 0

    avg_answer = sum(answer_token_counts) / len(answer_token_counts) if answer_token_counts else 0
    std_answer = (sum((x - avg_answer) ** 2 for x in answer_token_counts) / len(answer_token_counts)) ** 0.5 if answer_token_counts else 0

    avg_total = sum(total_token_counts) / len(total_token_counts) if total_token_counts else 0
    std_total = (sum((x - avg_total) ** 2 for x in total_token_counts) / len(total_token_counts)) ** 0.5 if total_token_counts else 0
    min_total = min(total_token_counts) if total_token_counts else 0
    max_total = max(total_token_counts) if total_token_counts else 0

    # GTPA@1 (Ground Truth Prediction Accuracy)
    gtpa_at_1 = correct_count / len(results) if results else 0

    print(f"\n[토큰 통계]")
    print(f"  Thinking 토큰:")
    print(f"    평균: {avg_thinking:.1f} ± {std_thinking:.1f}")
    print(f"    범위: {min_thinking} ~ {max_thinking}")
    print(f"  Answer 토큰:")
    print(f"    평균: {avg_answer:.1f} ± {std_answer:.1f}")
    print(f"  전체 토큰:")
    print(f"    평균: {avg_total:.1f} ± {std_total:.1f}")
    print(f"    범위: {min_total} ~ {max_total}")

    print(f"\n[정확도]")
    print(f"  GTPA@1: {correct_count}/{len(results)} ({100*gtpa_at_1:.1f}%)")
    print(f"  답변 추출 실패: {extraction_failures}/{len(results)} ({100*extraction_failures/len(results):.1f}%)")

    print(f"\n[Selection 분포]")
    for sel in sorted(selection_distribution.keys()):
        count = selection_distribution[sel]
        pct = 100 * count / len(results) if results else 0
        bar = "█" * int(pct / 5)
        print(f"  {sel}번: {count:3d} ({pct:5.1f}%) {bar}")

    # 랜덤 기준선
    random_baseline = 100 / top_n
    print(f"\n[기준선 비교]")
    print(f"  랜덤 기준선: {random_baseline:.1f}%")
    print(f"  GTPA@1: {100*gtpa_at_1:.1f}%")
    print(f"  개선: {100*gtpa_at_1 - random_baseline:+.1f}%p")

    # 결과 저장
    model_short = model.split("/")[-1].replace("-", "_").lower()
    output_path = Path(f"results/thinking_onestage_{model_short}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": model,
            "n_samples": len(results),
            "shuffle": shuffle,
            "top_n": top_n,
            "thinking_tokens": {
                "avg": avg_thinking,
                "std": std_thinking,
                "min": min_thinking,
                "max": max_thinking,
            },
            "answer_tokens": {
                "avg": avg_answer,
                "std": std_answer,
            },
            "total_tokens": {
                "avg": avg_total,
                "std": std_total,
                "min": min_total,
                "max": max_total,
            },
            "gtpa_at_1": gtpa_at_1,
            "extraction_failure_rate": extraction_failures / len(results) if results else 0,
            "selection_distribution": selection_distribution,
            "random_baseline": random_baseline / 100,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {output_path}")

    return results


if __name__ == "__main__":
    run_thinking_onestage_benchmark()
