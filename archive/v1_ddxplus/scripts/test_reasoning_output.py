#!/usr/bin/env python3
"""Reasoning 모델 출력 방식 테스트.

다양한 프롬프트 전략으로 증상 선택 추출 테스트:
1. 번호 선택 (후처리)
2. 증상 이름 직접 출력
3. JSON 형식
4. Final Answer 형식
5. Bracket 형식 [Answer]
"""

import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from vllm import LLM, SamplingParams

from src.data_loader import DDXPlusLoader
from src.umls_kg import UMLSKG


def similarity(a: str, b: str) -> float:
    """문자열 유사도 계산."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_match(text: str, candidates: list[str]) -> tuple[int, str, float]:
    """텍스트에서 가장 유사한 후보 찾기."""
    best_idx = -1
    best_name = ""
    best_score = 0.0

    for idx, name in enumerate(candidates):
        # 정확한 매칭
        if name.lower() in text.lower():
            return idx, name, 1.0

        # 유사도 기반 매칭
        score = similarity(name, text)
        if score > best_score:
            best_score = score
            best_idx = idx
            best_name = name

    return best_idx, best_name, best_score


def extract_answer_method1_number(response: str, top_n: int) -> int | None:
    """방법 1: 번호 추출 (</think> 이후)."""
    text = response
    if "</think" in response.lower():
        parts = re.split(r"</think(?:ing)?>", response, flags=re.IGNORECASE)
        text = parts[-1] if len(parts) > 1 else response

    # 숫자 찾기
    numbers = re.findall(rf"[1-{top_n}]", text)
    if numbers:
        return int(numbers[0])
    return None


def extract_answer_method2_name(response: str, candidates: list[str]) -> int | None:
    """방법 2: 증상 이름 직접 매칭."""
    text = response
    if "</think" in response.lower():
        parts = re.split(r"</think(?:ing)?>", response, flags=re.IGNORECASE)
        text = parts[-1] if len(parts) > 1 else response

    idx, name, score = find_best_match(text, candidates)
    if score >= 0.6:
        return idx + 1
    return None


def extract_answer_method3_json(response: str, candidates: list[str]) -> int | None:
    """방법 3: JSON 형식 파싱."""
    text = response
    if "</think" in response.lower():
        parts = re.split(r"</think(?:ing)?>", response, flags=re.IGNORECASE)
        text = parts[-1] if len(parts) > 1 else response

    # JSON 추출
    json_match = re.search(r"\{[^}]+\}", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            symptom = data.get("symptom", data.get("answer", ""))
            if isinstance(symptom, int):
                return symptom
            idx, _, score = find_best_match(str(symptom), candidates)
            if score >= 0.6:
                return idx + 1
        except json.JSONDecodeError:
            pass
    return None


def extract_answer_method4_final(response: str, candidates: list[str]) -> int | None:
    """방법 4: Final Answer 패턴."""
    text = response
    if "</think" in response.lower():
        parts = re.split(r"</think(?:ing)?>", response, flags=re.IGNORECASE)
        text = parts[-1] if len(parts) > 1 else response

    # Final Answer: ... 패턴
    match = re.search(r"(?:final\s+)?answer[:\s]+([^\n.]+)", text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # 숫자면 바로 반환
        if answer.isdigit():
            return int(answer)
        # 이름 매칭
        idx, _, score = find_best_match(answer, candidates)
        if score >= 0.6:
            return idx + 1
    return None


def extract_answer_method5_bracket(response: str, candidates: list[str]) -> int | None:
    """방법 5: [Answer] 형식."""
    text = response
    if "</think" in response.lower():
        parts = re.split(r"</think(?:ing)?>", response, flags=re.IGNORECASE)
        text = parts[-1] if len(parts) > 1 else response

    # [증상이름] 또는 **증상이름** 패턴
    patterns = [
        r"\[([^\]]+)\]",
        r"\*\*([^*]+)\*\*",
        r":\s*([A-Za-z\s]+)$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            answer = matches[-1].strip()
            idx, _, score = find_best_match(answer, candidates)
            if score >= 0.6:
                return idx + 1
    return None


# 프롬프트 템플릿들
PROMPTS = {
    "number": """You are a medical diagnostic assistant. Select the best symptom to ask next.

Patient: {sex}, {age} years old
Chief complaint: {initial}

Candidate symptoms:
{candidates}

Think step by step, then output only the number (1-{top_n}) of your selection.""",

    "name": """You are a medical diagnostic assistant. Select the best symptom to ask next.

Patient: {sex}, {age} years old
Chief complaint: {initial}

Candidate symptoms: {names}

Think step by step, then output ONLY the exact symptom name you select.""",

    "json": """You are a medical diagnostic assistant. Select the best symptom to ask next.

Patient: {sex}, {age} years old
Chief complaint: {initial}

Candidate symptoms: {names}

Think step by step, then output your answer as JSON:
{{"symptom": "exact symptom name", "reason": "brief reason"}}""",

    "final": """You are a medical diagnostic assistant. Select the best symptom to ask next.

Patient: {sex}, {age} years old
Chief complaint: {initial}

Candidate symptoms: {names}

Think step by step about which symptom would be most diagnostically useful.
End with: Final Answer: [symptom name]""",

    "bracket": """You are a medical diagnostic assistant.

Patient: {sex}, {age} years old
Chief complaint: {initial}

Which symptom should be asked next? Choose from: {names}

Think carefully, then write your final choice in brackets like [Fever] or [Cough].""",
}

EXTRACTORS = {
    "number": lambda r, c, n: extract_answer_method1_number(r, n),
    "name": lambda r, c, n: extract_answer_method2_name(r, c),
    "json": lambda r, c, n: extract_answer_method3_json(r, c),
    "final": lambda r, c, n: extract_answer_method4_final(r, c),
    "bracket": lambda r, c, n: extract_answer_method5_bracket(r, c),
}


def run_test(
    model_name: str = "Qwen/Qwen3.5-9B",
    n_samples: int = 10,
    top_n: int = 5,
):
    """다양한 프롬프트 방식 테스트."""
    print("=" * 70)
    print(f"Reasoning 모델 출력 방식 테스트")
    print(f"모델: {model_name}")
    print(f"샘플: {n_samples}, Top-N: {top_n}")
    print("=" * 70)

    # 모델 로드
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )

    params = SamplingParams(
        temperature=0.6,  # Reasoning 모델 권장
        top_p=0.95,
        max_tokens=2048,
    )

    # 데이터 로드
    loader = DDXPlusLoader()
    kg = UMLSKG()
    patients = loader.load_patients(split="test", n_samples=n_samples * 3, severity=2)

    # 유효한 케이스 수집
    cases = []
    for patient in patients:
        if len(cases) >= n_samples:
            break

        initial_cui = loader.get_symptom_cui(patient.initial_evidence)
        if not initial_cui:
            continue

        candidates = kg.get_candidate_symptoms(
            initial_cui=initial_cui,
            limit=top_n,
            asked_cuis={initial_cui},
        )
        if len(candidates) < top_n:
            continue

        # GT 찾기
        gt_cuis = set()
        for ev_str in patient.evidences:
            code = ev_str.split("_@_")[0] if "_@_" in ev_str else ev_str
            cui = loader.get_symptom_cui(code)
            if cui:
                gt_cuis.add(cui)

        gt_idx = None
        for idx, c in enumerate(candidates):
            if c.cui in gt_cuis:
                gt_idx = idx + 1
                break

        if gt_idx is None:
            continue

        cases.append({
            "patient": patient,
            "candidates": candidates,
            "gt_idx": gt_idx,
            "names": [c.name for c in candidates],
        })

    kg.close()

    print(f"\n유효 케이스: {len(cases)}\n")

    # 각 방법별 테스트
    results = {method: {"correct": 0, "total": 0, "answers": []} for method in PROMPTS}

    for method, template in PROMPTS.items():
        print(f"\n{'='*70}")
        print(f"방법: {method}")
        print("=" * 70)

        for i, case in enumerate(cases):
            patient = case["patient"]
            candidates = case["candidates"]
            names = case["names"]
            gt_idx = case["gt_idx"]

            # 프롬프트 생성
            if method == "number":
                candidate_list = "\n".join([
                    f"{j+1}. {c.name}"
                    for j, c in enumerate(candidates)
                ])
                prompt = template.format(
                    sex=patient.sex,
                    age=patient.age,
                    initial=patient.initial_evidence,
                    candidates=candidate_list,
                    top_n=top_n,
                )
            else:
                prompt = template.format(
                    sex=patient.sex,
                    age=patient.age,
                    initial=patient.initial_evidence,
                    names=", ".join(names),
                )

            # 추론
            output = llm.generate([prompt], params)
            response = output[0].outputs[0].text

            # 답변 추출
            extractor = EXTRACTORS[method]
            pred_idx = extractor(response, names, top_n)

            is_correct = pred_idx == gt_idx
            results[method]["total"] += 1
            if is_correct:
                results[method]["correct"] += 1
            results[method]["answers"].append(pred_idx)

            # 출력 (처음 3개만 상세히)
            status = "✅" if is_correct else "❌"
            print(f"[{i+1}] GT:{gt_idx} Pred:{pred_idx} {status}")

            if i < 3:
                # </think> 이후만 표시
                answer_part = response
                if "</think" in response.lower():
                    parts = re.split(r"</think(?:ing)?>", response, flags=re.IGNORECASE)
                    answer_part = parts[-1].strip() if len(parts) > 1 else response
                print(f"    Answer: {answer_part[:100]}...")
                print()

    # 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    for method in PROMPTS:
        r = results[method]
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        # Position distribution
        dist = {}
        for a in r["answers"]:
            if a:
                dist[a] = dist.get(a, 0) + 1
        print(f"\n[{method}]")
        print(f"  정확도: {r['correct']}/{r['total']} ({acc:.1%})")
        print(f"  분포: {dist}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("-n", "--n-samples", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    run_test(args.model, args.n_samples, args.top_n)
