#!/usr/bin/env python3
"""Instruct 모델 Structured Output 벤치마크.

Tiny Instruct 모델들의 의학적 추론 능력 평가:
- JSON structured output으로 reason + answer 추출
- reason 내용을 모니터링하여 의학적 지식 수준 평가
"""

import json
import os
import random
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from src.data_loader import DDXPlusLoader
from src.umls_kg import UMLSKG


# Tiny Instruct 모델 목록 (~4B)
TINY_INSTRUCT_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "mistralai/Ministral-3-3B-Instruct-2512",
    # "meta-llama/Llama-3.1-8B-Instruct",  # 8B, MEDDxAgent 비교용
]


def create_prompt(patient, candidates: list, top_n: int = 5) -> str:
    """진단 프롬프트 생성."""
    candidate_list = "\n".join([
        f"{i+1}. {c.name} (disease coverage: {c.disease_coverage})"
        for i, c in enumerate(candidates[:top_n])
    ])

    prompt = f"""You are a medical diagnostic assistant. Select the best symptom to ask next.

Patient Information:
- Sex: {patient.sex}
- Age: {patient.age} years old
- Chief complaint: {patient.initial_evidence}

Candidate symptoms to ask:
{candidate_list}

Respond in JSON format with:
- "reason": Brief medical reasoning (1-2 sentences)
- "answer": Your selection (1-{top_n})

Example: {{"reason": "Chest pain with dyspnea suggests cardiac or pulmonary issues.", "answer": 2}}"""

    return prompt


def run_benchmark(
    model_name: str,
    n_samples: int = 50,
    top_n: int = 5,
    verbose: bool = True,
) -> dict:
    """단일 모델 벤치마크 실행."""
    print(f"\n{'='*70}")
    print(f"모델: {model_name}")
    print(f"샘플: {n_samples}, Top-N: {top_n}")
    print("=" * 70)

    # 모델 로드
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )

    # JSON Schema 정의
    json_schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "answer": {"type": "integer", "minimum": 1, "maximum": top_n}
        },
        "required": ["reason", "answer"]
    }

    params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        structured_outputs=StructuredOutputsParams(json=json_schema),
    )

    # 데이터 로드
    loader = DDXPlusLoader()
    kg = UMLSKG()
    patients = loader.load_patients(split="test", n_samples=n_samples, severity=2)

    results = []
    correct = 0
    position_dist = {i: 0 for i in range(1, top_n + 1)}

    print(f"\n테스트 진행 중... (환자 수: {len(patients)})\n")

    skipped_no_cui = 0
    skipped_few_candidates = 0
    skipped_no_gt = 0

    for i, patient in enumerate(patients):
        # 주호소 CUI 얻기
        initial_cui = loader.get_symptom_cui(patient.initial_evidence)
        if not initial_cui:
            skipped_no_cui += 1
            if verbose and i < 5:
                print(f"[{i}] SKIP: no CUI for initial_evidence={patient.initial_evidence}")
            continue

        # KG에서 후보 증상 가져오기
        candidates = kg.get_candidate_symptoms(
            initial_cui=initial_cui,
            limit=top_n,
            asked_cuis={initial_cui},
        )

        if len(candidates) < top_n:
            skipped_few_candidates += 1
            if verbose and i < 5:
                print(f"[{i}] SKIP: only {len(candidates)} candidates (need {top_n})")
            continue

        # Ground truth CUI 집합 (환자의 실제 양성 증상들)
        # evidences 형식: ["code", "code_@_value", ...]
        # 모든 evidence는 양성 증상
        gt_cuis = set()
        for ev_str in patient.evidences:
            # "_@_" 형식 처리 (다중 값 증상)
            if "_@_" in ev_str:
                code = ev_str.split("_@_")[0]
            else:
                code = ev_str

            cui = loader.get_symptom_cui(code)
            if cui:
                gt_cuis.add(cui)

        # GT 인덱스 찾기
        gt_idx = None
        for idx, c in enumerate(candidates):
            if c.cui in gt_cuis:
                gt_idx = idx + 1
                break

        if gt_idx is None:
            skipped_no_gt += 1
            if verbose and i < 5:
                print(f"[{i}] SKIP: no GT in candidates. gt_cuis={len(gt_cuis)}, candidates={[c.name for c in candidates]}")
            continue

        # 프롬프트 생성 및 추론
        prompt = create_prompt(patient, candidates, top_n)
        output = llm.generate([prompt], params)
        response = output[0].outputs[0].text.strip()

        # JSON 파싱
        try:
            result = json.loads(response)
            answer = result.get("answer")
            reason = result.get("reason", "")
        except json.JSONDecodeError:
            answer = None
            reason = f"[JSON 파싱 실패] {response[:100]}"

        # 정확도 계산
        is_correct = answer == gt_idx
        if is_correct:
            correct += 1

        if answer and 1 <= answer <= top_n:
            position_dist[answer] += 1

        # 결과 저장
        result_entry = {
            "patient_id": i,
            "gt_idx": gt_idx,
            "pred_idx": answer,
            "correct": is_correct,
            "reason": reason,
            "candidates": [(c.name, c.disease_coverage) for c in candidates],
        }
        results.append(result_entry)

        # 상세 출력 (처음 10개만)
        if verbose and i < 10:
            status = "✅" if is_correct else "❌"
            print(f"[{i+1:2d}] GT:{gt_idx} Pred:{answer} {status}")
            print(f"     Reason: {reason[:80]}...")
            print(f"     Candidates: {[c.name for c in candidates]}")
            print()

    # 통계 계산
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"결과 요약")
    print("=" * 70)
    print(f"스킵 통계:")
    print(f"  - No CUI: {skipped_no_cui}")
    print(f"  - Few candidates: {skipped_few_candidates}")
    print(f"  - No GT: {skipped_no_gt}")
    print(f"정확도 (GTPA@1): {correct}/{total} ({accuracy:.1%})")
    print(f"\n위치 분포:")
    for pos, count in position_dist.items():
        pct = 100 * count / total if total > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {pos}번: {count:3d} ({pct:5.1f}%) {bar}")

    # Position bias 검사
    if total > 0:
        pos1_ratio = position_dist[1] / total
        if pos1_ratio > 0.3:
            print(f"\n⚠️  Position bias 의심: 1번 선택 비율 {pos1_ratio:.1%}")
        else:
            print(f"\n✅ Position bias 없음: 1번 선택 비율 {pos1_ratio:.1%}")

    # Reason 품질 샘플 출력
    print(f"\n{'='*70}")
    print("Reason 샘플 (의학적 지식 평가용)")
    print("=" * 70)

    for j, r in enumerate(results[:5]):
        print(f"\n[케이스 {j+1}]")
        print(f"  후보: {r['candidates']}")
        print(f"  GT: {r['gt_idx']}번, Pred: {r['pred_idx']}번 {'✅' if r['correct'] else '❌'}")
        print(f"  Reason: {r['reason']}")

    # KG 연결 종료
    kg.close()

    return {
        "model": model_name,
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "position_dist": position_dist,
        "results": results,
    }


def run_all_models(n_samples: int = 50, top_n: int = 5):
    """모든 Tiny Instruct 모델 벤치마크."""
    print("=" * 70)
    print("Tiny Instruct 모델 Structured Output 벤치마크")
    print(f"모델 수: {len(TINY_INSTRUCT_MODELS)}")
    print(f"샘플: {n_samples}, Top-N: {top_n}")
    print("=" * 70)

    all_results = []

    for model_name in TINY_INSTRUCT_MODELS:
        try:
            result = run_benchmark(model_name, n_samples, top_n)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ {model_name} 실패: {e}")
            import traceback
            traceback.print_exc()

        # GPU 메모리 정리
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 최종 비교
    print("\n" + "=" * 70)
    print("모델 비교 (GTPA@1)")
    print("=" * 70)

    for r in sorted(all_results, key=lambda x: x["accuracy"], reverse=True):
        model_short = r["model"].split("/")[-1]
        print(f"  {model_short:40s} {r['accuracy']:.1%} ({r['correct']}/{r['total']})")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-samples", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--model", type=str, default=None, help="특정 모델만 테스트")
    args = parser.parse_args()

    if args.model:
        run_benchmark(args.model, args.n_samples, args.top_n)
    else:
        run_all_models(args.n_samples, args.top_n)
