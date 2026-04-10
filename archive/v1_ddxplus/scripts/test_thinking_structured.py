#!/usr/bin/env python3
"""Thinking 모델 + Structured Output 테스트.

세 가지 방법 비교:
1. V0 + enable_reasoning
2. JSON Schema (response_format)
3. 후처리 파싱 (기존 방식)
"""

import json
import os
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)


def test_method1_v0_reasoning():
    """방법 1: V0 + enable_reasoning 테스트."""
    print("\n" + "=" * 70)
    print("방법 1: VLLM_USE_V1=0 + enable_reasoning")
    print("=" * 70)

    # V0 강제
    os.environ["VLLM_USE_V1"] = "0"

    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model="Qwen/Qwen3-4B-Thinking-2507",
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
        )

        prompt = """Select the best symptom to ask (1-5):
1. Fever
2. Cough
3. Headache
4. Chest Pain
5. Fatigue

Output only the number:"""

        # Regex 제약
        from vllm.sampling_params import StructuredOutputsParams
        structured_params = StructuredOutputsParams(regex="[1-5]")
        params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            structured_outputs=structured_params,
        )

        output = llm.generate([prompt], params)
        response = output[0].outputs[0].text

        print(f"\n응답 (처음 500자):\n{response[:500]}")
        print(f"\n응답 (마지막 200자):\n...{response[-200:]}")

        # 결과 분석
        if '</think' in response.lower():
            print("\n✅ Thinking 태그 발견!")
            parts = re.split(r'</think(?:ing)?>', response, flags=re.IGNORECASE)
            if len(parts) > 1:
                answer = parts[-1].strip()
                print(f"최종 답변: {answer}")
        else:
            print("\n❌ Thinking 태그 없음")
            print(f"전체 응답: {response}")

        return True

    except Exception as e:
        print(f"\n❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # V1으로 복원
        os.environ.pop("VLLM_USE_V1", None)


def test_method2_json_schema():
    """방법 2: JSON Schema 테스트."""
    print("\n" + "=" * 70)
    print("방법 2: JSON Schema (response_format)")
    print("=" * 70)

    try:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        llm = LLM(
            model="Qwen/Qwen3-4B-Thinking-2507",
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
        )

        # JSON Schema 정의
        json_schema = {
            "type": "object",
            "properties": {
                "selection": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5
                },
                "reason": {
                    "type": "string"
                }
            },
            "required": ["selection"]
        }

        prompt = """Select the best symptom to ask for a patient with chest pain.

Candidates:
1. Fever
2. Cough
3. Headache
4. Dyspnea
5. Fatigue

Respond in JSON format with "selection" (1-5) and "reason"."""

        structured_params = StructuredOutputsParams(json_schema=json_schema)
        params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            structured_outputs=structured_params,
        )

        output = llm.generate([prompt], params)
        response = output[0].outputs[0].text

        print(f"\n응답:\n{response[:1000]}")

        # JSON 파싱 시도
        # </thinking> 이후 부분에서 JSON 추출
        if '</think' in response.lower():
            parts = re.split(r'</think(?:ing)?>', response, flags=re.IGNORECASE)
            json_part = parts[-1].strip() if len(parts) > 1 else response
        else:
            json_part = response

        try:
            # JSON 추출
            json_match = re.search(r'\{[^}]+\}', json_part)
            if json_match:
                result = json.loads(json_match.group())
                print(f"\n✅ JSON 파싱 성공: {result}")
            else:
                print(f"\n❌ JSON 형식 없음")
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON 파싱 실패: {e}")

        return True

    except Exception as e:
        print(f"\n❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_method3_post_processing():
    """방법 3: 후처리 파싱 테스트."""
    print("\n" + "=" * 70)
    print("방법 3: 후처리 파싱 (Regex 제약 없음)")
    print("=" * 70)

    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model="Qwen/Qwen3-4B-Thinking-2507",
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
        )

        prompt = """You are a medical diagnostic assistant. Select the best symptom to ask.

Patient: Male, 45 years old
Chief complaint: Chest pain

Candidates:
1. Fever (15%)
2. Cough (12%)
3. Headache (10%)
4. Dyspnea (35%)
5. Fatigue (28%)

Think step by step, then provide your final answer as a single number (1-5)."""

        params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
        )

        output = llm.generate([prompt], params)
        response = output[0].outputs[0].text

        print(f"\n응답 (처음 500자):\n{response[:500]}")
        print(f"\n응답 (마지막 300자):\n...{response[-300:]}")

        # 후처리 파싱
        answer = None

        # 1. </thinking> 이후 숫자 추출
        if '</think' in response.lower():
            parts = re.split(r'</think(?:ing)?>', response, flags=re.IGNORECASE)
            if len(parts) > 1:
                after_think = parts[-1].strip()
                numbers = re.findall(r'[1-5]', after_think)
                if numbers:
                    answer = int(numbers[0])
                    print(f"\n✅ </thinking> 이후 추출: {answer}")

        # 2. Fallback: 마지막 500자에서 "final answer" 패턴
        if answer is None:
            patterns = [
                r'final\s+answer[:\s]+(\d)',
                r'(?:select|choose|pick)\s+(?:option\s+)?(\d)',
                r'answer\s*(?:is|:)\s*(\d)',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, response[-500:], re.IGNORECASE)
                if matches:
                    answer = int(matches[-1])
                    print(f"\n✅ 패턴 매칭으로 추출: {answer}")
                    break

        # 3. Fallback: 마지막 숫자
        if answer is None:
            numbers = re.findall(r'[1-5]', response[-100:])
            if numbers:
                answer = int(numbers[-1])
                print(f"\n⚠️ 마지막 숫자 추출: {answer}")

        if answer:
            print(f"\n최종 답변: {answer}")
        else:
            print(f"\n❌ 답변 추출 실패")

        return True

    except Exception as e:
        print(f"\n❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_methods():
    """모든 방법 테스트."""
    print("=" * 70)
    print("Thinking 모델 + Structured Output 테스트")
    print("모델: Qwen/Qwen3-4B-Thinking-2507")
    print("=" * 70)

    results = {}

    # 방법 3 먼저 (가장 안정적)
    print("\n[테스트 1/3]")
    results["method3_post_processing"] = test_method3_post_processing()

    # GPU 메모리 정리
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 방법 2
    print("\n[테스트 2/3]")
    results["method2_json_schema"] = test_method2_json_schema()

    # GPU 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 방법 1
    print("\n[테스트 3/3]")
    results["method1_v0_reasoning"] = test_method1_v0_reasoning()

    # 결과 요약
    print("\n" + "=" * 70)
    print("테스트 결과 요약")
    print("=" * 70)
    for method, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  {method}: {status}")

    return results


if __name__ == "__main__":
    test_all_methods()
