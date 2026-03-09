#!/usr/bin/env python3
"""Thinking 모델 + Choice 방식 테스트.

regex 대신 choice를 사용하면 position bias가 없을 수 있음.
"""

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


def run_choice_test(n_samples: int = 20):
    """Choice 방식 테스트."""
    print("=" * 70)
    print("Thinking 모델 + Choice 방식 테스트")
    print("=" * 70)

    llm = LLM(
        model="Qwen/Qwen3-4B-Thinking-2507",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )

    # 방법 1: Regex
    params_regex = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        structured_outputs=StructuredOutputsParams(regex="[1-5]"),
    )

    # 방법 2: Choice
    params_choice = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        structured_outputs=StructuredOutputsParams(choice=["1", "2", "3", "4", "5"]),
    )

    # 방법 3: 후처리 (Regex 없음)
    params_free = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
    )

    # 데이터 로드
    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test", n_samples=n_samples, severity=2)

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
    ]

    results_regex = []
    results_choice = []
    results_free = []

    print(f"\n{n_samples}개 케이스 테스트 중...\n")

    for i, patient in enumerate(patients):
        random.seed(i)
        candidates = random.sample(symptom_pool, 5)
        random.shuffle(candidates)

        candidate_list = "\n".join([
            f"{j+1}. {name} ({score:.0%})"
            for j, (_, name, score) in enumerate(candidates)
        ])

        prompt = f"""Select the best symptom to ask (1-5):

Patient: {patient.sex}, {patient.age} years old
Chief complaint: {patient.initial_evidence}

Candidates:
{candidate_list}

Output only the number:"""

        # 테스트 1: Regex
        out1 = llm.generate([prompt], params_regex)
        ans_regex = out1[0].outputs[0].text.strip()
        results_regex.append(ans_regex)

        # 테스트 2: Choice
        out2 = llm.generate([prompt], params_choice)
        ans_choice = out2[0].outputs[0].text.strip()
        results_choice.append(ans_choice)

        # 테스트 3: Free (후처리)
        out3 = llm.generate([prompt], params_free)
        response = out3[0].outputs[0].text.strip()

        # </thinking> 이후 숫자 추출
        ans_free = None
        if '</think' in response.lower():
            parts = re.split(r'</think(?:ing)?>', response, flags=re.IGNORECASE)
            if len(parts) > 1:
                numbers = re.findall(r'[1-5]', parts[-1])
                if numbers:
                    ans_free = numbers[0]
        if not ans_free:
            numbers = re.findall(r'[1-5]', response[-100:])
            ans_free = numbers[-1] if numbers else "?"
        results_free.append(ans_free)

        # 출력
        match = "✅" if ans_regex == ans_choice == ans_free else "❌"
        print(f"[{i+1:2d}] Regex:{ans_regex} | Choice:{ans_choice} | Free:{ans_free} {match}")

    # 분포 계산
    print("\n" + "=" * 70)
    print("결과 분포")
    print("=" * 70)

    for name, results in [("Regex", results_regex), ("Choice", results_choice), ("Free", results_free)]:
        dist = {str(i): 0 for i in range(1, 6)}
        for r in results:
            if r in dist:
                dist[r] += 1

        print(f"\n[{name}]")
        for k in sorted(dist.keys()):
            pct = 100 * dist[k] / len(results)
            bar = "█" * int(pct / 5)
            print(f"  {k}번: {dist[k]:2d} ({pct:5.1f}%) {bar}")

    # 일치율
    match_regex_choice = sum(1 for r, c in zip(results_regex, results_choice) if r == c)
    match_regex_free = sum(1 for r, f in zip(results_regex, results_free) if r == f)
    match_choice_free = sum(1 for c, f in zip(results_choice, results_free) if c == f)

    print(f"\n[일치율]")
    print(f"  Regex vs Choice: {match_regex_choice}/{n_samples} ({100*match_regex_choice/n_samples:.1f}%)")
    print(f"  Regex vs Free:   {match_regex_free}/{n_samples} ({100*match_regex_free/n_samples:.1f}%)")
    print(f"  Choice vs Free:  {match_choice_free}/{n_samples} ({100*match_choice_free/n_samples:.1f}%)")


if __name__ == "__main__":
    run_choice_test()
