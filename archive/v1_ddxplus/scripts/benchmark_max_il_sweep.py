#!/usr/bin/env python3
"""max_il 값에 따른 성능 비교 실험 - 병렬 버전.

max_il: 10, 15, 20, 25, 30, 35, 40, 45, 50, None(무제한)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import numpy as np
from tqdm import tqdm
import time

from src.data_loader import DDXPlusLoader, Patient
from src.umls_kg import UMLSKG


@dataclass
class TestResult:
    correct_at_1: bool
    correct_at_10: bool
    il: int
    max_il_reached: bool


def get_top_k_ranks(candidates, k):
    return tuple(c.cui for c in candidates[:k])


def process_patient(args) -> TestResult | None:
    """단일 환자 처리."""
    patient_data, symptom_mapping, disease_mapping, fr_to_eng, max_il, stability_k, stability_n = args

    # Patient 재구성
    patient = Patient(
        age=patient_data['age'],
        sex=patient_data['sex'],
        pathology=patient_data['pathology'],
        initial_evidence=patient_data['initial_evidence'],
        evidences=patient_data['evidences'],
        differential_diagnosis=patient_data.get('differential_diagnosis', []),
    )

    # CUI 조회 헬퍼
    def get_symptom_cui(code):
        info = symptom_mapping.get(code, {})
        return info.get('cui')

    def get_disease_cui(name_eng):
        info = disease_mapping.get(name_eng, {})
        return info.get('umls_cui')

    kg = UMLSKG()

    gt_disease_eng = fr_to_eng.get(patient.pathology, patient.pathology)
    gt_cui = get_disease_cui(gt_disease_eng)

    patient_positive_cuis = set()
    for ev_str in patient.evidences:
        code = ev_str.split('_@_')[0] if '_@_' in ev_str else ev_str
        cui = get_symptom_cui(code)
        if cui:
            patient_positive_cuis.add(cui)

    initial_cui = get_symptom_cui(patient.initial_evidence)
    if not initial_cui:
        kg.close()
        return None

    kg.reset_state()
    kg.state.add_confirmed(initial_cui)

    il = 1
    rank_history = deque(maxlen=stability_n)
    hard_limit = max_il if max_il else 500  # 무제한인 경우 500을 hard limit으로

    max_il_reached = False

    for _ in range(hard_limit):
        candidates = kg.get_candidate_symptoms(
            initial_cui=initial_cui,
            limit=10,
            confirmed_cuis=kg.state.confirmed_cuis,
            denied_cuis=kg.state.denied_cuis,
        )
        if not candidates:
            break

        next_cui = candidates[0].cui
        if next_cui in patient_positive_cuis:
            kg.state.add_confirmed(next_cui)
        else:
            kg.state.add_denied(next_cui)

        il += 1

        # max_il 체크 (설정된 경우)
        if max_il and il >= max_il:
            max_il_reached = True
            break

        # Stability 체크
        diagnosis_candidates = kg.get_diagnosis_candidates(top_k=stability_k)
        current_ranks = get_top_k_ranks(diagnosis_candidates, stability_k)
        rank_history.append(current_ranks)

        if len(rank_history) == stability_n:
            if all(r == rank_history[0] for r in rank_history):
                break

    final_candidates = kg.get_diagnosis_candidates(top_k=10)
    correct_at_1 = False
    correct_at_10 = False

    if final_candidates:
        if final_candidates[0].cui == gt_cui:
            correct_at_1 = True
        for c in final_candidates[:10]:
            if c.cui == gt_cui:
                correct_at_10 = True
                break

    kg.close()

    return TestResult(
        correct_at_1=correct_at_1,
        correct_at_10=correct_at_10,
        il=il,
        max_il_reached=max_il_reached
    )


def run_experiment(max_il_value, patient_data_list, symptom_mapping, disease_mapping, fr_to_eng, n_workers=8):
    """특정 max_il 값으로 실험 실행."""
    stability_k = 3
    stability_n = 5

    tasks = [
        (pd, symptom_mapping, disease_mapping, fr_to_eng, max_il_value, stability_k, stability_n)
        for pd in patient_data_list
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_patient, task): i for i, task in enumerate(tasks)}

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return results


def analyze_results(results, max_il_value):
    """결과 분석."""
    correct_at_1 = sum(1 for r in results if r.correct_at_1)
    correct_at_10 = sum(1 for r in results if r.correct_at_10)
    il_distribution = [r.il for r in results]
    max_il_reached_count = sum(1 for r in results if r.max_il_reached)

    count = len(results)
    gtpa_1 = correct_at_1 / count if count > 0 else 0
    gtpa_10 = correct_at_10 / count if count > 0 else 0
    avg_il = np.mean(il_distribution) if il_distribution else 0

    return {
        "max_il": max_il_value if max_il_value else "unlimited",
        "count": count,
        "gtpa_1": gtpa_1,
        "gtpa_10": gtpa_10,
        "avg_il": float(avg_il),
        "il_std": float(np.std(il_distribution)) if il_distribution else 0,
        "il_min": int(min(il_distribution)) if il_distribution else 0,
        "il_max": int(max(il_distribution)) if il_distribution else 0,
        "il_median": float(np.median(il_distribution)) if il_distribution else 0,
        "max_il_reached_count": max_il_reached_count,
        "max_il_reached_pct": max_il_reached_count / count if count > 0 else 0,
    }


def main():
    print("=== max_il Sweep 실험 ===")
    print()

    # max_il 값들: 10부터 50까지 5단위 + 무제한(None)
    max_il_values = [10, 15, 20, 25, 30, 35, 40, 45, 50, None]
    n_workers = 8

    loader = DDXPlusLoader()
    patients = loader.load_patients(split='test')

    print(f"샘플 수: {len(patients):,}")
    print(f"Workers: {n_workers}")
    print(f"테스트할 max_il 값: {[v if v else 'unlimited' for v in max_il_values]}")
    print()

    # 직렬화 가능한 형태로 변환
    patient_data_list = []
    for p in patients:
        patient_data_list.append({
            'age': p.age,
            'sex': p.sex,
            'pathology': p.pathology,
            'initial_evidence': p.initial_evidence,
            'evidences': p.evidences,
            'differential_diagnosis': p.differential_diagnosis,
        })

    # 공유 데이터
    symptom_mapping = loader.symptom_mapping
    disease_mapping = loader.disease_mapping
    fr_to_eng = loader.fr_to_eng

    all_results = []

    for max_il_value in max_il_values:
        label = max_il_value if max_il_value else "unlimited"
        print(f"\n{'='*50}")
        print(f"max_il = {label} 실험 시작...")
        start_time = time.time()

        results = run_experiment(
            max_il_value,
            patient_data_list,
            symptom_mapping,
            disease_mapping,
            fr_to_eng,
            n_workers
        )

        elapsed = time.time() - start_time
        analysis = analyze_results(results, max_il_value)
        analysis["elapsed"] = elapsed

        print(f"  완료: {elapsed:.1f}초")
        print(f"  GTPA@1: {analysis['gtpa_1']:.2%}")
        print(f"  GTPA@10: {analysis['gtpa_10']:.2%}")
        print(f"  Avg IL: {analysis['avg_il']:.2f}")
        print(f"  max_il 도달: {analysis['max_il_reached_count']:,}건 ({analysis['max_il_reached_pct']:.2%})")

        all_results.append(analysis)

    # 결과 요약 출력
    print("\n" + "="*80)
    print("=== 최종 결과 요약 ===")
    print("="*80)
    print(f"{'max_il':>10} | {'GTPA@1':>8} | {'GTPA@10':>8} | {'Avg IL':>8} | {'max_il 도달':>12}")
    print("-"*60)

    for r in all_results:
        max_il_label = str(r['max_il']) if r['max_il'] != "unlimited" else "∞"
        print(f"{max_il_label:>10} | {r['gtpa_1']:>7.2%} | {r['gtpa_10']:>7.2%} | {r['avg_il']:>8.2f} | {r['max_il_reached_pct']:>11.2%}")

    # 최적 max_il 찾기 (GTPA@1 기준)
    best_result = max(all_results, key=lambda x: x['gtpa_1'])
    print()
    print(f"최적 max_il (GTPA@1 기준): {best_result['max_il']}")
    print(f"  - GTPA@1: {best_result['gtpa_1']:.2%}")
    print(f"  - Avg IL: {best_result['avg_il']:.2f}")

    # 결과 저장
    output = {
        "experiment": "max_il_sweep",
        "stability_criteria": "Top3_stable_5",
        "total_samples": len(patients),
        "max_il_values_tested": [v if v else "unlimited" for v in max_il_values],
        "results": all_results,
        "best_max_il": best_result['max_il'],
        "best_gtpa_1": best_result['gtpa_1'],
    }

    output_file = Path(__file__).parent.parent / "results" / "max_il_sweep.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_file}")


if __name__ == "__main__":
    main()
