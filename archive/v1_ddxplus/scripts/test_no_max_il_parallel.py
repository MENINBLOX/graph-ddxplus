#!/usr/bin/env python3
"""max_il 제거 실험 - 병렬 버전."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import numpy as np
from tqdm import tqdm

from src.data_loader import DDXPlusLoader, Patient
from src.umls_kg import UMLSKG


@dataclass
class TestResult:
    correct_at_1: bool
    correct_at_10: bool
    il: int


def get_top_k_ranks(candidates, k):
    return tuple(c.cui for c in candidates[:k])


def process_patient(args) -> TestResult | None:
    """단일 환자 처리."""
    patient_data, symptom_mapping, disease_mapping, fr_to_eng, stability_k, stability_n, hard_limit = args

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

    return TestResult(correct_at_1=correct_at_1, correct_at_10=correct_at_10, il=il)


def main():
    print("=== max_il 제거 실험 (병렬) ===")

    stability_k = 3
    stability_n = 5
    hard_limit = 500
    n_workers = 8

    loader = DDXPlusLoader()
    patients = loader.load_patients(split='test')

    print(f"샘플 수: {len(patients):,}")
    print(f"Workers: {n_workers}")
    print(f"설정: Top{stability_k}_stable_{stability_n}, hard_limit={hard_limit}")
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

    # 작업 생성
    tasks = [
        (pd, symptom_mapping, disease_mapping, fr_to_eng, stability_k, stability_n, hard_limit)
        for pd in patient_data_list
    ]

    # 병렬 실행
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_patient, task): i for i, task in enumerate(tasks)}

        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)

    # 결과 분석
    correct_at_1 = sum(1 for r in results if r.correct_at_1)
    correct_at_10 = sum(1 for r in results if r.correct_at_10)
    il_distribution = [r.il for r in results]

    count = len(results)
    gtpa_1 = correct_at_1 / count if count > 0 else 0
    gtpa_10 = correct_at_10 / count if count > 0 else 0
    avg_il = np.mean(il_distribution) if il_distribution else 0

    print()
    print("=== 최종 결과 ===")
    print(f"Count: {count:,}")
    print(f"GTPA@1: {gtpa_1:.2%}")
    print(f"GTPA@10: {gtpa_10:.2%}")
    print(f"Avg IL: {avg_il:.2f}")
    print()
    print("=== IL 통계 ===")
    print(f"Min: {min(il_distribution)}")
    print(f"Max: {max(il_distribution)}")
    print(f"Mean: {np.mean(il_distribution):.2f}")
    print(f"Std: {np.std(il_distribution):.2f}")
    print()

    for p in [50, 75, 90, 95, 99, 100]:
        print(f"{p}%: {np.percentile(il_distribution, p):.0f}")

    print()
    for threshold in [30, 40, 50, 100, 200]:
        cases = sum(1 for il in il_distribution if il >= threshold)
        pct = cases / len(il_distribution) * 100
        print(f"IL >= {threshold}: {cases:,}건 ({pct:.3f}%)")

    # 결과 저장
    output = {
        "method": "Top3_stable_5_no_max_il",
        "count": count,
        "gtpa_1": gtpa_1,
        "gtpa_10": gtpa_10,
        "avg_il": avg_il,
        "il_min": min(il_distribution),
        "il_max": max(il_distribution),
        "il_std": float(np.std(il_distribution)),
        "percentiles": {
            "50": float(np.percentile(il_distribution, 50)),
            "75": float(np.percentile(il_distribution, 75)),
            "90": float(np.percentile(il_distribution, 90)),
            "95": float(np.percentile(il_distribution, 95)),
            "99": float(np.percentile(il_distribution, 99)),
        },
    }

    output_file = Path(__file__).parent.parent / "results" / "no_max_il_experiment.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n결과 저장: {output_file}")


if __name__ == "__main__":
    main()
