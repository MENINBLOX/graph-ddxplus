#!/usr/bin/env python3
"""단일 max_il 값 실험 - 병렬 처리.

사용법: python benchmark_max_il_single.py --max-il 30
        python benchmark_max_il_single.py --unlimited
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
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
    patient_data, symptom_mapping, disease_mapping, fr_to_eng, min_il, max_il, stability_k, stability_n = args

    patient = Patient(
        age=patient_data['age'],
        sex=patient_data['sex'],
        pathology=patient_data['pathology'],
        initial_evidence=patient_data['initial_evidence'],
        evidences=patient_data['evidences'],
        differential_diagnosis=patient_data.get('differential_diagnosis', []),
    )

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
    hard_limit = max_il if max_il else 500
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

        # min_il 이후에만 stability 체크
        if il >= min_il:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-il', type=int, default=10, help='min_il 값 (기본값: 10)')
    parser.add_argument('--max-il', type=int, default=None, help='max_il 값')
    parser.add_argument('--unlimited', action='store_true', help='max_il 무제한')
    parser.add_argument('--workers', type=int, default=4, help='워커 수')
    args = parser.parse_args()

    min_il_value = args.min_il
    max_il_value = None if args.unlimited else args.max_il
    max_label = "unlimited" if max_il_value is None else max_il_value

    print(f"=== min_il={min_il_value}, max_il={max_label} 실험 ===")

    stability_k = 3
    stability_n = 5
    n_workers = args.workers

    loader = DDXPlusLoader()
    patients = loader.load_patients(split='test')

    print(f"샘플 수: {len(patients):,}, Workers: {n_workers}")

    patient_data_list = [
        {
            'age': p.age,
            'sex': p.sex,
            'pathology': p.pathology,
            'initial_evidence': p.initial_evidence,
            'evidences': p.evidences,
            'differential_diagnosis': p.differential_diagnosis,
        }
        for p in patients
    ]

    symptom_mapping = loader.symptom_mapping
    disease_mapping = loader.disease_mapping
    fr_to_eng = loader.fr_to_eng

    tasks = [
        (pd, symptom_mapping, disease_mapping, fr_to_eng, min_il_value, max_il_value, stability_k, stability_n)
        for pd in patient_data_list
    ]

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_patient, task): i for i, task in enumerate(tasks)}

        with tqdm(total=len(futures), desc=f"max_il={max_label}") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)

    elapsed = time.time() - start_time

    # 분석
    correct_at_1 = sum(1 for r in results if r.correct_at_1)
    correct_at_10 = sum(1 for r in results if r.correct_at_10)
    il_distribution = [r.il for r in results]
    max_il_reached_count = sum(1 for r in results if r.max_il_reached)

    count = len(results)
    gtpa_1 = correct_at_1 / count if count > 0 else 0
    gtpa_10 = correct_at_10 / count if count > 0 else 0
    avg_il = np.mean(il_distribution) if il_distribution else 0

    output = {
        "min_il": min_il_value,
        "max_il": max_label,
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
        "elapsed": elapsed,
    }

    print()
    print(f"완료: {elapsed:.1f}초")
    print(f"GTPA@1: {gtpa_1:.2%}")
    print(f"GTPA@10: {gtpa_10:.2%}")
    print(f"Avg IL: {avg_il:.2f}")
    print(f"max_il 도달: {max_il_reached_count:,}건 ({max_il_reached_count/count:.2%})")

    output_file = Path(__file__).parent.parent / "results" / f"max_il_{max_label}_min{min_il_value}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"저장: {output_file}")


if __name__ == "__main__":
    main()
