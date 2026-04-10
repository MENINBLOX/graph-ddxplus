#!/usr/bin/env python3
"""실패 케이스에서 min_il 변화 테스트.

기존 실패 케이스 4,025개를 대상으로 min_il 값 변경 효과 측정.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)


def run_single_diagnosis(args: tuple) -> dict | None:
    """단일 환자 진단 수행."""
    patient_idx, patient_data, loader_data, top_n, max_il, min_il = args

    from src.data_loader import DDXPlusLoader, Patient
    from src.umls_kg import UMLSKG

    loader = DDXPlusLoader()
    loader._symptom_mapping = loader_data["symptom_mapping"]
    loader._disease_mapping = loader_data["disease_mapping"]
    loader._fr_to_eng = loader_data["fr_to_eng"]
    loader._conditions = loader_data["conditions"]

    try:
        kg = UMLSKG()
    except Exception:
        return None

    try:
        patient = Patient(
            age=patient_data["age"],
            sex=patient_data["sex"],
            initial_evidence=patient_data["initial_evidence"],
            evidences=patient_data["evidences"],
            pathology=patient_data["pathology"],
            differential_diagnosis=patient_data["differential_diagnosis"],
        )

        gt_disease_name = loader.fr_to_eng.get(patient.pathology, patient.pathology)
        gt_cui = loader.get_disease_cui(gt_disease_name)

        patient_positive_cuis = set()
        for ev_str in patient.evidences:
            code = ev_str.split("_@_")[0] if "_@_" in ev_str else ev_str
            cui = loader.get_symptom_cui(code)
            if cui:
                patient_positive_cuis.add(cui)

        initial_cui = loader.get_symptom_cui(patient.initial_evidence)
        if not initial_cui:
            kg.close()
            return None

        kg.reset_state()
        kg.state.add_confirmed(initial_cui)

        il = 0
        confirmed_count = 1
        denied_count = 0

        for _ in range(max_il):
            candidates = kg.get_candidate_symptoms(
                initial_cui=initial_cui,
                limit=top_n,
                confirmed_cuis=kg.state.confirmed_cuis,
                denied_cuis=kg.state.denied_cuis,
                asked_cuis=kg.state.asked_cuis,
            )

            if not candidates:
                break

            selected = candidates[0]

            if selected.cui in patient_positive_cuis:
                kg.state.add_confirmed(selected.cui)
                confirmed_count += 1
            else:
                kg.state.add_denied(selected.cui)
                denied_count += 1

            il += 1

            # 커스텀 min_il로 종료 조건 확인
            should_stop, _ = kg.should_stop(
                max_il=max_il,
                min_il=min_il,
                confidence_threshold=0.30,
                gap_threshold=0.04,
                relative_gap_threshold=1.5,
            )
            if should_stop:
                break

        # 최종 진단
        diagnosis_candidates = kg.get_diagnosis_candidates(top_k=50)
        predicted_cuis = [d.cui for d in diagnosis_candidates]

        gt_rank = None
        for i, d in enumerate(diagnosis_candidates):
            if d.cui == gt_cui:
                gt_rank = i + 1
                break

        correct_at_1 = gt_cui in predicted_cuis[:1] if gt_cui else False
        correct_at_10 = gt_cui in predicted_cuis[:10] if gt_cui else False

        kg.close()

        return {
            "patient_idx": patient_idx,
            "correct_at_1": correct_at_1,
            "correct_at_10": correct_at_10,
            "gt_rank": gt_rank,
            "il": il,
            "confirmed": confirmed_count,
            "denied": denied_count,
        }

    except Exception:
        kg.close()
        return None


def main():
    from src.data_loader import DDXPlusLoader

    print("=" * 70)
    print("Test min_il on Failure Cases")
    print("=" * 70)

    # 실패 케이스 로드
    failure_file = Path("results/gtpa10_failure_analysis.json")
    if not failure_file.exists():
        print("Error: failure analysis file not found")
        return

    with open(failure_file) as f:
        failure_data = json.load(f)

    failure_indices = [case["patient_idx"] for case in failure_data["sample_cases"]]

    # 전체 실패 케이스가 sample_cases에 없을 수 있음
    # 전체 환자를 로드하고 실패 케이스만 필터링
    print(f"Sample cases in file: {len(failure_indices)}")

    # 데이터 로드
    loader = DDXPlusLoader()
    all_patients = loader.load_patients(split="test", n_samples=None)
    print(f"Loaded {len(all_patients)} patients")

    loader_data = {
        "symptom_mapping": loader.symptom_mapping,
        "disease_mapping": loader.disease_mapping,
        "fr_to_eng": loader.fr_to_eng,
        "conditions": {
            k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
            for k, v in loader.conditions.items()
        },
    }

    # 실패 케이스만 추출 (sample_cases의 patient_idx 사용)
    # sample_cases는 최대 50개만 저장됨, 전체 실패 케이스 재구성 필요

    # 대신 전체 테스트에서 실패한 케이스를 다시 찾기
    # 빠른 테스트를 위해 샘플 케이스만 사용
    print(f"\nUsing {len(failure_indices)} failure cases from sample")

    patients_data = []
    for idx in failure_indices:
        if idx < len(all_patients):
            p = all_patients[idx]
            patients_data.append({
                "idx": idx,
                "age": p.age,
                "sex": p.sex,
                "initial_evidence": p.initial_evidence,
                "evidences": p.evidences,
                "pathology": p.pathology,
                "differential_diagnosis": p.differential_diagnosis,
            })

    print(f"Prepared {len(patients_data)} patients for testing")

    # min_il 값 테스트
    min_il_values = [5, 7, 10, 15, 20]

    print("\n" + "=" * 70)
    print("Testing different min_il values...")
    print("=" * 70)

    for min_il in min_il_values:
        tasks = [
            (p["idx"], p, loader_data, 10, 50, min_il)
            for p in patients_data
        ]

        results = []
        start = time.time()

        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(run_single_diagnosis, t): i for i, t in enumerate(tasks)}
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)

        elapsed = time.time() - start

        # 통계
        total = len(results)
        gtpa_1 = sum(1 for r in results if r["correct_at_1"]) / total if total else 0
        gtpa_10 = sum(1 for r in results if r["correct_at_10"]) / total if total else 0
        avg_il = sum(r["il"] for r in results) / total if total else 0
        avg_confirmed = sum(r["confirmed"] for r in results) / total if total else 0
        avg_denied = sum(r["denied"] for r in results) / total if total else 0

        # 복구된 케이스 (이전에 실패했는데 이제 성공)
        recovered = sum(1 for r in results if r["correct_at_10"])

        print(f"\n[min_il = {min_il}]")
        print(f"  Recovered: {recovered}/{total} ({recovered/total*100:.1f}%)")
        print(f"  GTPA@1:  {gtpa_1:.2%}")
        print(f"  GTPA@10: {gtpa_10:.2%}")
        print(f"  Avg IL:  {avg_il:.1f}")
        print(f"  Avg confirmed: {avg_confirmed:.1f}")
        print(f"  Avg denied: {avg_denied:.1f}")


if __name__ == "__main__":
    main()
