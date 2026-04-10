#!/usr/bin/env python3
"""전체 실패 케이스에서 min_il 변화 테스트.

1단계: 현재 설정(min_il=5)으로 실패 케이스 식별
2단계: 실패 케이스에 대해 다양한 min_il 값 테스트
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


def run_diagnosis_with_min_il(args: tuple) -> dict | None:
    """단일 환자 진단 수행."""
    patient_idx, patient_data, loader_data, min_il = args

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

        for _ in range(50):  # max_il = 50
            candidates = kg.get_candidate_symptoms(
                initial_cui=initial_cui,
                limit=10,
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

            should_stop, _ = kg.should_stop(
                max_il=50,
                min_il=min_il,
                confidence_threshold=0.30,
                gap_threshold=0.04,
                relative_gap_threshold=1.5,
            )
            if should_stop:
                break

        diagnosis_candidates = kg.get_diagnosis_candidates(top_k=10)
        predicted_cuis = [d.cui for d in diagnosis_candidates]

        correct_at_1 = gt_cui in predicted_cuis[:1] if gt_cui else False
        correct_at_10 = gt_cui in predicted_cuis[:10] if gt_cui else False

        kg.close()

        return {
            "patient_idx": patient_idx,
            "correct_at_1": correct_at_1,
            "correct_at_10": correct_at_10,
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
    print("Full Failure Cases min_il Test")
    print("=" * 70)

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

    patients_data = [
        {
            "age": p.age,
            "sex": p.sex,
            "initial_evidence": p.initial_evidence,
            "evidences": p.evidences,
            "pathology": p.pathology,
            "differential_diagnosis": p.differential_diagnosis,
        }
        for p in all_patients
    ]

    # 1단계: min_il=5로 실패 케이스 식별
    print("\n[Phase 1] Identifying failure cases with min_il=5...")

    tasks = [(i, patients_data[i], loader_data, 5) for i in range(len(patients_data))]

    failure_indices = []
    start = time.time()

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_diagnosis_with_min_il, t): t[0] for t in tasks}

        done = 0
        for future in as_completed(futures):
            r = future.result()
            done += 1

            if r and not r["correct_at_10"]:
                failure_indices.append(r["patient_idx"])

            if done % 10000 == 0:
                print(f"  Progress: {done}/{len(tasks)}, failures so far: {len(failure_indices)}")

    print(f"  Total failures: {len(failure_indices)}")
    print(f"  Time: {time.time() - start:.1f}s")

    # 2단계: 실패 케이스에 대해 다양한 min_il 테스트
    print("\n[Phase 2] Testing different min_il values on failure cases...")

    min_il_values = [5, 7, 10, 15, 20]
    results_summary = {}

    for min_il in min_il_values:
        tasks = [(idx, patients_data[idx], loader_data, min_il) for idx in failure_indices]

        results = []
        start = time.time()

        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(run_diagnosis_with_min_il, t): i for i, t in enumerate(tasks)}
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)

        elapsed = time.time() - start

        total = len(results)
        recovered = sum(1 for r in results if r["correct_at_10"])
        gtpa_1 = sum(1 for r in results if r["correct_at_1"]) / total if total else 0
        gtpa_10 = sum(1 for r in results if r["correct_at_10"]) / total if total else 0
        avg_il = sum(r["il"] for r in results) / total if total else 0
        avg_confirmed = sum(r["confirmed"] for r in results) / total if total else 0
        avg_denied = sum(r["denied"] for r in results) / total if total else 0

        results_summary[min_il] = {
            "total": total,
            "recovered": recovered,
            "recovery_rate": recovered / total if total else 0,
            "gtpa_1": gtpa_1,
            "gtpa_10": gtpa_10,
            "avg_il": avg_il,
            "avg_confirmed": avg_confirmed,
            "avg_denied": avg_denied,
        }

        print(f"\n[min_il = {min_il}]")
        print(f"  Recovered: {recovered}/{total} ({recovered/total*100:.1f}%)")
        print(f"  GTPA@1:  {gtpa_1:.2%}")
        print(f"  GTPA@10: {gtpa_10:.2%}")
        print(f"  Avg IL:  {avg_il:.1f}")
        print(f"  Avg confirmed: {avg_confirmed:.1f}")
        print(f"  Avg denied: {avg_denied:.1f}")
        print(f"  Time: {elapsed:.1f}s")

    # 결과 저장
    output = {
        "total_failures": len(failure_indices),
        "failure_indices": failure_indices,
        "results": results_summary,
    }

    output_path = Path("results/min_il_test_on_failures.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
