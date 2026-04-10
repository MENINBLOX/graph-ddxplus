#!/usr/bin/env python3
"""Scoring 전략 비교 테스트.

v15_ratio vs v23_mild_denied 비교.
"""

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
    patient_idx, patient_data, loader_data, top_n, max_il, scoring = args

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
            else:
                kg.state.add_denied(selected.cui)

            il += 1

            # 종료 조건 확인 (최적화된 파라미터 사용)
            should_stop, _ = kg.should_stop(max_il=max_il)
            if should_stop:
                break

        # 최종 진단 - scoring 전략 사용
        diagnosis_candidates = kg.get_diagnosis_candidates(top_k=10, scoring=scoring)
        predicted_cuis = [d.cui for d in diagnosis_candidates]

        correct_at_1 = gt_cui in predicted_cuis[:1] if gt_cui else False
        correct_at_10 = gt_cui in predicted_cuis[:10] if gt_cui else False

        kg.close()

        return {
            "correct_at_1": correct_at_1,
            "correct_at_10": correct_at_10,
            "il": il,
        }

    except Exception:
        kg.close()
        return None


def run_test(
    patients_data: list[dict],
    loader_data: dict,
    scoring: str,
    n_workers: int = 8,
) -> dict:
    """단일 scoring 전략 테스트."""
    tasks = [
        (i, patients_data[i], loader_data, 10, 50, scoring)
        for i in range(len(patients_data))
    ]

    results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_diagnosis, task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    elapsed = time.time() - start_time
    total = len(results)

    gtpa_1 = sum(1 for r in results if r["correct_at_1"]) / total if total > 0 else 0
    gtpa_10 = sum(1 for r in results if r["correct_at_10"]) / total if total > 0 else 0
    avg_il = sum(r["il"] for r in results) / total if total > 0 else 0

    return {
        "scoring": scoring,
        "gtpa_1": gtpa_1,
        "gtpa_10": gtpa_10,
        "avg_il": avg_il,
        "total": total,
        "elapsed": elapsed,
    }


def main():
    from src.data_loader import DDXPlusLoader

    n_samples = 1000

    print("=" * 70)
    print("Scoring Strategy Comparison")
    print("=" * 70)

    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test", n_samples=n_samples)
    print(f"Loaded {len(patients)} patients")

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
        for p in patients
    ]

    strategies = ["v23_mild_denied", "v18_coverage", "v15_ratio"]

    print("\nTesting scoring strategies...\n")

    for strategy in strategies:
        result = run_test(patients_data, loader_data, strategy)
        print(f"[{strategy}]")
        print(f"  GTPA@1:  {result['gtpa_1']:.2%}")
        print(f"  GTPA@10: {result['gtpa_10']:.2%}")
        print(f"  Avg IL:  {result['avg_il']:.2f}")
        print(f"  Time:    {result['elapsed']:.1f}s")
        print()


if __name__ == "__main__":
    main()
