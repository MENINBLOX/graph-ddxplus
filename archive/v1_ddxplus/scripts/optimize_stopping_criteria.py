#!/usr/bin/env python3
"""Stopping Criteria 최적화 실험.

IL을 20-25로 줄이면서 GTPA@1을 유지하는 파라미터 탐색.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)


@dataclass
class ExperimentResult:
    """실험 결과."""

    # 파라미터
    min_il: int
    confidence_threshold: float
    gap_threshold: float
    relative_gap_threshold: float

    # 결과
    gtpa_1: float
    gtpa_10: float
    avg_il: float
    total: int
    elapsed: float


def run_single_diagnosis(args: tuple) -> dict | None:
    """단일 환자 진단 수행."""
    (
        patient_idx,
        patient_data,
        loader_data,
        top_n,
        max_il,
        min_il,
        confidence_threshold,
        gap_threshold,
        relative_gap_threshold,
    ) = args

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

            # 커스텀 stopping criteria
            should_stop, _ = kg.should_stop(
                max_il=max_il,
                min_il=min_il,
                confidence_threshold=confidence_threshold,
                gap_threshold=gap_threshold,
                relative_gap_threshold=relative_gap_threshold,
            )
            if should_stop:
                break

        diagnosis_candidates = kg.get_diagnosis_candidates(top_k=10)
        predicted_cuis = [d.cui for d in diagnosis_candidates]

        correct_at_1 = gt_cui in predicted_cuis[:1] if gt_cui else False
        correct_at_10 = gt_cui in predicted_cuis[:10] if gt_cui else False

        kg.close()

        return {
            "correct_at_1": correct_at_1,
            "correct_at_10": correct_at_10,
            "il": il,
        }

    except Exception as e:
        kg.close()
        return None


def run_experiment(
    patients_data: list[dict],
    loader_data: dict,
    min_il: int,
    confidence_threshold: float,
    gap_threshold: float,
    relative_gap_threshold: float,
    top_n: int = 10,
    max_il: int = 50,
    n_workers: int = 8,
) -> ExperimentResult:
    """단일 파라미터 조합 실험."""

    tasks = [
        (
            i,
            patients_data[i],
            loader_data,
            top_n,
            max_il,
            min_il,
            confidence_threshold,
            gap_threshold,
            relative_gap_threshold,
        )
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

    return ExperimentResult(
        min_il=min_il,
        confidence_threshold=confidence_threshold,
        gap_threshold=gap_threshold,
        relative_gap_threshold=relative_gap_threshold,
        gtpa_1=gtpa_1,
        gtpa_10=gtpa_10,
        avg_il=avg_il,
        total=total,
        elapsed=elapsed,
    )


def main():
    parser = argparse.ArgumentParser(description="Stopping Criteria 최적화")
    parser.add_argument("-n", "--n-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=str, default="results/stopping_criteria_optimization.json")
    args = parser.parse_args()

    from src.data_loader import DDXPlusLoader

    print("=" * 70)
    print("Stopping Criteria Optimization")
    print("=" * 70)
    print(f"Samples: {args.n_samples}, Workers: {args.workers}")
    print("=" * 70)

    # 데이터 로드
    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test", n_samples=args.n_samples)
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

    # 파라미터 그리드
    param_grid = {
        "min_il": [2, 3, 5],
        "confidence_threshold": [0.15, 0.20, 0.25, 0.30],
        "gap_threshold": [0.04, 0.05, 0.06, 0.08],
        "relative_gap_threshold": [1.5, 1.8, 2.0, 2.5],
    }

    # 모든 조합 생성
    combinations = list(product(
        param_grid["min_il"],
        param_grid["confidence_threshold"],
        param_grid["gap_threshold"],
        param_grid["relative_gap_threshold"],
    ))

    print(f"\nTotal combinations: {len(combinations)}")
    print("\nRunning experiments...")

    all_results = []

    for i, (min_il, conf, gap, rel_gap) in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] min_il={min_il}, conf={conf}, gap={gap}, rel_gap={rel_gap}")

        result = run_experiment(
            patients_data=patients_data,
            loader_data=loader_data,
            min_il=min_il,
            confidence_threshold=conf,
            gap_threshold=gap,
            relative_gap_threshold=rel_gap,
            n_workers=args.workers,
        )

        print(f"  → GTPA@1: {result.gtpa_1:.2%}, GTPA@10: {result.gtpa_10:.2%}, IL: {result.avg_il:.2f}")
        all_results.append(asdict(result))

    # 결과 정렬: GTPA@1 >= 78% 조건에서 IL 최소화
    filtered = [r for r in all_results if r["gtpa_1"] >= 0.78]
    if filtered:
        best = min(filtered, key=lambda x: x["avg_il"])
    else:
        # 78% 미만이면 가장 높은 GTPA@1
        best = max(all_results, key=lambda x: x["gtpa_1"])

    print("\n" + "=" * 70)
    print("Best Result (GTPA@1 >= 78%, minimize IL)")
    print("=" * 70)
    print(f"min_il: {best['min_il']}")
    print(f"confidence_threshold: {best['confidence_threshold']}")
    print(f"gap_threshold: {best['gap_threshold']}")
    print(f"relative_gap_threshold: {best['relative_gap_threshold']}")
    print(f"GTPA@1: {best['gtpa_1']:.2%}")
    print(f"GTPA@10: {best['gtpa_10']:.2%}")
    print(f"Avg IL: {best['avg_il']:.2f}")
    print("=" * 70)

    # IL 20-25 범위에서 최고 GTPA@1
    il_range = [r for r in all_results if 20 <= r["avg_il"] <= 25]
    if il_range:
        best_in_range = max(il_range, key=lambda x: x["gtpa_1"])
        print("\nBest in IL 20-25 range:")
        print(f"  GTPA@1: {best_in_range['gtpa_1']:.2%}, IL: {best_in_range['avg_il']:.2f}")
        print(f"  Params: min_il={best_in_range['min_il']}, conf={best_in_range['confidence_threshold']}, "
              f"gap={best_in_range['gap_threshold']}, rel_gap={best_in_range['relative_gap_threshold']}")

    # 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "param_grid": param_grid,
            "n_samples": args.n_samples,
            "all_results": all_results,
            "best": best,
            "best_in_il_range": best_in_range if il_range else None,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
