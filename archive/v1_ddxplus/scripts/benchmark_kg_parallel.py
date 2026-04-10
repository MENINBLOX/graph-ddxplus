#!/usr/bin/env python3
"""KG-Only 병렬 벤치마크.

8개 Neo4j 인스턴스를 활용한 ProcessPoolExecutor 병렬 처리.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# Neo4j 포트 목록 (8개 인스턴스)
NEO4J_PORTS = [7687, 7688, 7689, 7690, 7691, 7692, 7693, 7694]


def run_diagnosis(args: tuple) -> dict | None:
    """단일 환자 진단 수행."""
    patient_idx, patient_data, loader_data, neo4j_port = args

    from src.data_loader import DDXPlusLoader, Patient
    from src.umls_kg import UMLSKG

    # 데이터 로더 복원
    loader = DDXPlusLoader()
    loader._symptom_mapping = loader_data["symptom_mapping"]
    loader._disease_mapping = loader_data["disease_mapping"]
    loader._fr_to_eng = loader_data["fr_to_eng"]
    loader._conditions = loader_data["conditions"]

    try:
        # 지정된 포트의 Neo4j에 연결
        uri = f"bolt://localhost:{neo4j_port}"
        kg = UMLSKG(uri=uri)
    except Exception as e:
        return {"error": str(e), "patient_idx": patient_idx}

    try:
        patient = Patient(
            age=patient_data["age"],
            sex=patient_data["sex"],
            initial_evidence=patient_data["initial_evidence"],
            evidences=patient_data["evidences"],
            pathology=patient_data["pathology"],
            differential_diagnosis=patient_data["differential_diagnosis"],
        )

        # GT 질환 CUI
        gt_disease_name = loader.fr_to_eng.get(patient.pathology, patient.pathology)
        gt_cui = loader.get_disease_cui(gt_disease_name)

        # 환자 증상 CUI 집합
        patient_positive_cuis = set()
        for ev_str in patient.evidences:
            code = ev_str.split("_@_")[0] if "_@_" in ev_str else ev_str
            cui = loader.get_symptom_cui(code)
            if cui:
                patient_positive_cuis.add(cui)

        # 초기 증상
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

            # 종료 조건 확인 (min_il=10)
            should_stop, _ = kg.should_stop(
                max_il=50,
                min_il=10,
                confidence_threshold=0.30,
                gap_threshold=0.04,
                relative_gap_threshold=1.5,
            )
            if should_stop:
                break

        # 진단
        diagnosis_candidates = kg.get_diagnosis_candidates(top_k=10)
        predicted_cuis = [d.cui for d in diagnosis_candidates]

        correct_at_1 = gt_cui in predicted_cuis[:1] if gt_cui else False
        correct_at_3 = gt_cui in predicted_cuis[:3] if gt_cui else False
        correct_at_5 = gt_cui in predicted_cuis[:5] if gt_cui else False
        correct_at_10 = gt_cui in predicted_cuis[:10] if gt_cui else False

        kg.close()

        return {
            "patient_idx": patient_idx,
            "correct_at_1": correct_at_1,
            "correct_at_3": correct_at_3,
            "correct_at_5": correct_at_5,
            "correct_at_10": correct_at_10,
            "il": il,
            "confirmed": confirmed_count,
            "denied": denied_count,
        }

    except Exception as e:
        kg.close()
        return {"error": str(e), "patient_idx": patient_idx}


def main():
    parser = argparse.ArgumentParser(description="KG-Only Parallel Benchmark")
    parser.add_argument("-n", "--n-samples", type=int, default=None)
    parser.add_argument("--severity", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from src.data_loader import DDXPlusLoader

    print("=" * 70)
    print("KG-Only Parallel Benchmark")
    print(f"Workers: {args.workers}, Neo4j ports: {NEO4J_PORTS[:args.workers]}")
    print("=" * 70)

    # 데이터 로드
    loader = DDXPlusLoader()
    all_patients = loader.load_patients(
        split="test",
        n_samples=args.n_samples,
        severity=args.severity,
    )
    print(f"Loaded {len(all_patients):,} patients")

    # 직렬화 가능한 데이터 준비
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

    # 작업 분배: 라운드 로빈으로 Neo4j 포트 할당
    tasks = [
        (i, patients_data[i], loader_data, NEO4J_PORTS[i % args.workers])
        for i in range(len(patients_data))
    ]

    # 병렬 실행
    results = []
    errors = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_diagnosis, t): t[0] for t in tasks}

        with tqdm(total=len(tasks), desc="Diagnosing") as pbar:
            for future in as_completed(futures):
                r = future.result()
                if r:
                    if "error" in r:
                        errors += 1
                    else:
                        results.append(r)
                pbar.update(1)

    elapsed = time.time() - start_time

    # 결과 집계
    total = len(results)
    gtpa_1 = sum(1 for r in results if r["correct_at_1"]) / total if total else 0
    gtpa_3 = sum(1 for r in results if r["correct_at_3"]) / total if total else 0
    gtpa_5 = sum(1 for r in results if r["correct_at_5"]) / total if total else 0
    gtpa_10 = sum(1 for r in results if r["correct_at_10"]) / total if total else 0
    avg_il = sum(r["il"] for r in results) / total if total else 0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Samples: {total:,} (errors: {errors})")
    print(f"GTPA@1:  {gtpa_1:.2%}")
    print(f"GTPA@3:  {gtpa_3:.2%}")
    print(f"GTPA@5:  {gtpa_5:.2%}")
    print(f"GTPA@10: {gtpa_10:.2%}")
    print(f"Avg IL:  {avg_il:.1f}")
    print(f"Time:    {elapsed:.1f}s ({total/elapsed:.1f} samples/s)")
    print("=" * 70)

    # 결과 저장
    output = {
        "config": {
            "n_samples": args.n_samples,
            "severity": args.severity,
            "workers": args.workers,
            "min_il": 10,
        },
        "metrics": {
            "total": total,
            "errors": errors,
            "gtpa_1": gtpa_1,
            "gtpa_3": gtpa_3,
            "gtpa_5": gtpa_5,
            "gtpa_10": gtpa_10,
            "avg_il": avg_il,
        },
        "elapsed": elapsed,
    }

    output_path = Path("results/kg_only_parallel.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
