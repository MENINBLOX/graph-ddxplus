#!/usr/bin/env python3
"""KG-only Full Metrics Benchmark.

DDXPlus 전체 성능 지표 측정:
- GTPA@1, GTPA@3, GTPA@5, GTPA@10
- IL (Interaction Length)
- DDR (Differential Diagnosis Recall)
- DDP (Differential Diagnosis Precision)
- DDF1 (Differential Diagnosis F1)

CPU 멀티프로세스 지원.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)


@dataclass
class DiagnosisResult:
    """단일 환자 진단 결과."""

    patient_id: int
    ground_truth_cui: str
    ground_truth_name: str
    predicted_cuis: list[str]  # Top-10 예측 CUI
    predicted_names: list[str]  # Top-10 예측 이름
    ground_truth_dd_cuis: list[str]  # GT 감별진단 CUI 목록
    interaction_length: int

    # Top-k 정확도
    correct_at_1: bool = False
    correct_at_3: bool = False
    correct_at_5: bool = False
    correct_at_10: bool = False


def run_single_diagnosis(args: tuple) -> DiagnosisResult | None:
    """단일 환자 진단 수행 (프로세스 워커)."""
    patient_idx, patient_data, loader_data, top_n, max_il = args

    # 각 프로세스에서 import (pickle 문제 방지)
    from src.data_loader import DDXPlusLoader, Patient
    from src.umls_kg import UMLSKG

    # 로더 설정
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
        # 환자 데이터 복원
        patient = Patient(
            age=patient_data["age"],
            sex=patient_data["sex"],
            initial_evidence=patient_data["initial_evidence"],
            evidences=patient_data["evidences"],
            pathology=patient_data["pathology"],
            differential_diagnosis=patient_data["differential_diagnosis"],
        )

        # Ground truth CUI
        gt_disease_name = loader.fr_to_eng.get(patient.pathology, patient.pathology)
        gt_cui = loader.get_disease_cui(gt_disease_name)

        # Ground truth differential diagnosis CUIs
        gt_dd_cuis = []
        for dd_name_fr, _ in patient.differential_diagnosis:
            dd_name_eng = loader.fr_to_eng.get(dd_name_fr, dd_name_fr)
            dd_cui = loader.get_disease_cui(dd_name_eng)
            if dd_cui:
                gt_dd_cuis.append(dd_cui)

        # 환자의 실제 양성 증상 CUI 집합
        patient_positive_cuis = set()
        for ev_str in patient.evidences:
            code = ev_str.split("_@_")[0] if "_@_" in ev_str else ev_str
            cui = loader.get_symptom_cui(code)
            if cui:
                patient_positive_cuis.add(cui)

        # 초기 증상 CUI
        initial_cui = loader.get_symptom_cui(patient.initial_evidence)
        if not initial_cui:
            kg.close()
            return None

        # 진단 시작
        kg.reset_state()
        kg.state.add_confirmed(initial_cui)

        il = 0
        for step in range(max_il):
            # 후보 증상 가져오기
            candidates = kg.get_candidate_symptoms(
                initial_cui=initial_cui,
                limit=top_n,
                confirmed_cuis=kg.state.confirmed_cuis,
                denied_cuis=kg.state.denied_cuis,
                asked_cuis=kg.state.asked_cuis,
            )

            if not candidates:
                break

            # 첫 번째 후보 선택
            selected = candidates[0]

            # 환자 응답 시뮬레이션
            if selected.cui in patient_positive_cuis:
                kg.state.add_confirmed(selected.cui)
            else:
                kg.state.add_denied(selected.cui)

            il += 1

            # 종료 조건 확인
            should_stop, _ = kg.should_stop(max_il=max_il)
            if should_stop:
                break

        # 최종 진단 (Top-10)
        diagnosis_candidates = kg.get_diagnosis_candidates(top_k=10)
        predicted_cuis = [d.cui for d in diagnosis_candidates]
        predicted_names = [d.name for d in diagnosis_candidates]

        # Top-k 정확도 계산
        correct_at_1 = gt_cui in predicted_cuis[:1] if gt_cui else False
        correct_at_3 = gt_cui in predicted_cuis[:3] if gt_cui else False
        correct_at_5 = gt_cui in predicted_cuis[:5] if gt_cui else False
        correct_at_10 = gt_cui in predicted_cuis[:10] if gt_cui else False

        kg.close()

        return DiagnosisResult(
            patient_id=patient_idx,
            ground_truth_cui=gt_cui or "",
            ground_truth_name=gt_disease_name,
            predicted_cuis=predicted_cuis,
            predicted_names=predicted_names,
            ground_truth_dd_cuis=gt_dd_cuis,
            interaction_length=il,
            correct_at_1=correct_at_1,
            correct_at_3=correct_at_3,
            correct_at_5=correct_at_5,
            correct_at_10=correct_at_10,
        )

    except Exception as e:
        kg.close()
        print(f"Error processing patient {patient_idx}: {e}")
        return None


def compute_dd_metrics(results: list[DiagnosisResult]) -> dict:
    """DDR, DDP, DDF1 계산.

    DDR = |예측DD ∩ 정답DD| / |정답DD|
    DDP = |예측DD ∩ 정답DD| / |예측DD|
    DDF1 = 2 * DDR * DDP / (DDR + DDP)
    """
    total_recall_num = 0
    total_recall_den = 0
    total_precision_num = 0
    total_precision_den = 0

    for r in results:
        gt_set = set(r.ground_truth_dd_cuis)
        pred_set = set(r.predicted_cuis)

        intersection = len(gt_set & pred_set)

        total_recall_num += intersection
        total_recall_den += len(gt_set) if gt_set else 1
        total_precision_num += intersection
        total_precision_den += len(pred_set) if pred_set else 1

    ddr = total_recall_num / total_recall_den if total_recall_den > 0 else 0.0
    ddp = total_precision_num / total_precision_den if total_precision_den > 0 else 0.0

    if ddr + ddp > 0:
        ddf1 = 2 * ddr * ddp / (ddr + ddp)
    else:
        ddf1 = 0.0

    return {"ddr": ddr, "ddp": ddp, "ddf1": ddf1}


def run_benchmark(
    n_samples: int | None = None,
    severity: int | None = None,
    top_n: int = 10,
    max_il: int = 50,
    n_workers: int = 8,
) -> dict:
    """KG-only 벤치마크 실행."""
    from src.data_loader import DDXPlusLoader

    print("=" * 70)
    print("KG-only Full Metrics Benchmark")
    print("=" * 70)
    samples_str = str(n_samples) if n_samples else "ALL"
    severity_str = str(severity) if severity else "ALL"
    print(f"Samples: {samples_str}, Severity: {severity_str}")
    print(f"Top-N: {top_n}, Max IL: {max_il}, Workers: {n_workers}")
    print("=" * 70)

    # 데이터 로드
    loader = DDXPlusLoader()
    patients = loader.load_patients(
        split="test",
        n_samples=n_samples,
        severity=severity,
    )

    print(f"Loaded {len(patients)} patients")

    # 로더 데이터 직렬화 (프로세스 간 공유)
    loader_data = {
        "symptom_mapping": loader.symptom_mapping,
        "disease_mapping": loader.disease_mapping,
        "fr_to_eng": loader.fr_to_eng,
        "conditions": {k: asdict(v) if hasattr(v, '__dataclass_fields__') else v
                       for k, v in loader.conditions.items()},
    }

    # 환자 데이터 직렬화
    patient_data_list = [
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

    # 작업 생성
    tasks = [
        (i, patient_data_list[i], loader_data, top_n, max_il)
        for i in range(len(patients))
    ]

    # 멀티프로세스 실행
    results: list[DiagnosisResult] = []
    start_time = time.time()

    if n_workers == 1:
        for i, task in enumerate(tasks):
            result = run_single_diagnosis(task)
            if result:
                results.append(result)
            if (i + 1) % 1000 == 0:
                print(f"Progress: {i + 1}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(run_single_diagnosis, task): i
                      for i, task in enumerate(tasks)}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

                if len(results) % 1000 == 0:
                    print(f"Progress: {len(results)}/{len(tasks)}")

    elapsed = time.time() - start_time

    # 메트릭 계산
    total = len(results)

    # GTPA@k
    gtpa_1 = sum(1 for r in results if r.correct_at_1) / total if total > 0 else 0
    gtpa_3 = sum(1 for r in results if r.correct_at_3) / total if total > 0 else 0
    gtpa_5 = sum(1 for r in results if r.correct_at_5) / total if total > 0 else 0
    gtpa_10 = sum(1 for r in results if r.correct_at_10) / total if total > 0 else 0

    # Average IL
    avg_il = sum(r.interaction_length for r in results) / total if total > 0 else 0

    # DDR, DDP, DDF1
    dd_metrics = compute_dd_metrics(results)

    # 결과 출력
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Total Patients: {total}")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.3f}s/case)")
    print()
    print("[Primary Diagnosis Accuracy]")
    print(f"  GTPA@1:  {gtpa_1:.2%} ({sum(1 for r in results if r.correct_at_1)}/{total})")
    print(f"  GTPA@3:  {gtpa_3:.2%}")
    print(f"  GTPA@5:  {gtpa_5:.2%}")
    print(f"  GTPA@10: {gtpa_10:.2%}")
    print()
    print("[Interaction Length]")
    print(f"  Avg IL: {avg_il:.2f}")
    print()
    print("[Differential Diagnosis Metrics]")
    print(f"  DDR:  {dd_metrics['ddr']:.2%}")
    print(f"  DDP:  {dd_metrics['ddp']:.2%}")
    print(f"  DDF1: {dd_metrics['ddf1']:.2%}")
    print("=" * 70)

    # 반환 데이터
    return {
        "config": {
            "n_samples": n_samples,
            "severity": severity,
            "top_n": top_n,
            "max_il": max_il,
        },
        "metrics": {
            "total": total,
            "gtpa_1": gtpa_1,
            "gtpa_3": gtpa_3,
            "gtpa_5": gtpa_5,
            "gtpa_10": gtpa_10,
            "avg_il": avg_il,
            "ddr": dd_metrics["ddr"],
            "ddp": dd_metrics["ddp"],
            "ddf1": dd_metrics["ddf1"],
        },
        "elapsed": elapsed,
        "results": [asdict(r) for r in results],
    }


def main():
    parser = argparse.ArgumentParser(description="KG-only Full Metrics Benchmark")
    parser.add_argument("-n", "--n-samples", type=int, default=None,
                        help="Number of samples (default: all)")
    parser.add_argument("--severity", type=int, default=None,
                        help="Severity filter (1-5, default: all)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Top-N candidates for symptom selection")
    parser.add_argument("--max-il", type=int, default=50,
                        help="Maximum interaction length")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of worker processes")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    results = run_benchmark(
        n_samples=args.n_samples,
        severity=args.severity,
        top_n=args.top_n,
        max_il=args.max_il,
        n_workers=args.workers,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 결과에서 individual results 제외 (파일 크기 절약)
        save_data = {
            "config": results["config"],
            "metrics": results["metrics"],
            "elapsed": results["elapsed"],
        }

        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
