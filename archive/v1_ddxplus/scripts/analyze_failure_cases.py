#!/usr/bin/env python3
"""GTPA@10 Failure Case Analysis (Multiprocess).

GTPA@10에서 실패한 극단적 케이스들을 분석하여
실패 원인을 파악하고 논문에 기술할 내용을 생성.
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)


@dataclass
class FailureCase:
    """실패 케이스 정보."""

    patient_idx: int
    ground_truth: str
    ground_truth_cui: str | None
    top_10_predictions: list[dict]
    confirmed_count: int
    denied_count: int
    patient_evidence_count: int
    failure_reason: str
    gt_rank: int | None
    severity: int


def analyze_single_patient(args: tuple) -> FailureCase | None:
    """단일 환자 분석 (프로세스 워커)."""
    patient_idx, patient_data, loader_data, top_n, max_il = args

    from src.data_loader import DDXPlusLoader, Patient
    from src.umls_kg import UMLSKG

    # 로더 복원
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

        # Ground truth
        gt_disease_name = loader.fr_to_eng.get(patient.pathology, patient.pathology)
        gt_cui = loader.get_disease_cui(gt_disease_name)

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
            return FailureCase(
                patient_idx=patient_idx,
                ground_truth=gt_disease_name,
                ground_truth_cui=gt_cui,
                top_10_predictions=[],
                confirmed_count=0,
                denied_count=0,
                patient_evidence_count=len(patient.evidences),
                failure_reason="INITIAL_SYMPTOM_NOT_MAPPED",
                gt_rank=None,
                severity=patient_data.get("severity", 0),
            )

        # 진단 시작
        kg.reset_state()
        kg.state.add_confirmed(initial_cui)

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

            should_stop, _ = kg.should_stop(max_il=max_il)
            if should_stop:
                break

        # 최종 진단 (top-50까지 확인)
        diagnosis_candidates = kg.get_diagnosis_candidates(top_k=50)

        # Top-10 예측
        top_10 = [
            {"rank": i + 1, "cui": c.cui, "name": c.name, "score": round(c.score, 4)}
            for i, c in enumerate(diagnosis_candidates[:10])
        ]

        # GT 순위 확인
        gt_rank = None
        for i, c in enumerate(diagnosis_candidates):
            if c.cui == gt_cui:
                gt_rank = i + 1
                break

        kg.close()

        # 성공 케이스 (top-10 내에 GT 존재)
        if gt_rank is not None and gt_rank <= 10:
            return None  # 성공

        # 실패 원인 분류
        if gt_cui is None:
            failure_reason = "GT_DISEASE_NOT_IN_KG"
        elif gt_rank is not None:
            failure_reason = f"GT_RANK_{gt_rank}"
        else:
            failure_reason = "GT_NOT_IN_TOP50"

        return FailureCase(
            patient_idx=patient_idx,
            ground_truth=gt_disease_name,
            ground_truth_cui=gt_cui,
            top_10_predictions=top_10,
            confirmed_count=confirmed_count,
            denied_count=denied_count,
            patient_evidence_count=len(patient.evidences),
            failure_reason=failure_reason,
            gt_rank=gt_rank,
            severity=patient_data.get("severity", 0),
        )

    except Exception as e:
        kg.close()
        return FailureCase(
            patient_idx=patient_idx,
            ground_truth="ERROR",
            ground_truth_cui=None,
            top_10_predictions=[],
            confirmed_count=0,
            denied_count=0,
            patient_evidence_count=0,
            failure_reason=f"ERROR: {e!s}",
            gt_rank=None,
            severity=0,
        )


def run_failure_analysis(
    n_samples: int | None = None,
    n_workers: int = 8,
    top_n: int = 10,
    max_il: int = 50,
) -> dict:
    """실패 케이스 분석 실행."""
    from src.data_loader import DDXPlusLoader

    print("=" * 70)
    print("GTPA@10 Failure Case Analysis (Multiprocess)")
    print("=" * 70)
    print(f"Workers: {n_workers}, Top-N: {top_n}, Max IL: {max_il}")

    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test", n_samples=n_samples)
    print(f"Loaded {len(patients)} patients")

    # 로더 데이터 직렬화
    loader_data = {
        "symptom_mapping": loader.symptom_mapping,
        "disease_mapping": loader.disease_mapping,
        "fr_to_eng": loader.fr_to_eng,
        "conditions": {
            k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
            for k, v in loader.conditions.items()
        },
    }

    # 환자 데이터 직렬화 (severity 포함)
    patient_data_list = []
    for p in patients:
        pdata = {
            "age": p.age,
            "sex": p.sex,
            "initial_evidence": p.initial_evidence,
            "evidences": p.evidences,
            "pathology": p.pathology,
            "differential_diagnosis": p.differential_diagnosis,
        }
        # Severity 추출
        disease_name = loader.fr_to_eng.get(p.pathology, p.pathology)
        condition = loader.conditions.get(disease_name)
        if condition and hasattr(condition, "severity"):
            pdata["severity"] = condition.severity
        else:
            pdata["severity"] = 0
        patient_data_list.append(pdata)

    # 작업 생성
    tasks = [
        (i, patient_data_list[i], loader_data, top_n, max_il)
        for i in range(len(patients))
    ]

    # 멀티프로세스 실행
    failure_cases: list[FailureCase] = []
    success_count = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(analyze_single_patient, task): i for i, task in enumerate(tasks)}

        processed = 0
        for future in as_completed(futures):
            result = future.result()
            processed += 1

            if result is None:
                success_count += 1
            else:
                failure_cases.append(result)

            if processed % 10000 == 0:
                print(f"Progress: {processed}/{len(tasks)} (failures: {len(failure_cases)})")

    elapsed = time.time() - start_time

    # 통계 분석
    total = len(patients)
    failure_count = len(failure_cases)
    gtpa_10 = success_count / total

    print(f"\n{'=' * 70}")
    print("Analysis Results")
    print("=" * 70)
    print(f"Total: {total}")
    print(f"GTPA@10: {gtpa_10:.4%} ({success_count}/{total})")
    print(f"Failures: {failure_count} ({failure_count / total:.4%})")
    print(f"Time: {elapsed:.1f}s ({elapsed / total:.4f}s/case)")

    # 실패 원인별 분류 (상세)
    rank_counter = Counter()
    for fc in failure_cases:
        if fc.failure_reason.startswith("GT_RANK_"):
            try:
                rank = int(fc.failure_reason.split("_")[-1])
                rank_counter[rank] = rank_counter.get(rank, 0) + 1
            except ValueError:
                rank_counter[fc.failure_reason] = rank_counter.get(fc.failure_reason, 0) + 1
        else:
            rank_counter[fc.failure_reason] = rank_counter.get(fc.failure_reason, 0) + 1

    print("\n[Failure Reason Distribution]")

    # GT Rank 분포 (10 초과인 경우만)
    rank_distribution = defaultdict(int)
    for fc in failure_cases:
        if fc.gt_rank:
            if fc.gt_rank <= 15:
                rank_distribution["11-15"] += 1
            elif fc.gt_rank <= 20:
                rank_distribution["16-20"] += 1
            elif fc.gt_rank <= 30:
                rank_distribution["21-30"] += 1
            elif fc.gt_rank <= 50:
                rank_distribution["31-50"] += 1
            else:
                rank_distribution[">50"] += 1
        else:
            rank_distribution["Not in top-50"] += 1

    for bin_name in ["11-15", "16-20", "21-30", "31-50", ">50", "Not in top-50"]:
        if bin_name in rank_distribution:
            count = rank_distribution[bin_name]
            print(f"  Rank {bin_name}: {count} ({count / failure_count:.1%})")

    # 질환별 실패율
    disease_failures = Counter(fc.ground_truth for fc in failure_cases)
    print("\n[Top 10 Diseases with Most Failures]")
    for disease, count in disease_failures.most_common(10):
        # 해당 질환의 총 케이스 수 계산
        total_disease = sum(1 for p in patient_data_list if loader.fr_to_eng.get(p["pathology"], p["pathology"]) == disease)
        fail_rate = count / total_disease if total_disease > 0 else 0
        print(f"  {disease}: {count}/{total_disease} ({fail_rate:.1%})")

    # Severity별 실패율
    severity_failures = Counter(fc.severity for fc in failure_cases)
    severity_totals = Counter(p["severity"] for p in patient_data_list)
    print("\n[Failures by Severity]")
    for sev in sorted(severity_failures.keys()):
        count = severity_failures[sev]
        total_sev = severity_totals[sev]
        rate = count / total_sev if total_sev > 0 else 0
        print(f"  Severity {sev}: {count}/{total_sev} ({rate:.2%})")

    # 증상 확인 패턴 분석
    confirmed_counts = [fc.confirmed_count for fc in failure_cases]
    denied_counts = [fc.denied_count for fc in failure_cases]
    evidence_counts = [fc.patient_evidence_count for fc in failure_cases]

    print("\n[Symptom Confirmation Pattern in Failures]")
    print(f"  Avg confirmed symptoms: {sum(confirmed_counts) / len(confirmed_counts):.1f}")
    print(f"  Avg denied symptoms: {sum(denied_counts) / len(denied_counts):.1f}")
    print(f"  Avg patient evidence count: {sum(evidence_counts) / len(evidence_counts):.1f}")

    # 상세 실패 케이스 샘플 (처음 20개)
    print("\n[Sample Failure Cases (first 20)]")
    print("-" * 70)
    for fc in failure_cases[:20]:
        print(f"Patient {fc.patient_idx}: {fc.ground_truth} (Severity {fc.severity})")
        print(f"  GT Rank: {fc.gt_rank or 'Not in top-50'}")
        print(f"  Confirmed: {fc.confirmed_count}, Denied: {fc.denied_count}, Total evidences: {fc.patient_evidence_count}")
        if fc.top_10_predictions:
            top3 = ", ".join(
                f"{p['name'][:20]}({p['score']:.2f})" for p in fc.top_10_predictions[:3]
            )
            print(f"  Top-3: {top3}")
        print()

    # 결과 저장
    results = {
        "total": total,
        "success": success_count,
        "failures": failure_count,
        "gtpa_10": gtpa_10,
        "elapsed": elapsed,
        "rank_distribution": dict(rank_distribution),
        "disease_failures": {
            disease: {
                "failures": count,
                "total": sum(1 for p in patient_data_list if loader.fr_to_eng.get(p["pathology"], p["pathology"]) == disease),
            }
            for disease, count in disease_failures.most_common(20)
        },
        "severity_failures": {
            str(sev): {
                "failures": severity_failures[sev],
                "total": severity_totals[sev],
                "rate": severity_failures[sev] / severity_totals[sev] if severity_totals[sev] > 0 else 0,
            }
            for sev in sorted(severity_failures.keys())
        },
        "symptom_pattern": {
            "avg_confirmed": sum(confirmed_counts) / len(confirmed_counts) if confirmed_counts else 0,
            "avg_denied": sum(denied_counts) / len(denied_counts) if denied_counts else 0,
            "avg_evidence": sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0,
        },
        "sample_cases": [asdict(fc) for fc in failure_cases[:50]],
    }

    output_path = Path("results/gtpa10_failure_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    run_failure_analysis(n_samples=args.n_samples, n_workers=args.workers)
