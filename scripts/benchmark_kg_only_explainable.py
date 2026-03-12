#!/usr/bin/env python3
"""KG-only Explainable Benchmark.

UMLS 기반 2-hop KG만으로 진단 수행.
- 각 증상 선택에 대한 근거 (연관 질환) 제공
- 최종 진단에 대한 근거 (일치 증상) 제공
- CPU 멀티프로세스 지원
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
class DiagnosisStep:
    """진단 단계 기록."""

    step: int
    symptom_cui: str
    symptom_name: str
    patient_response: str  # "yes", "no", "invalid"
    related_diseases: list[dict] = field(default_factory=list)


@dataclass
class DiagnosisResult:
    """단일 환자 진단 결과."""

    patient_id: int
    ground_truth: str
    predicted: str
    correct: bool
    interaction_length: int
    steps: list[DiagnosisStep] = field(default_factory=list)
    final_diagnosis_reason: dict = field(default_factory=dict)


def run_single_diagnosis(args: tuple) -> DiagnosisResult | None:
    """단일 환자 진단 수행 (프로세스 워커)."""
    patient_idx, patient_data, loader_data, top_n, max_il, include_explanation = args

    # 각 프로세스에서 import (pickle 문제 방지)
    from src.data_loader import DDXPlusLoader
    from src.umls_kg import UMLSKG

    # 로더와 KG 연결
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
        from src.data_loader import Patient
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

        steps = []

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

            # 첫 번째 후보 선택 (KG 기반 최적)
            selected = candidates[0]

            # 연관 질환 정보 (설명용)
            related_diseases = []
            if include_explanation:
                diseases = kg.get_related_diseases_for_symptom(
                    symptom_cui=selected.cui,
                    top_k=3,
                    confirmed_cuis=kg.state.confirmed_cuis,
                    denied_cuis=kg.state.denied_cuis,
                )
                related_diseases = [
                    {
                        "name": d.name,
                        "score": round(d.score, 3),
                        "matched": d.matched_symptoms,
                        "total": d.total_symptoms,
                    }
                    for d in diseases
                ]

            # 환자 응답 시뮬레이션
            if selected.cui in patient_positive_cuis:
                response = "yes"
                kg.state.add_confirmed(selected.cui)
            else:
                response = "no"
                kg.state.add_denied(selected.cui)

            steps.append(DiagnosisStep(
                step=step + 1,
                symptom_cui=selected.cui,
                symptom_name=selected.name,
                patient_response=response,
                related_diseases=related_diseases,
            ))

            # 종료 조건 확인
            should_stop, reason = kg.should_stop(max_il=max_il)
            if should_stop:
                break

        # 최종 진단 (설명 포함)
        if include_explanation:
            explained_candidates = kg.get_explained_diagnosis_candidates(top_k=5)
            predicted_cui = explained_candidates[0].cui if explained_candidates else None
            predicted_name = explained_candidates[0].name if explained_candidates else "Unknown"

            # 상세 진단 근거
            final_reason = {}
            if explained_candidates:
                top = explained_candidates[0]
                final_reason = {
                    "disease": top.name,
                    "score": round(top.score, 3),
                    "matched_symptoms": top.matched_symptoms,
                    "denied_symptoms": top.denied_symptoms,
                    "unasked_symptoms": top.unasked_symptoms,
                    "matched_count": top.matched_count,
                    "denied_count": top.denied_count,
                    "total_symptoms": top.total_symptoms,
                    "coverage": round(top.coverage, 3),
                    "explanation": top.explanation,
                }
        else:
            diagnosis_candidates = kg.get_diagnosis_candidates(top_k=5)
            predicted_cui = diagnosis_candidates[0].cui if diagnosis_candidates else None
            predicted_name = diagnosis_candidates[0].name if diagnosis_candidates else "Unknown"
            final_reason = {}

        correct = predicted_cui == gt_cui

        kg.close()

        return DiagnosisResult(
            patient_id=patient_idx,
            ground_truth=gt_disease_name,
            predicted=predicted_name,
            correct=correct,
            interaction_length=len(steps),
            steps=steps,
            final_diagnosis_reason=final_reason,
        )

    except Exception as e:
        kg.close()
        print(f"Error processing patient {patient_idx}: {e}")
        return None


def run_benchmark(
    n_samples: int = 100,
    severity: int = 2,
    top_n: int = 10,
    max_il: int = 50,
    n_workers: int = 4,
    include_explanation: bool = True,
    verbose: bool = True,
) -> dict:
    """KG-only 벤치마크 실행."""
    from src.data_loader import DDXPlusLoader

    print("=" * 70)
    print("KG-only Explainable Benchmark")
    print("=" * 70)
    print(f"Samples: {n_samples}, Severity: {severity}")
    print(f"Top-N: {top_n}, Max IL: {max_il}")
    print(f"Workers: {n_workers}, Explanation: {include_explanation}")
    print("=" * 70)

    # 데이터 로드
    loader = DDXPlusLoader()
    patients = loader.load_patients(
        split="test",
        n_samples=n_samples,
        severity=severity,
    )

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
        (i, patient_data_list[i], loader_data, top_n, max_il, include_explanation)
        for i in range(len(patients))
    ]

    # 멀티프로세스 실행
    results: list[DiagnosisResult] = []
    start_time = time.time()

    if n_workers == 1:
        # 싱글 프로세스 (디버깅용)
        for task in tasks:
            result = run_single_diagnosis(task)
            if result:
                results.append(result)
                if verbose and len(results) <= 5:
                    _print_result(result)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(run_single_diagnosis, task): i for i, task in enumerate(tasks)}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

                    if verbose and len(results) <= 5:
                        _print_result(result)

                    if len(results) % 100 == 0:
                        print(f"Progress: {len(results)}/{len(tasks)}")

    elapsed = time.time() - start_time

    # 통계 계산
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total > 0 else 0
    avg_il = sum(r.interaction_length for r in results) / total if total > 0 else 0

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Total: {total}")
    print(f"GTPA@1: {correct}/{total} ({accuracy:.1%})")
    print(f"Avg IL: {avg_il:.1f}")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.2f}s/case)")

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_il": avg_il,
        "elapsed": elapsed,
        "results": [asdict(r) for r in results],
    }


def _print_result(result: DiagnosisResult) -> None:
    """결과 출력."""
    status = "✅" if result.correct else "❌"
    print(f"\n[Patient {result.patient_id}] {status}")
    print(f"  GT: {result.ground_truth}")
    print(f"  Pred: {result.predicted}")
    print(f"  IL: {result.interaction_length}")

    if result.steps and result.steps[0].related_diseases:
        print("  Sample step:")
        step = result.steps[0]
        print(f"    Q: {step.symptom_name} → {step.patient_response}")
        if step.related_diseases:
            diseases = ", ".join([
                f"{d['name']}({d['score']:.0%})"
                for d in step.related_diseases[:2]
            ])
            print(f"    Related: {diseases}")

    if result.final_diagnosis_reason:
        r = result.final_diagnosis_reason
        # 새 형식 (ExplainedDiagnosis)
        if "explanation" in r:
            print("  " + "-" * 50)
            for line in r["explanation"].split("\n"):
                print(f"  {line}")
        # 이전 형식 (호환성)
        elif "matched_symptoms" in r and isinstance(r["matched_symptoms"], int):
            print(f"  Reason: {r['matched_symptoms']}/{r['total_symptoms']} symptoms matched")


def main():
    parser = argparse.ArgumentParser(description="KG-only Explainable Benchmark")
    parser.add_argument("-n", "--n-samples", type=int, default=100)
    parser.add_argument("--severity", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-il", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-explanation", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    results = run_benchmark(
        n_samples=args.n_samples,
        severity=args.severity,
        top_n=args.top_n,
        max_il=args.max_il,
        n_workers=args.workers,
        include_explanation=not args.no_explanation,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
