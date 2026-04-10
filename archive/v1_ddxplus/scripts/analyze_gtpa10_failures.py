#!/usr/bin/env python3
"""GTPA@10 실패 케이스 분석.

전체 134,529 케이스 중 GTPA@10 실패 케이스(0.23%, ~310개)를 추출하고 분석.
"""

import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

NEO4J_PORTS = [7687, 7688, 7689, 7690, 7691, 7692, 7693, 7694]


def run_diagnosis_detailed(args: tuple) -> dict | None:
    """상세 진단 정보 수집."""
    patient_idx, patient_data, loader_data, neo4j_port = args

    from src.data_loader import DDXPlusLoader, Patient
    from src.umls_kg import UMLSKG

    loader = DDXPlusLoader()
    loader._symptom_mapping = loader_data["symptom_mapping"]
    loader._disease_mapping = loader_data["disease_mapping"]
    loader._fr_to_eng = loader_data["fr_to_eng"]
    loader._conditions = loader_data["conditions"]

    try:
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

        # GT 질환 정보
        gt_disease_fr = patient.pathology
        gt_disease_eng = loader.fr_to_eng.get(gt_disease_fr, gt_disease_fr)
        gt_cui = loader.get_disease_cui(gt_disease_eng)

        # 환자 증상 CUI
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

        for _ in range(50):
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

            should_stop, stop_reason = kg.should_stop(
                max_il=50,
                min_il=10,
                confidence_threshold=0.30,
                gap_threshold=0.04,
                relative_gap_threshold=1.5,
            )
            if should_stop:
                break

        # 진단 결과
        diagnosis_candidates = kg.get_diagnosis_candidates(top_k=10)
        predicted_cuis = [d.cui for d in diagnosis_candidates]
        predicted_names = [d.name for d in diagnosis_candidates]
        predicted_scores = [d.score for d in diagnosis_candidates]

        correct_at_10 = gt_cui in predicted_cuis[:10] if gt_cui else False

        # GT 질환이 KG에 있는지 확인
        gt_in_kg = gt_cui is not None

        # GT 질환의 순위 (없으면 -1)
        gt_rank = -1
        if gt_cui:
            for i, cui in enumerate(predicted_cuis):
                if cui == gt_cui:
                    gt_rank = i + 1
                    break

        kg.close()

        return {
            "patient_idx": patient_idx,
            "correct_at_10": correct_at_10,
            "gt_disease_fr": gt_disease_fr,
            "gt_disease_eng": gt_disease_eng,
            "gt_cui": gt_cui,
            "gt_in_kg": gt_in_kg,
            "gt_rank": gt_rank,
            "predicted_top3": predicted_names[:3],
            "predicted_scores_top3": predicted_scores[:3],
            "il": il,
            "confirmed": confirmed_count,
            "denied": denied_count,
            "total_symptoms": len(patient_positive_cuis),
            "age": patient.age,
            "sex": patient.sex,
            "severity": loader_data.get("severity_map", {}).get(patient_idx),
        }

    except Exception as e:
        kg.close()
        return {"error": str(e), "patient_idx": patient_idx}


def main():
    from src.data_loader import DDXPlusLoader

    print("=" * 70)
    print("GTPA@10 Failure Analysis")
    print("=" * 70)

    # 데이터 로드
    loader = DDXPlusLoader()
    all_patients = loader.load_patients(split="test", n_samples=None, severity=None)
    print(f"Loaded {len(all_patients):,} patients")

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

    # 1단계: 모든 케이스 실행하여 실패 케이스 식별
    print("\n[Phase 1] Identifying GTPA@10 failures...")

    tasks = [
        (i, patients_data[i], loader_data, NEO4J_PORTS[i % 8])
        for i in range(len(patients_data))
    ]

    all_results = []
    failures = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_diagnosis_detailed, t): t[0] for t in tasks}

        with tqdm(total=len(tasks), desc="Diagnosing") as pbar:
            for future in as_completed(futures):
                r = future.result()
                if r and "error" not in r:
                    all_results.append(r)
                    if not r["correct_at_10"]:
                        failures.append(r)
                pbar.update(1)

    print(f"\nTotal: {len(all_results):,}, Failures: {len(failures)}")

    # 2단계: 실패 케이스 통계 분석
    print("\n[Phase 2] Statistical Analysis...")

    # 질환별 실패 분포
    disease_failures = Counter(f["gt_disease_eng"] for f in failures)

    # GT CUI 없음 (KG에 질환 없음)
    no_cui_failures = [f for f in failures if not f["gt_in_kg"]]

    # GT CUI 있음 but 실패
    has_cui_failures = [f for f in failures if f["gt_in_kg"]]

    # GT rank 분포 (Top-10 밖)
    rank_distribution = Counter()
    for f in has_cui_failures:
        if f["gt_rank"] == -1:
            rank_distribution["Not in results"] += 1
        elif f["gt_rank"] > 10:
            rank_distribution[f"Rank {f['gt_rank']}"] += 1

    # 증상 수 분포
    symptom_counts = [f["total_symptoms"] for f in failures]
    avg_symptoms_failure = sum(symptom_counts) / len(symptom_counts) if symptom_counts else 0

    # 전체 평균 증상 수
    all_symptom_counts = [r["total_symptoms"] for r in all_results]
    avg_symptoms_all = sum(all_symptom_counts) / len(all_symptom_counts) if all_symptom_counts else 0

    # IL 분포
    failure_ils = [f["il"] for f in failures]
    avg_il_failure = sum(failure_ils) / len(failure_ils) if failure_ils else 0

    # 결과 출력
    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS RESULTS")
    print("=" * 70)

    print(f"\n1. Overall Statistics:")
    print(f"   Total samples: {len(all_results):,}")
    print(f"   GTPA@10 failures: {len(failures)} ({len(failures)/len(all_results)*100:.3f}%)")

    print(f"\n2. Failure Breakdown:")
    print(f"   - GT disease not in KG: {len(no_cui_failures)} ({len(no_cui_failures)/len(failures)*100:.1f}%)")
    print(f"   - GT disease in KG but missed: {len(has_cui_failures)} ({len(has_cui_failures)/len(failures)*100:.1f}%)")

    print(f"\n3. Disease Distribution (Top 10 failure diseases):")
    for disease, count in disease_failures.most_common(10):
        pct = count / len(failures) * 100
        print(f"   - {disease}: {count} ({pct:.1f}%)")

    print(f"\n4. Symptom Count Comparison:")
    print(f"   - Avg symptoms (all): {avg_symptoms_all:.1f}")
    print(f"   - Avg symptoms (failures): {avg_symptoms_failure:.1f}")

    print(f"\n5. IL Comparison:")
    all_ils = [r["il"] for r in all_results]
    avg_il_all = sum(all_ils) / len(all_ils) if all_ils else 0
    print(f"   - Avg IL (all): {avg_il_all:.1f}")
    print(f"   - Avg IL (failures): {avg_il_failure:.1f}")

    if has_cui_failures:
        print(f"\n6. GT Rank Distribution (when GT in KG but missed):")
        for rank, count in sorted(rank_distribution.items(), key=lambda x: -x[1])[:10]:
            print(f"   - {rank}: {count}")

    # 개별 케이스 샘플
    print(f"\n7. Sample Failure Cases (first 10):")
    for i, f in enumerate(failures[:10]):
        print(f"\n   Case {i+1} (idx={f['patient_idx']}):")
        print(f"     GT: {f['gt_disease_eng']}")
        print(f"     GT in KG: {f['gt_in_kg']}, GT Rank: {f['gt_rank']}")
        print(f"     Predicted Top-3: {f['predicted_top3']}")
        print(f"     Scores: {[f'{s:.3f}' for s in f['predicted_scores_top3']]}")
        print(f"     Symptoms: {f['total_symptoms']}, Confirmed: {f['confirmed']}, IL: {f['il']}")

    # 결과 저장
    output = {
        "summary": {
            "total_samples": len(all_results),
            "total_failures": len(failures),
            "failure_rate": len(failures) / len(all_results),
            "no_cui_failures": len(no_cui_failures),
            "has_cui_failures": len(has_cui_failures),
            "avg_symptoms_all": avg_symptoms_all,
            "avg_symptoms_failures": avg_symptoms_failure,
            "avg_il_all": avg_il_all,
            "avg_il_failures": avg_il_failure,
        },
        "disease_distribution": dict(disease_failures.most_common()),
        "rank_distribution": dict(rank_distribution),
        "failure_cases": failures,
    }

    output_path = Path("results/failure_analysis_gtpa10.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
