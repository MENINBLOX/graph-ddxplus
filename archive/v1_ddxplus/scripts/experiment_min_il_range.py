#!/usr/bin/env python3
"""min_il 범위 테스트: 2~25.

학술적 배경:
- DDXPlus 평균 증상 수: 10.02개 (Tchango et al., 2022)
- 전문가 가설 생성: ~5분 내 (Elstein et al., 1978)
- Working memory 한계: 4-5개 항목 (Miller, 1956)

목적:
- min_il 변화에 따른 GTPA@1, Avg IL 트레이드오프 분석
- 최적 min_il 값의 근거 제시
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

NEO4J_PORTS = [7687, 7688, 7689, 7690, 7691, 7692, 7693, 7694]


def run_diagnosis_with_min_il(args: tuple) -> dict | None:
    """min_il별 진단 실행."""
    patient_idx, patient_data, loader_data, neo4j_port, min_il = args

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

        gt_disease_eng = loader.fr_to_eng.get(patient.pathology, patient.pathology)
        gt_cui = loader.get_disease_cui(gt_disease_eng)

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

        max_il = 50  # 안전장치
        il = 0

        for _ in range(max_il):
            candidates = kg.get_candidate_symptoms(
                initial_cui=initial_cui,
                limit=10,
                confirmed_cuis=kg.state.confirmed_cuis,
                denied_cuis=kg.state.denied_cuis,
            )

            if not candidates:
                break

            next_cui = candidates[0].cui

            # add_confirmed/add_denied both add to asked_cuis
            if next_cui in patient_positive_cuis:
                kg.state.add_confirmed(next_cui)
            else:
                kg.state.add_denied(next_cui)

            il += 1

            # Stopping criteria with variable min_il
            should_stop, _ = kg.should_stop(
                max_il=max_il,
                min_il=min_il,
                confidence_threshold=0.30,
                gap_threshold=0.005,
                relative_gap_threshold=1.5,
            )

            if should_stop:
                break

        # 최종 진단
        final_candidates = kg.get_diagnosis_candidates(top_k=10)
        top1_correct = False
        top10_correct = False

        if final_candidates:
            if final_candidates[0].cui == gt_cui:
                top1_correct = True
            for c in final_candidates[:10]:
                if c.cui == gt_cui:
                    top10_correct = True
                    break

        kg.close()

        return {
            "patient_idx": patient_idx,
            "min_il": min_il,
            "il": il,
            "top1_correct": top1_correct,
            "top10_correct": top10_correct,
        }

    except Exception as e:
        kg.close()
        return {"error": str(e), "patient_idx": patient_idx}


def main():
    from src.data_loader import DDXPlusLoader

    print("=" * 70)
    print("min_il 범위 테스트 (2~25)")
    print("=" * 70)

    loader = DDXPlusLoader()
    # Lazy loading via properties - access to initialize
    _ = loader.symptom_mapping
    _ = loader.disease_mapping
    _ = loader.fr_to_eng

    test_patients = loader.load_patients(split="test")
    total_patients = len(test_patients)
    print(f"Total test patients: {total_patients:,}")

    # 전체 테스트 또는 샘플링
    sample_size = total_patients  # 전체 테스트
    # sample_size = 10000  # 빠른 테스트용

    if sample_size < total_patients:
        import random
        random.seed(42)
        test_patients = random.sample(test_patients, sample_size)
        print(f"Sampled: {sample_size:,}")

    loader_data = {
        "symptom_mapping": loader._symptom_mapping,
        "disease_mapping": loader._disease_mapping,
        "fr_to_eng": loader._fr_to_eng,
        "conditions": loader._conditions,
    }

    patient_data_list = []
    for p in test_patients:
        patient_data_list.append({
            "age": p.age,
            "sex": p.sex,
            "initial_evidence": p.initial_evidence,
            "evidences": p.evidences,
            "pathology": p.pathology,
            "differential_diagnosis": p.differential_diagnosis,
        })

    # min_il 범위: 2~25
    min_il_values = list(range(2, 26))

    results = {min_il: {"correct_at_1": 0, "correct_at_10": 0, "total_il": 0, "count": 0, "max_il_cases": 0}
               for min_il in min_il_values}

    num_workers = len(NEO4J_PORTS)

    for min_il in min_il_values:
        print(f"\n--- Testing min_il={min_il} ---")
        start_time = time.time()

        tasks = []
        for idx, pd in enumerate(patient_data_list):
            port = NEO4J_PORTS[idx % num_workers]
            tasks.append((idx, pd, loader_data, port, min_il))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(run_diagnosis_with_min_il, t): t for t in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"min_il={min_il}"):
                result = future.result()
                if result and "error" not in result:
                    results[min_il]["count"] += 1
                    results[min_il]["total_il"] += result["il"]
                    if result["top1_correct"]:
                        results[min_il]["correct_at_1"] += 1
                    if result["top10_correct"]:
                        results[min_il]["correct_at_10"] += 1
                    if result["il"] >= 50:
                        results[min_il]["max_il_cases"] += 1

        elapsed = time.time() - start_time
        r = results[min_il]
        if r["count"] > 0:
            gtpa1 = r["correct_at_1"] / r["count"]
            gtpa10 = r["correct_at_10"] / r["count"]
            avg_il = r["total_il"] / r["count"]
            max_il_pct = r["max_il_cases"] / r["count"]
            print(f"  GTPA@1: {gtpa1:.2%}, GTPA@10: {gtpa10:.2%}, Avg IL: {avg_il:.1f}, max_il: {max_il_pct:.2%}")
            print(f"  Time: {elapsed/60:.1f} min")

    # 결과 저장
    output = {}
    for min_il in min_il_values:
        r = results[min_il]
        if r["count"] > 0:
            output[f"min_il_{min_il}"] = {
                "min_il": min_il,
                "total": r["count"],
                "correct_at_1": r["correct_at_1"],
                "correct_at_10": r["correct_at_10"],
                "gtpa_1": r["correct_at_1"] / r["count"],
                "gtpa_10": r["correct_at_10"] / r["count"],
                "avg_il": r["total_il"] / r["count"],
                "max_il_cases": r["max_il_cases"],
                "max_il_pct": r["max_il_cases"] / r["count"],
            }

    output_path = Path("results/min_il_range_experiment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # 요약 테이블
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'min_il':>8} {'GTPA@1':>10} {'GTPA@10':>10} {'Avg IL':>10} {'max_il%':>10}")
    print("-" * 50)
    for min_il in min_il_values:
        if f"min_il_{min_il}" in output:
            r = output[f"min_il_{min_il}"]
            print(f"{min_il:>8} {r['gtpa_1']:>10.2%} {r['gtpa_10']:>10.2%} {r['avg_il']:>10.1f} {r['max_il_pct']:>10.2%}")


if __name__ == "__main__":
    main()
