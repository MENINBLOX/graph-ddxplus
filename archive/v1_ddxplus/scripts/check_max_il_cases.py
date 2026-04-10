#!/usr/bin/env python3
"""max_il 달성 케이스 분석."""

import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

NEO4J_PORTS = [7687, 7688, 7689, 7690, 7691, 7692, 7693, 7694]


def run_diagnosis(args: tuple) -> dict | None:
    """진단 수행 및 IL 기록."""
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

        il = 0
        confirmed_count = 1
        max_il = 50
        stop_reason = None

        for _ in range(max_il):
            candidates = kg.get_candidate_symptoms(
                initial_cui=initial_cui,
                limit=10,
                confirmed_cuis=kg.state.confirmed_cuis,
                denied_cuis=kg.state.denied_cuis,
                asked_cuis=kg.state.asked_cuis,
            )

            if not candidates:
                stop_reason = "no_candidates"
                break

            selected = candidates[0]

            if selected.cui in patient_positive_cuis:
                kg.state.add_confirmed(selected.cui)
                confirmed_count += 1
            else:
                kg.state.add_denied(selected.cui)

            il += 1

            should_stop, reason = kg.should_stop(max_il=max_il, min_il=10)
            if should_stop:
                stop_reason = reason
                break

        if il == max_il:
            stop_reason = "max_il_reached"

        diagnosis = kg.get_diagnosis_candidates(top_k=10)
        predicted_cuis = [d.cui for d in diagnosis]
        correct_at_10 = gt_cui in predicted_cuis if gt_cui else False

        kg.close()

        return {
            "patient_idx": patient_idx,
            "correct_at_10": correct_at_10,
            "il": il,
            "confirmed": confirmed_count,
            "gt_disease": gt_disease_eng,
            "stop_reason": stop_reason,
            "reached_max_il": il >= max_il,
        }

    except Exception as e:
        kg.close()
        return {"error": str(e), "patient_idx": patient_idx}


def main():
    from src.data_loader import DDXPlusLoader

    loader = DDXPlusLoader()
    all_patients = loader.load_patients(split="test", n_samples=None, severity=None)
    print(f"Total patients: {len(all_patients)}")

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

    tasks = [
        (idx, patients_data[idx], loader_data, NEO4J_PORTS[idx % 8])
        for idx in range(len(patients_data))
    ]

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_diagnosis, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Analyzing"):
            r = future.result()
            if r and "error" not in r:
                results.append(r)

    # 분석
    print(f"\n{'='*60}")
    print("IL 분포 분석")
    print("="*60)

    il_distribution = Counter(r["il"] for r in results)
    max_il_cases = [r for r in results if r["reached_max_il"]]

    print(f"\n총 케이스: {len(results)}")
    print(f"max_il(50) 달성 케이스: {len(max_il_cases)} ({len(max_il_cases)/len(results)*100:.2f}%)")

    # IL 분포 (상위 10개)
    print("\nIL 분포 (상위 10개):")
    for il, count in sorted(il_distribution.items(), key=lambda x: -x[1])[:10]:
        print(f"  IL={il}: {count} ({count/len(results)*100:.2f}%)")

    # 평균/중간값
    ils = [r["il"] for r in results]
    avg_il = sum(ils) / len(ils)
    sorted_ils = sorted(ils)
    median_il = sorted_ils[len(sorted_ils)//2]
    max_observed_il = max(ils)

    print(f"\n평균 IL: {avg_il:.1f}")
    print(f"중간값 IL: {median_il}")
    print(f"최대 IL: {max_observed_il}")

    # max_il 케이스 상세
    if max_il_cases:
        print(f"\n{'='*60}")
        print(f"max_il 달성 케이스 상세 ({len(max_il_cases)}건)")
        print("="*60)

        disease_dist = Counter(r["gt_disease"] for r in max_il_cases)
        print("\n질환별 분포:")
        for disease, count in disease_dist.most_common(10):
            print(f"  {disease}: {count}")

        correct_in_max_il = sum(1 for r in max_il_cases if r["correct_at_10"])
        print(f"\nmax_il 케이스 중 GTPA@10 정확: {correct_in_max_il}/{len(max_il_cases)} ({correct_in_max_il/len(max_il_cases)*100:.1f}%)")

    # Stop reason 분포
    stop_reasons = Counter(r["stop_reason"] for r in results)
    print(f"\n{'='*60}")
    print("종료 이유 분포")
    print("="*60)
    for reason, count in stop_reasons.most_common():
        print(f"  {reason}: {count} ({count/len(results)*100:.2f}%)")


if __name__ == "__main__":
    main()
