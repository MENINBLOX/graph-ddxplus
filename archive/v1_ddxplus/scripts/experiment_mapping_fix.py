#!/usr/bin/env python3
"""매핑 수정 실험.

Atrial fibrillation 실패 케이스에서:
- pls_irreg: Arrhythmia (C0003811) → Palpitations (C0030252) 변경 테스트
"""

import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

NEO4J_PORTS = [7687, 7688, 7689, 7690, 7691, 7692, 7693, 7694]

# 매핑 수정 후보
MAPPING_FIXES = {
    # pls_irreg: Arrhythmia → Palpitations
    "pls_irreg": {
        "original": "C0003811",  # Arrhythmia (1 disease)
        "alternatives": [
            ("C0030252", "Palpitations"),  # 6 diseases
            ("C0237892", "Irregular heartbeat"),  # 확인 필요
        ],
    },
}


def run_diagnosis_with_mapping(args: tuple) -> dict | None:
    """매핑 변경 후 진단."""
    patient_idx, patient_data, loader_data, neo4j_port, mapping_override = args

    from src.data_loader import DDXPlusLoader, Patient
    from src.umls_kg import UMLSKG

    loader = DDXPlusLoader()
    loader._symptom_mapping = loader_data["symptom_mapping"].copy()
    loader._disease_mapping = loader_data["disease_mapping"]
    loader._fr_to_eng = loader_data["fr_to_eng"]
    loader._conditions = loader_data["conditions"]

    # 매핑 오버라이드
    for code, cui in mapping_override.items():
        if code in loader._symptom_mapping:
            loader._symptom_mapping[code]["cui"] = cui

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

        # 수정된 매핑으로 환자 증상 CUI 가져오기
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

            il += 1

            should_stop, _ = kg.should_stop(max_il=50, min_il=10)
            if should_stop:
                break

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
        }

    except Exception as e:
        kg.close()
        return {"error": str(e), "patient_idx": patient_idx}


def main():
    from src.data_loader import DDXPlusLoader

    # 실패 케이스 로드
    with open("results/failure_analysis_gtpa10.json") as f:
        failure_data = json.load(f)

    failures = failure_data["failure_cases"]

    # Atrial fibrillation 실패 케이스만 필터
    afib_failures = [f for f in failures if f["gt_disease_eng"] == "Atrial fibrillation"]
    print(f"Atrial fibrillation failures: {len(afib_failures)}")

    # 데이터 로드
    loader = DDXPlusLoader()
    all_patients = loader.load_patients(split="test", n_samples=None, severity=None)

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

    failure_indices = [f["patient_idx"] for f in afib_failures]

    # 테스트할 매핑
    mapping_tests = [
        {"name": "original", "mapping": {}},
        {"name": "pls_irreg→Palpitations", "mapping": {"pls_irreg": "C0030252"}},
    ]

    results_summary = {}

    for test in mapping_tests:
        print(f"\n{'='*60}")
        print(f"Mapping: {test['name']}")
        print("=" * 60)

        tasks = [
            (idx, patients_data[idx], loader_data, NEO4J_PORTS[i % 8], test["mapping"])
            for i, idx in enumerate(failure_indices)
        ]

        results = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(run_diagnosis_with_mapping, t): t[0] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Testing"):
                r = future.result()
                if r and "error" not in r:
                    results.append(r)

        recovered = sum(1 for r in results if r["correct_at_10"])
        recovery_rate = recovered / len(results) if results else 0
        avg_confirmed = sum(r["confirmed"] for r in results) / len(results) if results else 0

        results_summary[test["name"]] = {
            "mapping": test["mapping"],
            "total": len(results),
            "recovered": recovered,
            "recovery_rate": recovery_rate,
            "avg_confirmed": avg_confirmed,
        }

        print(f"\n  Recovered: {recovered}/{len(results)} ({recovery_rate:.1%})")
        print(f"  Avg confirmed: {avg_confirmed:.1f}")

    # 결과 저장
    print("\n" + "=" * 60)
    print("SUMMARY (Atrial fibrillation)")
    print("=" * 60)

    for name, data in results_summary.items():
        print(f"\n{name}:")
        print(f"  Recovered: {data['recovered']}/{data['total']} ({data['recovery_rate']:.1%})")
        print(f"  Avg confirmed: {data['avg_confirmed']:.1f}")

    with open("results/mapping_fix_experiment.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\nResults saved to: results/mapping_fix_experiment.json")


if __name__ == "__main__":
    main()
