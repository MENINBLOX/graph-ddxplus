#!/usr/bin/env python3
"""매핑 수정만 적용한 전체 벤치마크.

목표: pls_irreg 매핑 수정이 전체 성능에 미치는 영향 확인.
- pls_irreg: C0003811 (Arrhythmia) → C0030252 (Palpitations)
"""

import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

NEO4J_PORTS = [7687, 7688, 7689, 7690, 7691, 7692, 7693, 7694]

# 매핑 수정
MAPPING_FIX = {"pls_irreg": "C0030252"}  # Palpitations


def run_diagnosis_with_mapping(args: tuple) -> dict | None:
    """매핑 수정 적용 진단."""
    patient_idx, patient_data, loader_data, neo4j_port, apply_fix = args

    from src.data_loader import DDXPlusLoader, Patient
    from src.umls_kg import UMLSKG

    loader = DDXPlusLoader()
    loader._symptom_mapping = loader_data["symptom_mapping"].copy()
    loader._disease_mapping = loader_data["disease_mapping"]
    loader._fr_to_eng = loader_data["fr_to_eng"]
    loader._conditions = loader_data["conditions"]

    # 매핑 수정 적용
    if apply_fix:
        for code, cui in MAPPING_FIX.items():
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

        # 환자 증상 CUI
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
        correct_at_1 = gt_cui == predicted_cuis[0] if gt_cui and predicted_cuis else False
        correct_at_10 = gt_cui in predicted_cuis if gt_cui else False

        kg.close()

        return {
            "patient_idx": patient_idx,
            "correct_at_1": correct_at_1,
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

    # 전체 테스트셋 로드
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

    # 테스트 설정
    tests = [
        {"name": "original", "apply_fix": False},
        {"name": "pls_irreg_fixed", "apply_fix": True},
    ]

    results_summary = {}

    for test in tests:
        print(f"\n{'='*60}")
        print(f"Test: {test['name']}")
        print("=" * 60)

        start_time = time.time()

        tasks = [
            (idx, patients_data[idx], loader_data, NEO4J_PORTS[idx % 8], test["apply_fix"])
            for idx in range(len(patients_data))
        ]

        results = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(run_diagnosis_with_mapping, t): t[0] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Testing"):
                r = future.result()
                if r and "error" not in r:
                    results.append(r)

        elapsed = time.time() - start_time

        # 집계
        correct_at_1 = sum(1 for r in results if r["correct_at_1"])
        correct_at_10 = sum(1 for r in results if r["correct_at_10"])
        avg_il = sum(r["il"] for r in results) / len(results) if results else 0

        gtpa_1 = correct_at_1 / len(results) if results else 0
        gtpa_10 = correct_at_10 / len(results) if results else 0

        # 질환별 통계
        disease_stats = {}
        for r in results:
            disease = r["gt_disease"]
            if disease not in disease_stats:
                disease_stats[disease] = {"total": 0, "correct_1": 0, "correct_10": 0}
            disease_stats[disease]["total"] += 1
            if r["correct_at_1"]:
                disease_stats[disease]["correct_1"] += 1
            if r["correct_at_10"]:
                disease_stats[disease]["correct_10"] += 1

        results_summary[test["name"]] = {
            "config": test,
            "total": len(results),
            "correct_at_1": correct_at_1,
            "correct_at_10": correct_at_10,
            "gtpa_1": gtpa_1,
            "gtpa_10": gtpa_10,
            "avg_il": avg_il,
            "elapsed_sec": elapsed,
            "disease_stats": disease_stats,
        }

        print(f"\n  Total: {len(results)}")
        print(f"  GTPA@1: {correct_at_1}/{len(results)} ({gtpa_1:.2%})")
        print(f"  GTPA@10: {correct_at_10}/{len(results)} ({gtpa_10:.2%})")
        print(f"  Avg IL: {avg_il:.1f}")

    # 비교
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    orig = results_summary["original"]
    fixed = results_summary["pls_irreg_fixed"]

    print(f"\n{'Metric':<15} {'Original':<15} {'Fixed':<15} {'Delta':<15}")
    print("-" * 60)
    print(f"{'GTPA@1':<15} {orig['gtpa_1']:.2%}{'':>6} {fixed['gtpa_1']:.2%}{'':>6} {(fixed['gtpa_1'] - orig['gtpa_1'])*100:+.2f}pp")
    print(f"{'GTPA@10':<15} {orig['gtpa_10']:.2%}{'':>6} {fixed['gtpa_10']:.2%}{'':>6} {(fixed['gtpa_10'] - orig['gtpa_10'])*100:+.2f}pp")

    # Atrial fibrillation 비교
    print("\n-- Atrial fibrillation --")
    for name in ["original", "pls_irreg_fixed"]:
        stats = results_summary[name]["disease_stats"]["Atrial fibrillation"]
        gtpa1 = stats["correct_1"] / stats["total"]
        gtpa10 = stats["correct_10"] / stats["total"]
        print(f"  {name}: GTPA@1={gtpa1:.2%}, GTPA@10={gtpa10:.2%}")

    with open("results/mapping_only_experiment.json", "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print("\nResults saved to: results/mapping_only_experiment.json")


if __name__ == "__main__":
    main()
