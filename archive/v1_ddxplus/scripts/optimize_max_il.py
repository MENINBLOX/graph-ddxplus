#!/usr/bin/env python3
"""max_il 도달 최소화를 위한 stopping criteria 최적화.

목표: max_il 도달 비율을 26% → 5% 이하로 줄이면서 GTPA@10 유지.
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


def run_diagnosis_with_params(args: tuple) -> dict | None:
    """파라미터별 진단."""
    patient_idx, patient_data, loader_data, neo4j_port, params = args

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

        for _ in range(max_il):
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

            # 커스텀 파라미터로 should_stop 호출
            should_stop, _ = kg.should_stop(
                max_il=max_il,
                min_il=params["min_il"],
                confidence_threshold=params["confidence"],
                gap_threshold=params["gap"],
                relative_gap_threshold=params["ratio"],
            )
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
            "reached_max_il": il >= max_il - 1,  # IL=49도 max_il 도달로 간주
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

    # 테스트할 파라미터 조합
    param_sets = [
        # 현재 설정 (baseline)
        {"name": "baseline", "min_il": 10, "confidence": 0.30, "gap": 0.04, "ratio": 1.5},

        # confidence 낮춤
        {"name": "conf_0.25", "min_il": 10, "confidence": 0.25, "gap": 0.04, "ratio": 1.5},
        {"name": "conf_0.20", "min_il": 10, "confidence": 0.20, "gap": 0.04, "ratio": 1.5},

        # gap 낮춤
        {"name": "gap_0.03", "min_il": 10, "confidence": 0.30, "gap": 0.03, "ratio": 1.5},
        {"name": "gap_0.02", "min_il": 10, "confidence": 0.30, "gap": 0.02, "ratio": 1.5},

        # ratio 낮춤
        {"name": "ratio_1.3", "min_il": 10, "confidence": 0.30, "gap": 0.04, "ratio": 1.3},
        {"name": "ratio_1.2", "min_il": 10, "confidence": 0.30, "gap": 0.04, "ratio": 1.2},

        # 복합 조정 (보수적)
        {"name": "combo_1", "min_il": 10, "confidence": 0.25, "gap": 0.03, "ratio": 1.3},

        # 복합 조정 (적극적)
        {"name": "combo_2", "min_il": 10, "confidence": 0.20, "gap": 0.02, "ratio": 1.2},
    ]

    results_summary = {}

    for params in param_sets:
        print(f"\n{'='*60}")
        print(f"Testing: {params['name']}")
        print(f"  min_il={params['min_il']}, conf={params['confidence']}, gap={params['gap']}, ratio={params['ratio']}")
        print("=" * 60)

        start_time = time.time()

        tasks = [
            (idx, patients_data[idx], loader_data, NEO4J_PORTS[idx % 8], params)
            for idx in range(len(patients_data))
        ]

        results = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(run_diagnosis_with_params, t): t[0] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Testing"):
                r = future.result()
                if r and "error" not in r:
                    results.append(r)

        elapsed = time.time() - start_time

        correct_at_10 = sum(1 for r in results if r["correct_at_10"])
        max_il_cases = sum(1 for r in results if r["reached_max_il"])
        avg_il = sum(r["il"] for r in results) / len(results) if results else 0

        gtpa_10 = correct_at_10 / len(results) if results else 0
        max_il_rate = max_il_cases / len(results) if results else 0

        results_summary[params["name"]] = {
            "params": params,
            "total": len(results),
            "correct_at_10": correct_at_10,
            "gtpa_10": gtpa_10,
            "max_il_cases": max_il_cases,
            "max_il_rate": max_il_rate,
            "avg_il": avg_il,
            "elapsed_sec": elapsed,
        }

        print(f"\n  GTPA@10: {gtpa_10:.2%}")
        print(f"  max_il rate: {max_il_rate:.2%} ({max_il_cases} cases)")
        print(f"  Avg IL: {avg_il:.1f}")

    # 비교 요약
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Name':<15} {'GTPA@10':<10} {'max_il%':<10} {'Avg IL':<10} {'GTPA@10 Δ':<12}")
    print("-" * 60)

    baseline_gtpa = results_summary["baseline"]["gtpa_10"]
    for name, data in sorted(results_summary.items(), key=lambda x: -x[1]["gtpa_10"]):
        delta = (data["gtpa_10"] - baseline_gtpa) * 100
        print(f"{name:<15} {data['gtpa_10']:.2%}{'':>2} {data['max_il_rate']:.2%}{'':>3} {data['avg_il']:.1f}{'':>5} {delta:+.2f}pp")

    # 최적 파라미터 선택 (max_il < 10% && GTPA@10 감소 < 0.1pp)
    print("\n" + "=" * 60)
    print("OPTIMAL CANDIDATES (max_il < 10% && GTPA@10 loss < 0.1pp)")
    print("=" * 60)

    for name, data in results_summary.items():
        gtpa_loss = (baseline_gtpa - data["gtpa_10"]) * 100
        if data["max_il_rate"] < 0.10 and gtpa_loss < 0.1:
            print(f"\n✅ {name}:")
            print(f"   GTPA@10: {data['gtpa_10']:.2%} (Δ={-gtpa_loss:+.2f}pp)")
            print(f"   max_il: {data['max_il_rate']:.2%}")
            print(f"   Avg IL: {data['avg_il']:.1f}")

    with open("results/max_il_optimization.json", "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print("\nResults saved to: results/max_il_optimization.json")


if __name__ == "__main__":
    main()
