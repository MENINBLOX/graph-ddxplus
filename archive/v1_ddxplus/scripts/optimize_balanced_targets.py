#!/usr/bin/env python3
"""목표: GTPA@1 > 83%, max_il < 1.0%, Avg IL <= 16.

기준점 분석:
- gap_0.01 (min_il=10): GTPA@1=80.05%, max_il=0.82% (1098), Avg IL=11.2
- min_il_14: GTPA@1=84.18%, max_il=1.45% (1944), Avg IL=14.8

목표 범위: min_il 10~14 사이에서 탐색
"""

import json
import sys
import time
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
            else:
                kg.state.add_denied(selected.cui)

            il += 1

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
        correct_at_1 = gt_cui == predicted_cuis[0] if gt_cui and predicted_cuis else False
        correct_at_10 = gt_cui in predicted_cuis if gt_cui else False

        kg.close()

        return {
            "patient_idx": patient_idx,
            "correct_at_1": correct_at_1,
            "correct_at_10": correct_at_10,
            "il": il,
            "reached_max_il": il >= max_il - 1,
        }

    except Exception as e:
        kg.close()
        return {"error": str(e), "patient_idx": patient_idx}


def main():
    from src.data_loader import DDXPlusLoader

    loader = DDXPlusLoader()
    all_patients = loader.load_patients(split="test", n_samples=None, severity=None)
    total_patients = len(all_patients)
    target_max_il_pct = 0.01  # 1%
    target_max_il_count = int(total_patients * target_max_il_pct)

    print(f"Total patients: {total_patients}")
    print(f"Target: GTPA@1 > 83%, max_il < 1.0% ({target_max_il_count} cases), Avg IL <= 16")
    print()

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

    # min_il 10~14 사이에서 탐색 + gap/conf/ratio 조정
    param_sets = [
        # 기준점
        {"name": "min_il_10_base", "min_il": 10, "confidence": 0.30, "gap": 0.01, "ratio": 1.5},

        # min_il 증가 (핵심 탐색 범위)
        {"name": "min_il_11", "min_il": 11, "confidence": 0.30, "gap": 0.01, "ratio": 1.5},
        {"name": "min_il_12", "min_il": 12, "confidence": 0.30, "gap": 0.01, "ratio": 1.5},
        {"name": "min_il_13", "min_il": 13, "confidence": 0.30, "gap": 0.01, "ratio": 1.5},

        # gap 감소 (더 빨리 종료 → max_il 감소)
        {"name": "min_il_12_gap005", "min_il": 12, "confidence": 0.30, "gap": 0.005, "ratio": 1.5},
        {"name": "min_il_13_gap005", "min_il": 13, "confidence": 0.30, "gap": 0.005, "ratio": 1.5},

        # confidence 감소 (더 빨리 종료)
        {"name": "min_il_12_conf25", "min_il": 12, "confidence": 0.25, "gap": 0.01, "ratio": 1.5},
        {"name": "min_il_13_conf25", "min_il": 13, "confidence": 0.25, "gap": 0.01, "ratio": 1.5},

        # ratio 감소 (더 빨리 종료)
        {"name": "min_il_12_ratio1.2", "min_il": 12, "confidence": 0.30, "gap": 0.01, "ratio": 1.2},
        {"name": "min_il_13_ratio1.2", "min_il": 13, "confidence": 0.30, "gap": 0.01, "ratio": 1.2},

        # 복합 조정 (max_il 낮추면서 GTPA@1 유지)
        {"name": "combo_12a", "min_il": 12, "confidence": 0.25, "gap": 0.005, "ratio": 1.3},
        {"name": "combo_12b", "min_il": 12, "confidence": 0.28, "gap": 0.008, "ratio": 1.4},
        {"name": "combo_13a", "min_il": 13, "confidence": 0.25, "gap": 0.005, "ratio": 1.3},
        {"name": "combo_13b", "min_il": 13, "confidence": 0.28, "gap": 0.008, "ratio": 1.4},
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

        correct_at_1 = sum(1 for r in results if r["correct_at_1"])
        correct_at_10 = sum(1 for r in results if r["correct_at_10"])
        max_il_cases = sum(1 for r in results if r["reached_max_il"])
        avg_il = sum(r["il"] for r in results) / len(results) if results else 0

        gtpa_1 = correct_at_1 / len(results) if results else 0
        gtpa_10 = correct_at_10 / len(results) if results else 0
        max_il_pct = max_il_cases / len(results) if results else 0

        # 목표 달성 여부
        meets_target = gtpa_1 > 0.83 and max_il_pct < 0.01 and avg_il <= 16

        results_summary[params["name"]] = {
            "params": params,
            "total": len(results),
            "correct_at_1": correct_at_1,
            "correct_at_10": correct_at_10,
            "gtpa_1": gtpa_1,
            "gtpa_10": gtpa_10,
            "max_il_cases": max_il_cases,
            "max_il_pct": max_il_pct,
            "avg_il": avg_il,
            "elapsed_sec": elapsed,
            "meets_target": meets_target,
        }

        status = "✅ PASS" if meets_target else "❌ FAIL"
        g1_ok = "✓" if gtpa_1 > 0.83 else "✗"
        mil_ok = "✓" if max_il_pct < 0.01 else "✗"
        il_ok = "✓" if avg_il <= 16 else "✗"

        print(f"\n  {status}")
        print(f"  GTPA@1: {gtpa_1:.2%} (> 83%: {g1_ok})")
        print(f"  GTPA@10: {gtpa_10:.2%}")
        print(f"  max_il: {max_il_cases} ({max_il_pct:.2%}) (< 1%: {mil_ok})")
        print(f"  Avg IL: {avg_il:.1f} (<= 16: {il_ok})")

    # 결과 요약
    print("\n" + "=" * 90)
    print("SUMMARY (sorted by GTPA@1)")
    print("=" * 90)
    print(f"\n{'Name':<20} {'GTPA@1':<10} {'GTPA@10':<10} {'max_il%':<10} {'Avg IL':<10} {'Status':<10}")
    print("-" * 90)

    for name, data in sorted(results_summary.items(), key=lambda x: -x[1]["gtpa_1"]):
        status = "✅" if data["meets_target"] else "❌"
        print(f"{name:<20} {data['gtpa_1']:.2%}{'':>2} {data['gtpa_10']:.2%}{'':>2} {data['max_il_pct']:.2%}{'':>3} {data['avg_il']:.1f}{'':>5} {status}")

    # 최적 설정 출력
    print("\n" + "=" * 60)
    print("OPTIMAL SETTINGS (GTPA@1 > 83% && max_il < 1% && Avg IL <= 16)")
    print("=" * 60)

    optimal = [(name, data) for name, data in results_summary.items() if data["meets_target"]]
    if optimal:
        optimal.sort(key=lambda x: -x[1]["gtpa_1"])
        for name, data in optimal:
            print(f"\n✅ {name}:")
            print(f"   GTPA@1: {data['gtpa_1']:.2%}")
            print(f"   GTPA@10: {data['gtpa_10']:.2%}")
            print(f"   max_il: {data['max_il_cases']} ({data['max_il_pct']:.2%})")
            print(f"   Avg IL: {data['avg_il']:.1f}")
            print(f"   Params: min_il={data['params']['min_il']}, conf={data['params']['confidence']}, gap={data['params']['gap']}, ratio={data['params']['ratio']}")
    else:
        print("\n❌ No settings meet all targets.")
        print("\nClosest settings:")
        # 가장 가까운 설정 찾기 (GTPA@1 > 83% 우선)
        for name, data in sorted(results_summary.items(), key=lambda x: -x[1]["gtpa_1"])[:3]:
            g1_ok = "✓" if data["gtpa_1"] > 0.83 else "✗"
            mil_ok = "✓" if data["max_il_pct"] < 0.01 else "✗"
            il_ok = "✓" if data["avg_il"] <= 16 else "✗"
            print(f"\n  {name}:")
            print(f"    GTPA@1: {data['gtpa_1']:.2%} ({g1_ok})")
            print(f"    max_il: {data['max_il_pct']:.2%} ({mil_ok})")
            print(f"    Avg IL: {data['avg_il']:.1f} ({il_ok})")

    with open("results/balanced_targets_optimization.json", "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print("\nResults saved to: results/balanced_targets_optimization.json")


if __name__ == "__main__":
    main()
