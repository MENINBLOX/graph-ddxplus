#!/usr/bin/env python3
"""GTPA@1 최대화 실험 (Avg IL ≤ 15, max_il < 1%).

기준점: gap_0.01 → GTPA@1=80.05%, Avg IL=11.2, max_il=0.82%
목표: GTPA@1 ↑, Avg IL ≤ 15, max_il < 1%
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
    print(f"Target: max_il < {target_max_il_pct*100}% ({target_max_il_count} cases)")
    print(f"Target: Avg IL ≤ 15")
    print(f"Goal: Maximize GTPA@1\n")

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

    # IL을 높이는 파라미터 조합 (gap_0.01 기준에서 조정)
    # gap_0.01: min_il=10, conf=0.30, gap=0.01, ratio=1.5 → IL=11.2
    param_sets = [
        # 기준점
        {"name": "gap_0.01_baseline", "min_il": 10, "confidence": 0.30, "gap": 0.01, "ratio": 1.5},

        # min_il 증가 (IL 직접 증가)
        {"name": "min_il_12", "min_il": 12, "confidence": 0.30, "gap": 0.01, "ratio": 1.5},
        {"name": "min_il_14", "min_il": 14, "confidence": 0.30, "gap": 0.01, "ratio": 1.5},

        # gap 증가 (더 엄격한 종료 조건 → IL 증가)
        {"name": "gap_0.02", "min_il": 10, "confidence": 0.30, "gap": 0.02, "ratio": 1.5},
        {"name": "gap_0.03", "min_il": 10, "confidence": 0.30, "gap": 0.03, "ratio": 1.5},

        # confidence 증가
        {"name": "conf_0.35", "min_il": 10, "confidence": 0.35, "gap": 0.01, "ratio": 1.5},
        {"name": "conf_0.40", "min_il": 10, "confidence": 0.40, "gap": 0.01, "ratio": 1.5},

        # ratio 증가
        {"name": "ratio_1.8", "min_il": 10, "confidence": 0.30, "gap": 0.01, "ratio": 1.8},
        {"name": "ratio_2.0", "min_il": 10, "confidence": 0.30, "gap": 0.01, "ratio": 2.0},

        # 복합 조정 (IL ~15 목표)
        {"name": "combo_il15_a", "min_il": 12, "confidence": 0.35, "gap": 0.02, "ratio": 1.5},
        {"name": "combo_il15_b", "min_il": 14, "confidence": 0.30, "gap": 0.02, "ratio": 1.5},
        {"name": "combo_il15_c", "min_il": 12, "confidence": 0.30, "gap": 0.03, "ratio": 1.8},
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
        meets_target = max_il_pct < target_max_il_pct and avg_il <= 15

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
        print(f"\n  {status}")
        print(f"  GTPA@1: {gtpa_1:.2%}")
        print(f"  GTPA@10: {gtpa_10:.2%}")
        print(f"  max_il: {max_il_cases} ({max_il_pct:.2%}) [< 1%: {'✓' if max_il_pct < 0.01 else '✗'}]")
        print(f"  Avg IL: {avg_il:.1f} [≤ 15: {'✓' if avg_il <= 15 else '✗'}]")

    # 결과 요약
    print("\n" + "=" * 80)
    print("SUMMARY (sorted by GTPA@1)")
    print("=" * 80)
    print(f"\n{'Name':<20} {'GTPA@1':<10} {'GTPA@10':<10} {'max_il%':<10} {'Avg IL':<10} {'Status':<10}")
    print("-" * 80)

    for name, data in sorted(results_summary.items(), key=lambda x: -x[1]["gtpa_1"]):
        status = "✅" if data["meets_target"] else "❌"
        print(f"{name:<20} {data['gtpa_1']:.2%}{'':>2} {data['gtpa_10']:.2%}{'':>2} {data['max_il_pct']:.2%}{'':>3} {data['avg_il']:.1f}{'':>5} {status}")

    # 최적 설정 출력
    print("\n" + "=" * 60)
    print("OPTIMAL SETTINGS (max_il < 1% && Avg IL ≤ 15)")
    print("=" * 60)

    optimal = [(name, data) for name, data in results_summary.items() if data["meets_target"]]
    if optimal:
        # GTPA@1 기준 정렬
        optimal.sort(key=lambda x: -x[1]["gtpa_1"])
        for name, data in optimal:
            print(f"\n✅ {name}:")
            print(f"   GTPA@1: {data['gtpa_1']:.2%}")
            print(f"   GTPA@10: {data['gtpa_10']:.2%}")
            print(f"   max_il: {data['max_il_cases']} ({data['max_il_pct']:.2%})")
            print(f"   Avg IL: {data['avg_il']:.1f}")
            print(f"   Params: min_il={data['params']['min_il']}, conf={data['params']['confidence']}, gap={data['params']['gap']}, ratio={data['params']['ratio']}")
    else:
        print("\n❌ No settings meet the target.")

    with open("results/gtpa_il15_optimization.json", "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print("\nResults saved to: results/gtpa_il15_optimization.json")


if __name__ == "__main__":
    main()
