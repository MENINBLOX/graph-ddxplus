#!/usr/bin/env python3
"""IL 분포 분석 - 특히 높은 IL 케이스 분석."""

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
    """진단 수행 및 IL 수집."""
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
    except Exception:
        return None

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
        stop_reason = ""

        for _ in range(50):
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

            should_stop, reason = kg.should_stop(
                max_il=50,
                min_il=10,
                confidence_threshold=0.30,
                gap_threshold=0.04,
                relative_gap_threshold=1.5,
            )
            if should_stop:
                stop_reason = reason
                break

        diagnosis = kg.get_diagnosis_candidates(top_k=10)
        predicted_cuis = [d.cui for d in diagnosis]
        correct = gt_cui in predicted_cuis if gt_cui else False

        kg.close()

        return {
            "idx": patient_idx,
            "il": il,
            "stop_reason": stop_reason,
            "confirmed": confirmed_count,
            "total_symptoms": len(patient_positive_cuis),
            "correct": correct,
            "disease": gt_disease_eng,
        }

    except Exception:
        kg.close()
        return None


def main():
    from src.data_loader import DDXPlusLoader

    print("=" * 70)
    print("IL Distribution Analysis")
    print("=" * 70)

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

    tasks = [
        (i, patients_data[i], loader_data, NEO4J_PORTS[i % 8])
        for i in range(len(patients_data))
    ]

    results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_diagnosis, t): t[0] for t in tasks}

        with tqdm(total=len(tasks), desc="Analyzing") as pbar:
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)
                pbar.update(1)

    # IL 분포 분석
    il_values = [r["il"] for r in results]
    il_counter = Counter(il_values)

    print("\n" + "=" * 70)
    print("IL DISTRIBUTION")
    print("=" * 70)

    print("\n1. IL Histogram (top 20):")
    for il, count in sorted(il_counter.items(), key=lambda x: -x[1])[:20]:
        pct = count / len(results) * 100
        bar = "█" * int(pct)
        print(f"   IL={il:2d}: {count:6d} ({pct:5.2f}%) {bar}")

    # 최대 IL 케이스
    max_il = max(il_values)
    max_il_cases = [r for r in results if r["il"] == max_il]

    print(f"\n2. Maximum IL = {max_il}")
    print(f"   Cases with max IL: {len(max_il_cases)}")

    # 높은 IL 케이스 분석 (IL >= 40)
    high_il_cases = [r for r in results if r["il"] >= 40]
    print(f"\n3. High IL Cases (IL >= 40): {len(high_il_cases)}")

    if high_il_cases:
        # 종료 사유 분포
        stop_reasons = Counter(r["stop_reason"] for r in high_il_cases)
        print("\n   Stop Reason Distribution:")
        for reason, count in stop_reasons.most_common():
            print(f"     - {reason}: {count}")

        # 질환 분포
        diseases = Counter(r["disease"] for r in high_il_cases)
        print("\n   Disease Distribution (top 10):")
        for disease, count in diseases.most_common(10):
            print(f"     - {disease}: {count}")

        # 증상 수
        avg_symptoms = sum(r["total_symptoms"] for r in high_il_cases) / len(high_il_cases)
        avg_confirmed = sum(r["confirmed"] for r in high_il_cases) / len(high_il_cases)
        print(f"\n   Avg total symptoms: {avg_symptoms:.1f}")
        print(f"   Avg confirmed: {avg_confirmed:.1f}")

        # 정확도
        accuracy = sum(1 for r in high_il_cases if r["correct"]) / len(high_il_cases)
        print(f"   Accuracy (GTPA@10): {accuracy:.2%}")

    # IL=50 케이스 상세
    il_50_cases = [r for r in results if r["il"] == 50]
    if il_50_cases:
        print(f"\n4. IL=50 Cases (max_il reached): {len(il_50_cases)}")
        print("\n   Sample cases:")
        for i, c in enumerate(il_50_cases[:5]):
            print(f"\n   Case {i+1} (idx={c['idx']}):")
            print(f"     Disease: {c['disease']}")
            print(f"     Stop reason: {c['stop_reason']}")
            print(f"     Total symptoms: {c['total_symptoms']}")
            print(f"     Confirmed: {c['confirmed']}")
            print(f"     Correct: {c['correct']}")

    # 결과 저장
    output = {
        "summary": {
            "total": len(results),
            "max_il": max_il,
            "max_il_cases": len(max_il_cases),
            "high_il_cases": len(high_il_cases),
            "avg_il": sum(il_values) / len(il_values),
        },
        "il_distribution": dict(il_counter),
        "high_il_cases": high_il_cases[:100],
    }

    with open("results/il_distribution_analysis.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\n\nResults saved to: results/il_distribution_analysis.json")


if __name__ == "__main__":
    main()
