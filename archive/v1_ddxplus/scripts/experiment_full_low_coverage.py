#!/usr/bin/env python3
"""전체 데이터셋에서 low_coverage 전략 검증.

목표: low_coverage 전략이 기존 성공 케이스에 regression을 일으키지 않는지 확인.
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


def run_diagnosis_with_strategy(args: tuple) -> dict | None:
    """전략별 진단 수행."""
    patient_idx, patient_data, loader_data, neo4j_port, use_low_coverage = args

    from src.data_loader import DDXPlusLoader, Patient
    from src.umls_kg import UMLSKG, SymptomCandidate

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
            # 전략별 후보 선택
            if use_low_coverage:
                candidates = get_low_coverage_candidates(kg, initial_cui, 10)
            else:
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


def get_low_coverage_candidates(kg, initial_cui, limit):
    """Low coverage 전략: 특이 증상 우선."""
    from src.umls_kg import SymptomCandidate

    _confirmed = kg.state.confirmed_cuis
    _denied = kg.state.denied_cuis
    _asked = kg.state.asked_cuis

    # IL=0이면 initial candidates 로직 사용
    if len(_confirmed) <= 1 and len(_denied) == 0:
        query = """
        MATCH (s:Symptom {cui: $initial_cui})-[:INDICATES]->(d:Disease)
        MATCH (d)<-[:INDICATES]-(related:Symptom)
        WHERE related.cui <> $initial_cui
          AND NOT related.cui IN $asked_cuis
        WITH related, count(DISTINCT d) AS disease_coverage
        RETURN related.cui AS cui,
               related.name AS name,
               disease_coverage,
               CASE WHEN related.is_antecedent = false THEN 0 ELSE 1 END AS priority
        ORDER BY priority ASC, disease_coverage ASC
        LIMIT $limit
        """
        with kg.driver.session() as session:
            result = session.run(
                query,
                initial_cui=initial_cui,
                asked_cuis=list(_asked),
                limit=limit,
            )
            return [
                SymptomCandidate(
                    cui=r["cui"],
                    name=r["name"],
                    disease_coverage=r["disease_coverage"],
                )
                for r in result
            ]
    else:
        # Accumulated candidates
        query = """
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d

        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        WITH d, count(DISTINCT denied) AS denied_count
        WHERE denied_count < 5

        WITH collect(DISTINCT d) AS valid_diseases

        UNWIND valid_diseases AS d
        MATCH (d)<-[:INDICATES]-(next:Symptom)
        WHERE NOT next.cui IN $confirmed_cuis
          AND NOT next.cui IN $denied_cuis
          AND NOT next.cui IN $asked_cuis
        WITH next, count(DISTINCT d) AS coverage,
             CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority

        RETURN next.cui AS cui,
               next.name AS name,
               coverage AS disease_coverage
        ORDER BY priority ASC, coverage ASC
        LIMIT $limit
        """
        with kg.driver.session() as session:
            result = session.run(
                query,
                confirmed_cuis=list(_confirmed),
                denied_cuis=list(_denied),
                asked_cuis=list(_asked),
                limit=limit,
            )
            return [
                SymptomCandidate(
                    cui=r["cui"],
                    name=r["name"],
                    disease_coverage=r["disease_coverage"],
                )
                for r in result
            ]


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

    # 테스트할 전략
    strategies = [
        {"name": "baseline", "use_low_coverage": False},
        {"name": "low_coverage", "use_low_coverage": True},
    ]

    results_summary = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy['name']}")
        print("=" * 60)

        start_time = time.time()

        tasks = [
            (idx, patients_data[idx], loader_data, NEO4J_PORTS[idx % 8], strategy["use_low_coverage"])
            for idx in range(len(patients_data))
        ]

        results = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(run_diagnosis_with_strategy, t): t[0] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Testing"):
                r = future.result()
                if r and "error" not in r:
                    results.append(r)

        elapsed = time.time() - start_time

        # 집계
        correct_at_1 = sum(1 for r in results if r["correct_at_1"])
        correct_at_10 = sum(1 for r in results if r["correct_at_10"])
        avg_il = sum(r["il"] for r in results) / len(results) if results else 0
        avg_confirmed = sum(r["confirmed"] for r in results) / len(results) if results else 0

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

        results_summary[strategy["name"]] = {
            "config": strategy,
            "total": len(results),
            "correct_at_1": correct_at_1,
            "correct_at_10": correct_at_10,
            "gtpa_1": gtpa_1,
            "gtpa_10": gtpa_10,
            "avg_il": avg_il,
            "avg_confirmed": avg_confirmed,
            "elapsed_sec": elapsed,
            "disease_stats": disease_stats,
        }

        print(f"\n  Total: {len(results)}")
        print(f"  GTPA@1: {correct_at_1}/{len(results)} ({gtpa_1:.2%})")
        print(f"  GTPA@10: {correct_at_10}/{len(results)} ({gtpa_10:.2%})")
        print(f"  Avg IL: {avg_il:.1f}")
        print(f"  Avg Confirmed: {avg_confirmed:.1f}")
        print(f"  Time: {elapsed:.1f}s")

    # 비교 요약
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    baseline = results_summary["baseline"]
    low_cov = results_summary["low_coverage"]

    print(f"\n{'Metric':<20} {'Baseline':<15} {'Low Coverage':<15} {'Delta':<15}")
    print("-" * 65)
    print(f"{'GTPA@1':<20} {baseline['gtpa_1']:.2%}{'':>6} {low_cov['gtpa_1']:.2%}{'':>6} {(low_cov['gtpa_1'] - baseline['gtpa_1'])*100:+.2f}pp")
    print(f"{'GTPA@10':<20} {baseline['gtpa_10']:.2%}{'':>6} {low_cov['gtpa_10']:.2%}{'':>6} {(low_cov['gtpa_10'] - baseline['gtpa_10'])*100:+.2f}pp")
    print(f"{'Avg IL':<20} {baseline['avg_il']:.1f}{'':>10} {low_cov['avg_il']:.1f}{'':>10} {low_cov['avg_il'] - baseline['avg_il']:+.1f}")

    # 결과 저장
    with open("results/full_low_coverage_experiment.json", "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print("\nResults saved to: results/full_low_coverage_experiment.json")


if __name__ == "__main__":
    main()
