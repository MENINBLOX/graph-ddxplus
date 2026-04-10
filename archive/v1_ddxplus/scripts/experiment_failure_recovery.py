#!/usr/bin/env python3
"""실패 케이스 복구 실험.

339개 실패 케이스를 대상으로 다양한 전략 테스트.
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


def run_diagnosis_with_strategy(args: tuple) -> dict | None:
    """전략별 진단 수행."""
    patient_idx, patient_data, loader_data, neo4j_port, strategy = args

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

        # 전략별 파라미터
        limit = strategy.get("limit", 10)
        use_ig_initial = strategy.get("use_ig_initial", False)
        coverage_preference = strategy.get("coverage_preference", "high")  # high, mid, low

        for _ in range(50):
            # 전략별 후보 선택
            if use_ig_initial or il > 0:
                candidates = get_candidates_with_strategy(
                    kg, initial_cui, limit, coverage_preference
                )
            else:
                candidates = kg.get_candidate_symptoms(
                    initial_cui=initial_cui,
                    limit=limit,
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


def get_candidates_with_strategy(kg, initial_cui, limit, coverage_preference):
    """전략별 후보 증상 선택."""
    from src.umls_kg import SymptomCandidate

    _confirmed = kg.state.confirmed_cuis
    _denied = kg.state.denied_cuis
    _asked = kg.state.asked_cuis

    # Information Gain 기반 쿼리
    if coverage_preference == "mid":
        # 중간 coverage 우선 (감별력 최대화)
        query = """
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d

        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        WITH d, count(DISTINCT denied) AS denied_count
        WHERE denied_count < 5

        WITH collect(DISTINCT d) AS valid_diseases
        WITH valid_diseases, size(valid_diseases) AS total
        WHERE total > 0

        UNWIND valid_diseases AS d
        MATCH (d)<-[:INDICATES]-(next:Symptom)
        WHERE NOT next.cui IN $confirmed_cuis
          AND NOT next.cui IN $denied_cuis
          AND NOT next.cui IN $asked_cuis
        WITH next, total, count(DISTINCT d) AS coverage,
             CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority

        // 중간 coverage 우선 (Information Gain 최대)
        WITH next, coverage, total, priority,
             abs(toFloat(coverage) - toFloat(total) / 2.0) AS distance_from_mid

        RETURN next.cui AS cui,
               next.name AS name,
               coverage AS disease_coverage
        ORDER BY priority ASC, distance_from_mid ASC
        LIMIT $limit
        """
    elif coverage_preference == "low":
        # 낮은 coverage 우선 (특이 증상 우선)
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
    else:  # high (기본값)
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
        ORDER BY priority ASC, coverage DESC
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

    # 실패 케이스 로드
    with open("results/failure_analysis_gtpa10.json") as f:
        failure_data = json.load(f)

    failures = failure_data["failure_cases"]
    failure_indices = [f["patient_idx"] for f in failures]
    print(f"Total failure cases: {len(failure_indices)}")

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

    # 테스트할 전략들
    strategies = [
        {"name": "baseline", "limit": 10, "coverage_preference": "high"},
        {"name": "limit_30", "limit": 30, "coverage_preference": "high"},
        {"name": "limit_50", "limit": 50, "coverage_preference": "high"},
        {"name": "limit_100", "limit": 100, "coverage_preference": "high"},
        {"name": "mid_coverage", "limit": 10, "coverage_preference": "mid"},
        {"name": "mid_coverage_30", "limit": 30, "coverage_preference": "mid"},
        {"name": "low_coverage", "limit": 10, "coverage_preference": "low"},
        {"name": "low_coverage_30", "limit": 30, "coverage_preference": "low"},
    ]

    results_summary = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy['name']}")
        print(f"  limit={strategy['limit']}, coverage_preference={strategy['coverage_preference']}")
        print("=" * 60)

        tasks = [
            (idx, patients_data[idx], loader_data, NEO4J_PORTS[i % 8], strategy)
            for i, idx in enumerate(failure_indices)
        ]

        results = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(run_diagnosis_with_strategy, t): t[0] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Testing"):
                r = future.result()
                if r and "error" not in r:
                    results.append(r)

        # 집계
        recovered = sum(1 for r in results if r["correct_at_10"])
        recovery_rate = recovered / len(results) if results else 0

        # 질환별 복구율
        disease_recovery = {}
        for r in results:
            disease = r["gt_disease"]
            if disease not in disease_recovery:
                disease_recovery[disease] = {"total": 0, "recovered": 0}
            disease_recovery[disease]["total"] += 1
            if r["correct_at_10"]:
                disease_recovery[disease]["recovered"] += 1

        results_summary[strategy["name"]] = {
            "config": strategy,
            "total": len(results),
            "recovered": recovered,
            "recovery_rate": recovery_rate,
            "disease_recovery": disease_recovery,
        }

        print(f"\n  Recovered: {recovered}/{len(results)} ({recovery_rate:.1%})")
        print(f"\n  Top disease recovery:")
        for disease in ["Atrial fibrillation", "Viral pharyngitis", "Scombroid food poisoning"]:
            if disease in disease_recovery:
                dr = disease_recovery[disease]
                rate = dr["recovered"] / dr["total"] if dr["total"] else 0
                print(f"    {disease}: {dr['recovered']}/{dr['total']} ({rate:.1%})")

    # 결과 저장
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Strategy':<20} {'Recovered':<12} {'Rate':<10}")
    print("-" * 42)
    for name, data in sorted(results_summary.items(), key=lambda x: -x[1]["recovery_rate"]):
        print(f"{name:<20} {data['recovered']:>4}/{data['total']:<6} {data['recovery_rate']:.1%}")

    with open("results/failure_recovery_experiment.json", "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print("\nResults saved to: results/failure_recovery_experiment.json")


if __name__ == "__main__":
    main()
