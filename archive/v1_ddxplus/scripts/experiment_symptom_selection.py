#!/usr/bin/env python3
"""증상 선택 알고리즘 최적화 실험.

목표: 환자가 가진 증상을 더 빨리 찾아서 confirmed 증가, IL 감소.
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


# 새로운 증상 선택 쿼리들
QUERY_STRATEGIES = {
    # 현재 방식: IG 기반
    "baseline_ig": """
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

        WITH next, coverage, total, priority,
             abs(toFloat(coverage) - toFloat(total) / 2.0) AS distance_from_optimal
        WITH next, coverage, priority,
             CASE WHEN total > 0
                  THEN 1.0 - (distance_from_optimal / (toFloat(total) / 2.0 + 0.1))
                  ELSE 0.0 END AS ig_score

        RETURN next.cui AS cui,
               next.name AS name,
               coverage AS disease_coverage,
               ig_score AS score
        ORDER BY priority ASC, ig_score DESC
        LIMIT $limit
    """,

    # Co-occurrence 기반: confirmed 증상과 함께 자주 나타나는 증상 우선
    "cooccurrence": """
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d

        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        WITH d, count(DISTINCT denied) AS denied_count
        WHERE denied_count < 5

        WITH collect(DISTINCT d) AS valid_diseases
        WHERE size(valid_diseases) > 0

        UNWIND valid_diseases AS d
        MATCH (d)<-[:INDICATES]-(next:Symptom)
        WHERE NOT next.cui IN $confirmed_cuis
          AND NOT next.cui IN $denied_cuis
          AND NOT next.cui IN $asked_cuis

        // Co-occurrence score: 얼마나 많은 confirmed와 동일 질환을 공유하는지
        WITH next, d
        MATCH (d)<-[:INDICATES]-(conf:Symptom)
        WHERE conf.cui IN $confirmed_cuis
        WITH next, count(DISTINCT d) AS coverage, count(DISTINCT conf) AS cooccur_count,
             CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority

        // cooccur_count가 높을수록 confirmed와 함께 나타날 확률 높음
        RETURN next.cui AS cui,
               next.name AS name,
               coverage AS disease_coverage,
               toFloat(cooccur_count) * coverage AS score
        ORDER BY priority ASC, score DESC
        LIMIT $limit
    """,

    # Weighted coverage: confirmed가 많은 질환의 증상 우선
    "weighted_coverage": """
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH d, count(DISTINCT confirmed) AS match_count

        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        WITH d, match_count, count(DISTINCT denied) AS denied_count
        WHERE denied_count < 5

        WITH collect({disease: d, weight: match_count}) AS weighted_diseases
        WHERE size(weighted_diseases) > 0

        UNWIND weighted_diseases AS wd
        WITH wd.disease AS d, wd.weight AS weight
        MATCH (d)<-[:INDICATES]-(next:Symptom)
        WHERE NOT next.cui IN $confirmed_cuis
          AND NOT next.cui IN $denied_cuis
          AND NOT next.cui IN $asked_cuis

        WITH next, sum(weight) AS weighted_score, count(DISTINCT d) AS coverage,
             CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority

        RETURN next.cui AS cui,
               next.name AS name,
               coverage AS disease_coverage,
               weighted_score AS score
        ORDER BY priority ASC, weighted_score DESC
        LIMIT $limit
    """,

    # Hybrid: Coverage + Co-occurrence
    "hybrid": """
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH d, count(DISTINCT confirmed) AS conf_count

        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        WITH d, conf_count, count(DISTINCT denied) AS denied_count
        WHERE denied_count < 5

        WITH collect({disease: d, conf_count: conf_count}) AS valid_diseases
        WITH valid_diseases, size(valid_diseases) AS total
        WHERE total > 0

        UNWIND valid_diseases AS vd
        WITH vd.disease AS d, vd.conf_count AS conf_count, total
        MATCH (d)<-[:INDICATES]-(next:Symptom)
        WHERE NOT next.cui IN $confirmed_cuis
          AND NOT next.cui IN $denied_cuis
          AND NOT next.cui IN $asked_cuis

        WITH next, total,
             count(DISTINCT d) AS coverage,
             sum(conf_count) AS weighted_cooccur,
             CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority

        // Hybrid score: IG 요소 + co-occurrence 가중치
        WITH next, coverage, total, weighted_cooccur, priority,
             abs(toFloat(coverage) - toFloat(total) / 2.0) AS distance_from_mid
        WITH next, coverage, priority,
             CASE WHEN total > 0
                  THEN (1.0 - distance_from_mid / (toFloat(total) / 2.0 + 0.1)) * 0.3 +
                       (toFloat(weighted_cooccur) / (toFloat(coverage) + 0.1)) * 0.7
                  ELSE 0.0 END AS hybrid_score

        RETURN next.cui AS cui,
               next.name AS name,
               coverage AS disease_coverage,
               hybrid_score AS score
        ORDER BY priority ASC, hybrid_score DESC
        LIMIT $limit
    """,

    # High coverage (현재 _get_initial_candidates 방식)
    "high_coverage": """
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
               coverage AS disease_coverage,
               coverage AS score
        ORDER BY priority ASC, coverage DESC
        LIMIT $limit
    """,
}


def run_diagnosis_with_query(args: tuple) -> dict | None:
    """쿼리별 진단."""
    patient_idx, patient_data, loader_data, neo4j_port, query_name = args

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
        denied_count = 0
        max_il = 50

        for _ in range(max_il):
            # 커스텀 쿼리로 후보 가져오기
            candidates = get_candidates_with_query(
                kg, query_name, initial_cui, 10
            )

            if not candidates:
                break

            selected = candidates[0]

            if selected.cui in patient_positive_cuis:
                kg.state.add_confirmed(selected.cui)
                confirmed_count += 1
            else:
                kg.state.add_denied(selected.cui)
                denied_count += 1

            il += 1

            should_stop, _ = kg.should_stop(max_il=max_il, min_il=10)
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
            "denied": denied_count,
            "gt_disease": gt_disease_eng,
        }

    except Exception as e:
        kg.close()
        return {"error": str(e), "patient_idx": patient_idx}


def get_candidates_with_query(kg, query_name, initial_cui, limit):
    """쿼리별 후보 가져오기."""
    from src.umls_kg import SymptomCandidate

    _confirmed = kg.state.confirmed_cuis
    _denied = kg.state.denied_cuis
    _asked = kg.state.asked_cuis

    # IL=0이면 initial candidates (기존 로직 유지)
    if len(_confirmed) <= 1 and len(_denied) == 0:
        return kg._get_initial_candidates(initial_cui, limit, asked_cuis=_asked)

    query = QUERY_STRATEGIES[query_name]

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

    strategies = list(QUERY_STRATEGIES.keys())
    results_summary = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print("=" * 60)

        start_time = time.time()

        tasks = [
            (idx, patients_data[idx], loader_data, NEO4J_PORTS[idx % 8], strategy)
            for idx in range(len(patients_data))
        ]

        results = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(run_diagnosis_with_query, t): t[0] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Testing"):
                r = future.result()
                if r and "error" not in r:
                    results.append(r)

        elapsed = time.time() - start_time

        correct_at_1 = sum(1 for r in results if r["correct_at_1"])
        correct_at_10 = sum(1 for r in results if r["correct_at_10"])
        avg_il = sum(r["il"] for r in results) / len(results) if results else 0
        avg_confirmed = sum(r["confirmed"] for r in results) / len(results) if results else 0
        avg_denied = sum(r["denied"] for r in results) / len(results) if results else 0
        confirm_rate = avg_confirmed / (avg_confirmed + avg_denied) if (avg_confirmed + avg_denied) > 0 else 0

        results_summary[strategy] = {
            "total": len(results),
            "correct_at_1": correct_at_1,
            "correct_at_10": correct_at_10,
            "gtpa_1": correct_at_1 / len(results),
            "gtpa_10": correct_at_10 / len(results),
            "avg_il": avg_il,
            "avg_confirmed": avg_confirmed,
            "avg_denied": avg_denied,
            "confirm_rate": confirm_rate,
            "elapsed_sec": elapsed,
        }

        print(f"\n  GTPA@1: {correct_at_1/len(results):.2%}")
        print(f"  GTPA@10: {correct_at_10/len(results):.2%}")
        print(f"  Avg IL: {avg_il:.1f}")
        print(f"  Avg Confirmed: {avg_confirmed:.1f}")
        print(f"  Confirm Rate: {confirm_rate:.2%}")

    # 비교 요약
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Strategy':<20} {'GTPA@1':<10} {'GTPA@10':<10} {'Avg IL':<10} {'Confirm%':<10}")
    print("-" * 60)

    for name, data in sorted(results_summary.items(), key=lambda x: -x[1]["gtpa_10"]):
        print(f"{name:<20} {data['gtpa_1']:.2%}{'':>2} {data['gtpa_10']:.2%}{'':>2} {data['avg_il']:.1f}{'':>5} {data['confirm_rate']:.1%}")

    with open("results/symptom_selection_experiment.json", "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print("\nResults saved to: results/symptom_selection_experiment.json")


if __name__ == "__main__":
    main()
