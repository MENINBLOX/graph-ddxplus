#!/usr/bin/env python3
"""적응형 전략 실험.

연속 거부가 발생하면 low-coverage로 전환.
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


def run_diagnosis_adaptive(args: tuple) -> dict | None:
    """적응형 전략 진단."""
    patient_idx, patient_data, loader_data, neo4j_port, switch_threshold = args

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
        consecutive_denials = 0
        use_low_coverage = False
        switched_at = None

        for _ in range(50):
            # 적응형 전략: 연속 거부 시 low-coverage로 전환
            if consecutive_denials >= switch_threshold and not use_low_coverage:
                use_low_coverage = True
                switched_at = il

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
                consecutive_denials = 0  # 리셋
            else:
                kg.state.add_denied(selected.cui)
                consecutive_denials += 1

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
            "switched": switched_at is not None,
            "switched_at": switched_at,
        }

    except Exception as e:
        kg.close()
        return {"error": str(e), "patient_idx": patient_idx}


def get_low_coverage_candidates(kg, initial_cui, limit):
    """Low coverage 전략."""
    from src.umls_kg import SymptomCandidate

    _confirmed = kg.state.confirmed_cuis
    _denied = kg.state.denied_cuis
    _asked = kg.state.asked_cuis

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

    # 테스트할 threshold 값들
    thresholds = [3, 5, 7, 10]
    results_summary = {}

    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Adaptive Strategy: switch after {threshold} consecutive denials")
        print("=" * 60)

        start_time = time.time()

        tasks = [
            (idx, patients_data[idx], loader_data, NEO4J_PORTS[idx % 8], threshold)
            for idx in range(len(patients_data))
        ]

        results = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(run_diagnosis_adaptive, t): t[0] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Testing"):
                r = future.result()
                if r and "error" not in r:
                    results.append(r)

        elapsed = time.time() - start_time

        correct_at_1 = sum(1 for r in results if r["correct_at_1"])
        correct_at_10 = sum(1 for r in results if r["correct_at_10"])
        switched_count = sum(1 for r in results if r["switched"])
        avg_il = sum(r["il"] for r in results) / len(results) if results else 0

        gtpa_1 = correct_at_1 / len(results) if results else 0
        gtpa_10 = correct_at_10 / len(results) if results else 0

        results_summary[f"threshold_{threshold}"] = {
            "threshold": threshold,
            "total": len(results),
            "correct_at_1": correct_at_1,
            "correct_at_10": correct_at_10,
            "gtpa_1": gtpa_1,
            "gtpa_10": gtpa_10,
            "avg_il": avg_il,
            "switched_count": switched_count,
            "switch_rate": switched_count / len(results),
            "elapsed_sec": elapsed,
        }

        print(f"\n  Total: {len(results)}")
        print(f"  GTPA@1: {correct_at_1}/{len(results)} ({gtpa_1:.2%})")
        print(f"  GTPA@10: {correct_at_10}/{len(results)} ({gtpa_10:.2%})")
        print(f"  Switched: {switched_count} ({switched_count/len(results):.2%})")
        print(f"  Avg IL: {avg_il:.1f}")

    # 비교
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"\n{'Threshold':<12} {'GTPA@1':<12} {'GTPA@10':<12} {'Switched':<12}")
    print("-" * 50)
    for name, data in results_summary.items():
        print(f"{data['threshold']:<12} {data['gtpa_1']:.2%}{'':>4} {data['gtpa_10']:.2%}{'':>4} {data['switch_rate']:.2%}")

    with open("results/adaptive_strategy_experiment.json", "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print("\nResults saved to: results/adaptive_strategy_experiment.json")


if __name__ == "__main__":
    main()
