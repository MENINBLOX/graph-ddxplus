#!/usr/bin/env python3
"""Threshold validation sweep on full validation set.

Runs threshold 1-8 with fixed settings (Top-3 Stability, Evidence Ratio)
on the full validation set to select optimal threshold WITHOUT test set snooping.

Usage:
    uv run python scripts/experiment_threshold_validation.py \
        --workers 8 --ports "7687,7688,7689,7690,7691,7692,7693,7694"
"""

import argparse
import json
import math
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

MAX_IL = 223
STOPPING = "top3_stable_5"
SCORING = "v15_ratio"
USE_ANTECEDENT = False


def score_v15_ratio(c, d, t):
    return (float(c) / (float(c) + float(d) + 1.0)) * float(c)


def get_candidates(kg, initial_cui, deny_threshold,
                   confirmed_cuis, denied_cuis, asked_cuis):
    _confirmed = confirmed_cuis
    _denied = denied_cuis
    _asked = asked_cuis

    if not _confirmed - {initial_cui}:
        query = """
        MATCH (s:Symptom {cui: $initial_cui})-[:INDICATES]->(d:Disease)
        MATCH (d)<-[:INDICATES]-(related:Symptom)
        WHERE related.cui <> $initial_cui
          AND NOT related.cui IN $asked_cuis
        WITH related, count(DISTINCT d) AS disease_coverage
        RETURN related.cui AS cui, related.name AS name, disease_coverage,
               0 AS priority
        ORDER BY disease_coverage DESC
        LIMIT 10
        """
        with kg.driver.session() as session:
            result = session.run(query, initial_cui=initial_cui, asked_cuis=list(_asked))
            from src.umls_kg import SymptomCandidate
            return [SymptomCandidate(cui=r["cui"], name=r["name"],
                                     disease_coverage=r["disease_coverage"]) for r in result]

    deny_filter = f"WHERE denied_count < {deny_threshold}" if deny_threshold > 0 else ""

    query = f"""
    MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
    WHERE confirmed.cui IN $confirmed_cuis
    WITH DISTINCT d
    OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
    WHERE denied.cui IN $denied_cuis
    WITH d, count(DISTINCT denied) AS denied_count
    {deny_filter}
    WITH collect(DISTINCT d) AS valid_diseases
    WHERE size(valid_diseases) > 0
    UNWIND valid_diseases AS d
    MATCH (d)<-[:INDICATES]-(next:Symptom)
    WHERE NOT next.cui IN $confirmed_cuis
      AND NOT next.cui IN $denied_cuis
      AND NOT next.cui IN $asked_cuis
    WITH next, d
    MATCH (d)<-[:INDICATES]-(conf:Symptom)
    WHERE conf.cui IN $confirmed_cuis
    WITH next, count(DISTINCT d) AS coverage, count(DISTINCT conf) AS cooccur_count,
         0 AS priority
    RETURN next.cui AS cui, next.name AS name, coverage AS disease_coverage, priority
    ORDER BY toFloat(cooccur_count) * coverage DESC
    LIMIT 10
    """
    with kg.driver.session() as session:
        result = session.run(query,
                             confirmed_cuis=list(_confirmed),
                             denied_cuis=list(_denied),
                             asked_cuis=list(_asked))
        from src.umls_kg import SymptomCandidate
        return [SymptomCandidate(cui=r["cui"], name=r["name"],
                                 disease_coverage=r["disease_coverage"]) for r in result]


def check_stopping_top3_stable(rank_history):
    if len(rank_history) >= 5:
        recent = list(rank_history)[-5:]
        return all(r == recent[0] for r in recent)
    return False


def get_custom_diagnosis(kg, confirmed_cuis, denied_cuis, top_k=10):
    query = """
    MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
    WHERE confirmed.cui IN $confirmed_cuis
    WITH DISTINCT d
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d,
         count(DISTINCT s) AS total_symptoms,
         count(DISTINCT CASE WHEN s.cui IN $confirmed_cuis THEN s END) AS confirmed_count,
         count(DISTINCT CASE WHEN s.cui IN $denied_cuis THEN s END) AS denied_count
    WHERE confirmed_count > 0
    RETURN d.cui AS cui, d.name AS name,
           confirmed_count, denied_count, total_symptoms
    """
    with kg.driver.session() as session:
        result = session.run(query,
                             confirmed_cuis=list(confirmed_cuis),
                             denied_cuis=list(denied_cuis))
        candidates = []
        for r in result:
            raw = score_v15_ratio(r["confirmed_count"], r["denied_count"], r["total_symptoms"])
            candidates.append({"cui": r["cui"], "name": r["name"], "score": raw,
                               "confirmed_count": r["confirmed_count"],
                               "total_symptoms": r["total_symptoms"]})

    total = sum(c["score"] for c in candidates)
    for c in candidates:
        c["score"] = c["score"] / total if total > 0 else 0
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def run_single_patient(args):
    patient_data, loader_data, deny_threshold, neo4j_port = args

    from src.data_loader import DDXPlusLoader, Patient
    from src.umls_kg import UMLSKG

    loader = DDXPlusLoader()
    loader._symptom_mapping = loader_data["symptom_mapping"]
    loader._disease_mapping = loader_data["disease_mapping"]
    loader._fr_to_eng = loader_data["fr_to_eng"]
    loader._conditions = loader_data["conditions"]

    try:
        kg = UMLSKG(uri=f"bolt://localhost:{neo4j_port}")
    except Exception:
        return {"error": True}

    try:
        patient = Patient(
            age=patient_data["age"], sex=patient_data["sex"],
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
            return {"error": True}

        kg.reset_state()
        kg.state.add_confirmed(initial_cui)

        il = 0
        confirmed_count = 1
        rank_history = deque(maxlen=10)

        for _ in range(MAX_IL):
            candidates = get_candidates(
                kg, initial_cui, deny_threshold,
                kg.state.confirmed_cuis, kg.state.denied_cuis, kg.state.asked_cuis,
            )
            if not candidates:
                break

            selected_cui = candidates[0].cui

            hit = 1 if selected_cui in patient_positive_cuis else 0
            if hit:
                kg.state.add_confirmed(selected_cui)
                confirmed_count += 1
            else:
                kg.state.add_denied(selected_cui)
            il += 1

            kg_diag = kg.get_diagnosis_candidates(top_k=10)
            kg_dist = [(c.cui, c.score) for c in kg_diag] if kg_diag else []
            current_ranks = tuple(cui for cui, _ in kg_dist[:3])
            rank_history.append(current_ranks)

            if check_stopping_top3_stable(rank_history):
                break

        final = get_custom_diagnosis(kg, kg.state.confirmed_cuis, kg.state.denied_cuis, top_k=10)
        correct_at_1 = final[0]["cui"] == gt_cui if final else False

        kg.close()
        return {
            "error": False,
            "correct_at_1": int(correct_at_1),
            "il": il,
            "confirmed": confirmed_count,
        }
    except Exception:
        kg.close()
        return {"error": True}


def run_threshold(threshold, patients_data, loader_data, ports, workers):
    tasks = [(pd, loader_data, threshold, ports[i % len(ports)])
             for i, pd in enumerate(patients_data)]

    results, errors = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_single_patient, t): i for i, t in enumerate(tasks)}
        desc = f"threshold={threshold}"
        with tqdm(total=len(futures), desc=desc, leave=True) as pbar:
            for f in as_completed(futures):
                r = f.result()
                if r and not r.get("error"):
                    results.append(r)
                else:
                    errors += 1
                pbar.update(1)

    count = len(results)
    if count == 0:
        return None

    gtpa_1 = sum(r["correct_at_1"] for r in results) / count
    avg_il = float(np.mean([r["il"] for r in results]))
    avg_confirmed = float(np.mean([r["confirmed"] for r in results]))

    return {
        "deny_threshold": threshold,
        "count": count,
        "errors": errors,
        "gtpa_1": gtpa_1,
        "avg_il": avg_il,
        "avg_confirmed": avg_confirmed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ports", type=str, default="7687,7688,7689,7690,7691,7692,7693,7694")
    parser.add_argument("--thresholds", type=str, default="1,2,3,4,5,6,7,8")
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",")]
    thresholds = [int(t.strip()) for t in args.thresholds.split(",")]

    from src.data_loader import DDXPlusLoader
    loader = DDXPlusLoader()
    patients = loader.load_patients(split="validate")
    print(f"Validation set: {len(patients):,} cases")

    loader_data = {
        "symptom_mapping": loader.symptom_mapping,
        "disease_mapping": loader.disease_mapping,
        "fr_to_eng": loader.fr_to_eng,
        "conditions": {k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
                       for k, v in loader.conditions.items()},
    }
    patients_data = [
        {"age": p.age, "sex": p.sex, "initial_evidence": p.initial_evidence,
         "evidences": p.evidences, "pathology": p.pathology,
         "differential_diagnosis": p.differential_diagnosis}
        for p in patients
    ]

    all_results = []
    total_start = time.time()

    for threshold in thresholds:
        start = time.time()
        result = run_threshold(threshold, patients_data, loader_data, ports, args.workers)
        elapsed = time.time() - start

        if result:
            result["elapsed"] = elapsed
            all_results.append(result)
            print(f"  threshold={threshold}: GTPA@1={result['gtpa_1']:.2%}, "
                  f"Avg IL={result['avg_il']:.1f}, "
                  f"Confirmed={result['avg_confirmed']:.1f} ({elapsed/60:.1f}min)")

    total_elapsed = time.time() - total_start

    # Best threshold
    best = max(all_results, key=lambda x: x["gtpa_1"])
    print(f"\n{'='*60}")
    print(f"BEST THRESHOLD: {best['deny_threshold']} (GTPA@1={best['gtpa_1']:.2%})")
    print(f"Total time: {total_elapsed/60:.1f} min")
    print(f"{'='*60}")

    output = {
        "experiment": "threshold_validation_sweep",
        "split": "validate",
        "total_cases": len(patients),
        "stopping": STOPPING,
        "scoring": SCORING,
        "antecedent": USE_ANTECEDENT,
        "best_threshold": best["deny_threshold"],
        "best_gtpa_1": best["gtpa_1"],
        "results": all_results,
        "total_elapsed": total_elapsed,
    }

    path = Path("results") / "threshold_validation_sweep.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
