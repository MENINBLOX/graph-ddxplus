#!/usr/bin/env python3
"""Per-disease GTPA@1 failure analysis for the optimal configuration.

Runs deny5_noante + top3_stable_5 + v15_ratio on all 134,529 test cases,
recording per-patient pathology and correctness, then aggregates by disease.

Usage:
    python scripts/analyze_disease_failure.py \
        --workers 8 --ports "7687,7688,7689,7690,7691,7692,7693,7694"
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

MAX_IL = 223

# Fixed optimal config
DENY_THRESHOLD = 5
USE_ANTECEDENT = False
STOPPING = "top3_stable_5"
SCORING = "v15_ratio"


def score_v15_ratio(c, d, t):
    return (float(c) / (float(c) + float(d) + 1.0)) * float(c)


def get_candidates(kg, initial_cui, confirmed_cuis, denied_cuis, asked_cuis):
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
            return [SymptomCandidate(cui=r["cui"], name=r["name"], disease_coverage=r["disease_coverage"]) for r in result]

    query = f"""
    MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
    WHERE confirmed.cui IN $confirmed_cuis
    WITH DISTINCT d
    OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
    WHERE denied.cui IN $denied_cuis
    WITH d, count(DISTINCT denied) AS denied_count
    WHERE denied_count < {DENY_THRESHOLD}
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
        return [SymptomCandidate(cui=r["cui"], name=r["name"], disease_coverage=r["disease_coverage"]) for r in result]


def check_stopping(il, step_hits, confirmed_count, rank_history, kg_dist):
    # top3_stable_5 only
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
                               "confirmed_count": r["confirmed_count"], "total_symptoms": r["total_symptoms"]})

    total = sum(c["score"] for c in candidates)
    for c in candidates:
        c["score"] = c["score"] / total if total > 0 else 0
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def run_single_patient(args):
    patient_data, loader_data, neo4j_port = args

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
        step_hits = []
        rank_history = deque(maxlen=10)

        for _ in range(MAX_IL):
            candidates = get_candidates(
                kg, initial_cui,
                kg.state.confirmed_cuis, kg.state.denied_cuis, kg.state.asked_cuis,
            )
            if not candidates:
                break

            selected_cui = candidates[0].cui  # greedy

            hit = 1 if selected_cui in patient_positive_cuis else 0
            if hit:
                kg.state.add_confirmed(selected_cui)
                confirmed_count += 1
            else:
                kg.state.add_denied(selected_cui)
            il += 1
            step_hits.append(hit)

            kg_diag = kg.get_diagnosis_candidates(top_k=10)
            kg_dist = [(c.cui, c.score) for c in kg_diag] if kg_diag else []
            current_ranks = tuple(cui for cui, _ in kg_dist[:3])
            rank_history.append(current_ranks)

            if check_stopping(il, step_hits, confirmed_count, rank_history, kg_dist):
                break

        final = get_custom_diagnosis(kg, kg.state.confirmed_cuis, kg.state.denied_cuis, top_k=10)
        correct_at_1 = final[0]["cui"] == gt_cui if final else False
        correct_at_10 = any(c["cui"] == gt_cui for c in final[:10]) if final else False

        kg.close()
        return {
            "error": False,
            "disease": gt_disease_eng,
            "correct_at_1": int(correct_at_1),
            "correct_at_10": int(correct_at_10),
            "il": il,
            "confirmed": confirmed_count,
        }
    except Exception:
        kg.close()
        return {"error": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ports", type=str, default="7687,7688,7689,7690,7691,7692,7693,7694")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples (default: all)")
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",")]

    from src.data_loader import DDXPlusLoader
    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test")

    if args.n_samples:
        patients = patients[:args.n_samples]

    print(f"=== Disease Failure Analysis ({len(patients):,} cases) ===")
    print(f"Config: deny5_noante + top3_stable_5 + v15_ratio")
    print(f"Workers: {args.workers}, Ports: {ports}")

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

    tasks = [(pd, loader_data, ports[i % len(ports)])
             for i, pd in enumerate(patients_data)]

    start = time.time()
    results = []
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_single_patient, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc="Disease analysis") as pbar:
            for f in as_completed(futures):
                r = f.result()
                if r and not r.get("error"):
                    results.append(r)
                else:
                    errors += 1
                pbar.update(1)

    elapsed = time.time() - start

    # Aggregate by disease
    disease_stats = defaultdict(lambda: {"count": 0, "correct_1": 0, "correct_10": 0,
                                          "il_sum": 0, "ils": []})
    for r in results:
        d = r["disease"]
        disease_stats[d]["count"] += 1
        disease_stats[d]["correct_1"] += r["correct_at_1"]
        disease_stats[d]["correct_10"] += r["correct_at_10"]
        disease_stats[d]["il_sum"] += r["il"]
        disease_stats[d]["ils"].append(r["il"])

    # Build per-disease table
    per_disease = []
    for disease, stats in disease_stats.items():
        cnt = stats["count"]
        per_disease.append({
            "disease": disease,
            "count": cnt,
            "gtpa_1": stats["correct_1"] / cnt if cnt else 0,
            "gtpa_10": stats["correct_10"] / cnt if cnt else 0,
            "avg_il": stats["il_sum"] / cnt if cnt else 0,
            "il_std": float(np.std(stats["ils"])) if cnt > 1 else 0.0,
            "correct_1": stats["correct_1"],
            "correct_10": stats["correct_10"],
        })

    # Sort by GTPA@1 ascending (worst first)
    per_disease.sort(key=lambda x: (x["gtpa_1"], x["disease"]))

    # Print results
    total_count = len(results)
    overall_gtpa1 = sum(r["correct_at_1"] for r in results) / total_count if total_count else 0
    overall_gtpa10 = sum(r["correct_at_10"] for r in results) / total_count if total_count else 0
    overall_avg_il = np.mean([r["il"] for r in results]) if results else 0

    print(f"\n{'='*90}")
    print(f"OVERALL: GTPA@1={overall_gtpa1:.4f}, GTPA@10={overall_gtpa10:.4f}, "
          f"Avg IL={overall_avg_il:.1f}, N={total_count:,}, Errors={errors}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"{'='*90}")

    print(f"\n{'Rank':<5} {'Disease':<45} {'Count':>6} {'GTPA@1':>8} {'GTPA@10':>8} {'Avg IL':>7}")
    print("-" * 90)
    for i, d in enumerate(per_disease, 1):
        print(f"{i:<5} {d['disease']:<45} {d['count']:>6} {d['gtpa_1']:>8.4f} {d['gtpa_10']:>8.4f} {d['avg_il']:>7.1f}")

    print(f"\n--- WORST 5 ---")
    for d in per_disease[:5]:
        print(f"  {d['disease']}: GTPA@1={d['gtpa_1']:.4f}, count={d['count']}, avg_il={d['avg_il']:.1f}")

    print(f"\n--- BEST 5 ---")
    for d in per_disease[-5:]:
        print(f"  {d['disease']}: GTPA@1={d['gtpa_1']:.4f}, count={d['count']}, avg_il={d['avg_il']:.1f}")

    # Save
    output = {
        "config": {
            "deny_threshold": DENY_THRESHOLD,
            "antecedent": USE_ANTECEDENT,
            "stopping": STOPPING,
            "scoring": SCORING,
            "method": "deny5_noante+top3_stable_5+v15_ratio",
        },
        "overall": {
            "count": total_count,
            "errors": errors,
            "gtpa_1": overall_gtpa1,
            "gtpa_10": overall_gtpa10,
            "avg_il": float(overall_avg_il),
            "elapsed": elapsed,
        },
        "per_disease": per_disease,
    }

    # Remove raw il lists from output (not JSON-friendly for large data)
    out_path = Path("results") / "disease_failure_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
