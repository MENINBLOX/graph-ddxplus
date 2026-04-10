#!/usr/bin/env python3
"""Simple baselines for DDXPlus benchmark.

Baseline 1: Symptom Overlap Lookup - uses only initial evidence
Baseline 2: Most Frequent Disease - always predicts most common disease
Baseline 3: Random Symptom Inquiry + Evidence Ratio scoring

Usage:
    uv run python scripts/baseline_simple.py --all
    uv run python scripts/baseline_simple.py --baseline 1
    uv run python scripts/baseline_simple.py --baseline 3 --workers 8 --ports "7687,7688"
"""

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Baseline 1: Symptom Overlap Lookup
# ============================================================

def run_baseline_overlap(loader, patients):
    """Use initial evidence CUI to find diseases via UMLS KG, rank by symptom overlap."""
    print("\n=== Baseline 1: Symptom Overlap Lookup ===")
    start = time.time()

    # Build disease CUI -> set of symptom CUIs from UMLS KG
    from src.umls_kg import UMLSKG
    kg = UMLSKG()

    # Get all disease -> symptom relationships from KG
    query = """
    MATCH (s:Symptom)-[:INDICATES]->(d:Disease)
    RETURN d.cui AS disease_cui, d.name AS disease_name,
           collect(s.cui) AS symptom_cuis
    """
    disease_symptom_cuis = {}
    with kg.driver.session() as session:
        for r in session.run(query):
            disease_symptom_cuis[r["disease_cui"]] = set(r["symptom_cuis"])

    # Build disease CUI -> fr name mapping
    cui_to_disease_fr = {}
    for name_eng, info in loader.disease_mapping.items():
        cui = info.get("umls_cui")
        cond = loader.conditions.get(name_eng)
        if cui and cond:
            cui_to_disease_fr[cui] = cond.name_fr

    kg.close()

    correct_1, correct_3, correct_5 = 0, 0, 0
    total = len(patients)

    for patient in tqdm(patients, desc="Baseline 1"):
        # Patient's positive symptom CUIs
        patient_cuis = set()
        for ev in patient.evidences:
            base = ev.split("_@_")[0]
            cui = loader.get_symptom_cui(base)
            if cui:
                patient_cuis.add(cui)

        # Initial evidence CUI
        init_cui = loader.get_symptom_cui(patient.initial_evidence)

        # Score: only diseases containing initial evidence as candidates,
        # ranked by number of total symptoms (fewer = more specific)
        scores = []
        for disease_cui, symp_cuis in disease_symptom_cuis.items():
            if disease_cui not in cui_to_disease_fr:
                continue
            has_initial = init_cui in symp_cuis if init_cui else False
            if not has_initial:
                continue
            # Initial-evidence-only: rank by inverse symptom count (more specific first)
            scores.append((cui_to_disease_fr[disease_cui], len(symp_cuis)))

        # Fewer symptoms = more specific disease = ranked first
        scores.sort(key=lambda x: x[1])
        top_diseases = [s[0] for s in scores[:5]]

        if patient.pathology in top_diseases[:1]:
            correct_1 += 1
        if patient.pathology in top_diseases[:3]:
            correct_3 += 1
        if patient.pathology in top_diseases[:5]:
            correct_5 += 1

    elapsed = time.time() - start

    result = {
        "baseline": "symptom_overlap_lookup",
        "description": "Rank diseases by symptom overlap with patient evidences, initial evidence priority",
        "count": total,
        "gtpa_1": correct_1 / total,
        "gtpa_3": correct_3 / total,
        "gtpa_5": correct_5 / total,
        "avg_il": 0,
        "elapsed": elapsed,
    }

    print(f"GTPA@1: {result['gtpa_1']:.2%}")
    print(f"GTPA@3: {result['gtpa_3']:.2%}")
    print(f"GTPA@5: {result['gtpa_5']:.2%}")
    print(f"Time: {elapsed:.1f}s")

    return result


# ============================================================
# Baseline 2: Most Frequent Disease
# ============================================================

def run_baseline_most_frequent(loader, patients):
    """Always predict the most frequent disease."""
    print("\n=== Baseline 2: Most Frequent Disease ===")
    start = time.time()

    # Count disease frequencies in test set
    counter = Counter(p.pathology for p in patients)
    most_common = counter.most_common(5)
    print(f"Top-5 diseases: {most_common}")

    most_common_pathology = most_common[0][0]
    top3_pathologies = {p for p, _ in most_common[:3]}
    top5_pathologies = {p for p, _ in most_common[:5]}

    total = len(patients)
    correct_1 = sum(1 for p in patients if p.pathology == most_common_pathology)
    correct_3 = sum(1 for p in patients if p.pathology in top3_pathologies)
    correct_5 = sum(1 for p in patients if p.pathology in top5_pathologies)

    elapsed = time.time() - start

    chance = 1.0 / 49  # uniform

    result = {
        "baseline": "most_frequent_disease",
        "description": f"Always predict '{most_common_pathology}' (most frequent in test set)",
        "most_common": most_common_pathology,
        "most_common_count": most_common[0][1],
        "count": total,
        "gtpa_1": correct_1 / total,
        "gtpa_3": correct_3 / total,
        "gtpa_5": correct_5 / total,
        "chance_level": chance,
        "avg_il": 0,
        "elapsed": elapsed,
    }

    print(f"GTPA@1: {result['gtpa_1']:.2%} (chance: {chance:.2%})")
    print(f"GTPA@3: {result['gtpa_3']:.2%}")
    print(f"GTPA@5: {result['gtpa_5']:.2%}")

    return result


# ============================================================
# Baseline 3: Random Inquiry + Evidence Ratio
# ============================================================

def score_v15_ratio(c, d, t):
    return (float(c) / (float(c) + float(d) + 1.0)) * float(c)


def get_random_candidate(kg, initial_cui, deny_threshold,
                         confirmed_cuis, denied_cuis, asked_cuis):
    """Get all available symptom candidates and return a random one."""
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
        RETURN related.cui AS cui, related.name AS name, disease_coverage
        """
        with kg.driver.session() as session:
            result = list(session.run(query, initial_cui=initial_cui,
                                      asked_cuis=list(_asked)))
            if not result:
                return None
            chosen = random.choice(result)
            from src.umls_kg import SymptomCandidate
            return SymptomCandidate(cui=chosen["cui"], name=chosen["name"],
                                    disease_coverage=chosen["disease_coverage"])

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
    WITH next, count(DISTINCT d) AS coverage
    RETURN next.cui AS cui, next.name AS name, coverage AS disease_coverage
    """
    with kg.driver.session() as session:
        result = list(session.run(query,
                                  confirmed_cuis=list(_confirmed),
                                  denied_cuis=list(_denied),
                                  asked_cuis=list(_asked)))
        if not result:
            return None
        chosen = random.choice(result)
        from src.umls_kg import SymptomCandidate
        return SymptomCandidate(cui=chosen["cui"], name=chosen["name"],
                                disease_coverage=chosen["disease_coverage"])


def get_diagnosis_random(kg, confirmed_cuis, denied_cuis, top_k=10):
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
            raw = score_v15_ratio(r["confirmed_count"], r["denied_count"],
                                  r["total_symptoms"])
            candidates.append({"cui": r["cui"], "name": r["name"], "score": raw})

    total = sum(c["score"] for c in candidates)
    for c in candidates:
        c["score"] = c["score"] / total if total > 0 else 0
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def run_single_random(args):
    patient_data, loader_data, deny_threshold, seed, neo4j_port = args

    from src.data_loader import DDXPlusLoader, Patient
    from src.umls_kg import UMLSKG

    random.seed(seed + hash(patient_data["initial_evidence"]) + patient_data["age"])

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

        for _ in range(223):
            candidate = get_random_candidate(
                kg, initial_cui, deny_threshold,
                kg.state.confirmed_cuis, kg.state.denied_cuis, kg.state.asked_cuis,
            )
            if not candidate:
                break

            hit = 1 if candidate.cui in patient_positive_cuis else 0
            if hit:
                kg.state.add_confirmed(candidate.cui)
                confirmed_count += 1
            else:
                kg.state.add_denied(candidate.cui)
            il += 1

            # Top-3 Stability stopping
            kg_diag = kg.get_diagnosis_candidates(top_k=10)
            kg_dist = [(c.cui, c.score) for c in kg_diag] if kg_diag else []
            current_ranks = tuple(cui for cui, _ in kg_dist[:3])
            rank_history.append(current_ranks)

            if len(rank_history) >= 5:
                recent = list(rank_history)[-5:]
                if all(r == recent[0] for r in recent):
                    break

        final = get_diagnosis_random(kg, kg.state.confirmed_cuis,
                                      kg.state.denied_cuis, top_k=10)
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


def run_baseline_random(loader, patients, ports, workers, deny_threshold=6, seed=42):
    """Random symptom inquiry with Evidence Ratio scoring."""
    print(f"\n=== Baseline 3: Random Inquiry + Evidence Ratio (seed={seed}) ===")

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

    tasks = [(pd, loader_data, deny_threshold, seed, ports[i % len(ports)])
             for i, pd in enumerate(patients_data)]

    start = time.time()
    results, errors = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_single_random, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc=f"Baseline 3 (seed={seed})") as pbar:
            for f in as_completed(futures):
                r = f.result()
                if r and not r.get("error"):
                    results.append(r)
                else:
                    errors += 1
                pbar.update(1)

    elapsed = time.time() - start
    count = len(results)

    result = {
        "baseline": "random_inquiry_evidence_ratio",
        "description": f"Random symptom selection + Evidence Ratio scoring (seed={seed})",
        "deny_threshold": deny_threshold,
        "seed": seed,
        "count": count,
        "errors": errors,
        "gtpa_1": sum(r["correct_at_1"] for r in results) / count if count else 0,
        "avg_il": float(np.mean([r["il"] for r in results])) if results else 0,
        "avg_confirmed": float(np.mean([r["confirmed"] for r in results])) if results else 0,
        "elapsed": elapsed,
    }

    print(f"GTPA@1: {result['gtpa_1']:.2%}")
    print(f"Avg IL: {result['avg_il']:.1f}")
    print(f"Time: {elapsed/60:.1f} min")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=int, choices=[1, 2, 3])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ports", type=str, default="7687,7688,7689,7690,7691,7692,7693,7694")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated seeds for baseline 3")
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    from src.data_loader import DDXPlusLoader
    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test")
    print(f"Test set: {len(patients):,} cases")

    all_results = []

    if args.all or args.baseline == 1:
        r = run_baseline_overlap(loader, patients)
        all_results.append(r)

    if args.all or args.baseline == 2:
        r = run_baseline_most_frequent(loader, patients)
        all_results.append(r)

    if args.all or args.baseline == 3:
        for seed in seeds:
            r = run_baseline_random(loader, patients, ports, args.workers, seed=seed)
            all_results.append(r)

    path = Path("results") / "baselines.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved: {path}")


if __name__ == "__main__":
    main()
