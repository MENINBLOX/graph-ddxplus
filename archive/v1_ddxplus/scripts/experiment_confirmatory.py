#!/usr/bin/env python3
"""확인 실험: ANOVA에서 고정한 3개 요인의 GTPA@1 검증.

최적 조합(deny6, noante, greedy, cooccur=yes, top3_stable_5, v15_ratio)에서
각 고정 요인의 대안 1개씩 테스트:
  A. antecedent=Yes (vs No)
  B. selection=ig_binary_split (vs greedy)
  C. cooccur=No (vs Yes)

사용법:
  # 100건 빠른 테스트
  uv run python scripts/experiment_confirmatory.py --n-samples 100

  # 전체 134,529건
  uv run python scripts/experiment_confirmatory.py \
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
DENY_THRESHOLD = 6
STOPPING = "top3_stable_5"
SCORING = "v15_ratio"


# ============================================================
# 스코어링
# ============================================================

def score_v15_ratio(c, d, t):
    return (float(c) / (float(c) + float(d) + 1.0)) * float(c)

SCORING_FUNCTIONS = {"v15_ratio": score_v15_ratio}


# ============================================================
# 후보 생성 쿼리 (3가지 변형)
# ============================================================

def get_candidates_default(kg, initial_cui, confirmed, denied, asked):
    """기본: greedy + cooccur=Yes + antecedent=No"""
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
                             confirmed_cuis=list(confirmed),
                             denied_cuis=list(denied),
                             asked_cuis=list(asked))
        from src.umls_kg import SymptomCandidate
        return [SymptomCandidate(cui=r["cui"], name=r["name"], disease_coverage=r["disease_coverage"]) for r in result]


def get_candidates_ante_yes(kg, initial_cui, confirmed, denied, asked):
    """변형A: antecedent=Yes"""
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
         CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority
    RETURN next.cui AS cui, next.name AS name, coverage AS disease_coverage, priority
    ORDER BY priority ASC, toFloat(cooccur_count) * coverage DESC
    LIMIT 10
    """
    with kg.driver.session() as session:
        result = session.run(query,
                             confirmed_cuis=list(confirmed),
                             denied_cuis=list(denied),
                             asked_cuis=list(asked))
        from src.umls_kg import SymptomCandidate
        return [SymptomCandidate(cui=r["cui"], name=r["name"], disease_coverage=r["disease_coverage"]) for r in result]


def get_candidates_no_cooccur(kg, initial_cui, confirmed, denied, asked):
    """변형C: cooccur=No (coverage만 사용)"""
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
    WITH next, count(DISTINCT d) AS coverage,
         0 AS priority
    RETURN next.cui AS cui, next.name AS name, coverage AS disease_coverage, priority
    ORDER BY coverage DESC
    LIMIT 10
    """
    with kg.driver.session() as session:
        result = session.run(query,
                             confirmed_cuis=list(confirmed),
                             denied_cuis=list(denied),
                             asked_cuis=list(asked))
        from src.umls_kg import SymptomCandidate
        return [SymptomCandidate(cui=r["cui"], name=r["name"], disease_coverage=r["disease_coverage"]) for r in result]


# ============================================================
# 선택 전략
# ============================================================

def select_greedy(candidates):
    return candidates[0].cui if candidates else None


def select_ig_binary_split(candidates, kg, confirmed, denied):
    """Information Gain: binary split 기준으로 가장 균등하게 나누는 증상 선택."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0].cui

    best_cui = candidates[0].cui
    best_score = float('inf')

    for cand in candidates[:5]:  # 상위 5개만 평가
        # 이 증상이 yes일 때와 no일 때 후보 질환 수 차이
        query = """
        MATCH (s:Symptom {cui: $cui})-[:INDICATES]->(d:Disease)
        WHERE d.cui IN $disease_cuis
        RETURN count(DISTINCT d) AS yes_count
        """
        # 현재 후보 질환 목록
        disease_query = f"""
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d
        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        WITH d, count(DISTINCT denied) AS denied_count
        WHERE denied_count < {DENY_THRESHOLD}
        RETURN d.cui AS cui
        """
        with kg.driver.session() as session:
            diseases = [r["cui"] for r in session.run(disease_query,
                                                       confirmed_cuis=list(confirmed),
                                                       denied_cuis=list(denied))]
            if not diseases:
                continue
            result = session.run(query, cui=cand.cui, disease_cuis=diseases)
            yes_count = result.single()["yes_count"]
            no_count = len(diseases) - yes_count
            # 균등 분할에 가까울수록 좋음
            score = abs(yes_count - no_count)
            if score < best_score:
                best_score = score
                best_cui = cand.cui

    return best_cui


# ============================================================
# 종료 조건 + 진단
# ============================================================

def check_stopping(il, rank_history):
    if len(rank_history) >= 5:
        recent = list(rank_history)[-5:]
        return all(r == recent[0] for r in recent)
    return False


def get_diagnosis(kg, confirmed_cuis, denied_cuis, top_k=10):
    scoring_fn = score_v15_ratio
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
            raw = scoring_fn(r["confirmed_count"], r["denied_count"], r["total_symptoms"])
            candidates.append({"cui": r["cui"], "name": r["name"], "score": raw})

    total = sum(c["score"] for c in candidates)
    for c in candidates:
        c["score"] = c["score"] / total if total > 0 else 0
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


# ============================================================
# 환자 실행
# ============================================================

def run_single_patient(args):
    patient_data, loader_data, variant, neo4j_port = args

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

        # 후보 생성 함수 선택
        if variant == "ante_yes":
            get_cand_fn = get_candidates_ante_yes
        elif variant == "no_cooccur":
            get_cand_fn = get_candidates_no_cooccur
        else:
            get_cand_fn = get_candidates_default

        use_ig_split = (variant == "ig_binary_split")

        il = 0
        confirmed_count = 1
        rank_history = deque(maxlen=10)

        for _ in range(MAX_IL):
            candidates = get_cand_fn(
                kg, initial_cui,
                kg.state.confirmed_cuis, kg.state.denied_cuis, kg.state.asked_cuis,
            )
            if not candidates:
                break

            # 선택 전략
            if use_ig_split:
                selected_cui = select_ig_binary_split(
                    candidates, kg, kg.state.confirmed_cuis, kg.state.denied_cuis)
            else:
                selected_cui = select_greedy(candidates)

            if not selected_cui:
                break

            hit = 1 if selected_cui in patient_positive_cuis else 0
            if hit:
                kg.state.add_confirmed(selected_cui)
                confirmed_count += 1
            else:
                kg.state.add_denied(selected_cui)
            il += 1

            # 종료 판단
            kg_diag = get_diagnosis(kg, kg.state.confirmed_cuis, kg.state.denied_cuis, top_k=10)
            kg_dist = [(c["cui"], c["score"]) for c in kg_diag] if kg_diag else []
            current_ranks = tuple(cui for cui, _ in kg_dist[:3])
            rank_history.append(current_ranks)

            if check_stopping(il, rank_history):
                break

        # 최종 진단
        final = get_diagnosis(kg, kg.state.confirmed_cuis, kg.state.denied_cuis, top_k=10)
        correct_at_1 = final[0]["cui"] == gt_cui if final else False
        correct_at_10 = any(c["cui"] == gt_cui for c in final[:10]) if final else False

        kg.close()
        return {
            "error": False,
            "correct_at_1": int(correct_at_1),
            "correct_at_10": int(correct_at_10),
            "il": il,
            "confirmed": confirmed_count,
        }
    except Exception:
        kg.close()
        return {"error": True}


def run_variant(variant_name, patients_data, loader_data, ports, workers):
    tasks = [(pd, loader_data, variant_name, ports[i % len(ports)])
             for i, pd in enumerate(patients_data)]

    start = time.time()
    results, errors = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_single_patient, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc=f"confirm_{variant_name}") as pbar:
            for f in as_completed(futures):
                r = f.result()
                if r and not r.get("error"):
                    results.append(r)
                else:
                    errors += 1
                pbar.update(1)

    elapsed = time.time() - start
    count = len(results)
    if count == 0:
        return {"variant": variant_name, "error": True}

    return {
        "variant": variant_name,
        "count": count,
        "errors": errors,
        "gtpa_1": sum(r["correct_at_1"] for r in results) / count,
        "gtpa_10": sum(r["correct_at_10"] for r in results) / count,
        "avg_il": float(np.mean([r["il"] for r in results])),
        "avg_confirmed": float(np.mean([r["confirmed"] for r in results])),
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ports", type=str, default="7687,7688,7689,7690")
    parser.add_argument("--n-samples", type=int, default=0,
                        help="0=전체, N=상위 N건만")
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",")]

    from src.data_loader import DDXPlusLoader
    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test")

    if args.n_samples > 0:
        patients = patients[:args.n_samples]

    print(f"=== Confirmatory Experiments ({len(patients):,}건) ===")
    print(f"Base: deny{DENY_THRESHOLD}, noante, greedy, cooccur=yes, {STOPPING}, {SCORING}")
    print()

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

    variants = ["ante_yes", "ig_binary_split", "no_cooccur"]
    all_results = {}

    for v in variants:
        print(f"\n--- Variant: {v} ---")
        result = run_variant(v, patients_data, loader_data, ports, args.workers)
        all_results[v] = result
        if not result.get("error"):
            print(f"  GTPA@1: {result['gtpa_1']:.2%}, Avg IL: {result['avg_il']:.1f}")

    # 저장
    n_tag = args.n_samples if args.n_samples > 0 else len(patients)
    path = Path("results") / f"confirmatory_{n_tag}.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {path}")

    # 요약
    print("\n=== Summary ===")
    print(f"{'Variant':<20} {'GTPA@1':>8} {'Avg IL':>8}")
    print("-" * 40)
    print(f"{'baseline (default)':<20} {'91.05%':>8} {'23.1':>8}")
    for v, r in all_results.items():
        if not r.get("error"):
            print(f"{v:<20} {r['gtpa_1']:>7.2%} {r['avg_il']:>8.1f}")


if __name__ == "__main__":
    main()
