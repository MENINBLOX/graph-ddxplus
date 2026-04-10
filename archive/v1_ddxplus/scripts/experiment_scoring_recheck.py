#!/usr/bin/env python3
"""NB/BM25/LLR scoring functions 재검증.

기존 구현의 기술적 오류를 수정하고 134,529건에서 재실행.
설정: Threshold=6, Top-3 Stability, Antecedent=No, Greedy, Co-occurrence=Yes

Usage:
    uv run python scripts/experiment_scoring_recheck.py \
        --workers 48 --ports "7687,7688,7689,7690,7691,7692,7693,7694"
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
USE_ANTECEDENT = False


# ============================================================
# 수정된 scoring 함수들
# ============================================================

def score_naive_bayes_v2(c, d, t):
    """Naive Bayes (수정): 이진 KG에서의 근사.

    가정:
    - P(symptom present | disease has symptom) = alpha = 0.8
    - P(symptom present | disease lacks symptom) = beta = 0.1
    - Uniform prior P(d) = 1/49

    Score = c * log(alpha/beta) + d * log((1-alpha)/(1-beta))
          = c * log(8) + d * log(2/9)
    확인 증상은 양의 기여, 부정 증상은 음의 기여 (클리핑 없음).
    """
    c, d = float(c), float(d)
    alpha, beta = 0.8, 0.1
    log_pos = math.log(alpha / beta)       # ~2.08
    log_neg = math.log((1 - alpha) / (1 - beta))  # ~-1.50
    return c * log_pos + d * log_neg


def score_bm25_v2(c, d, t, k1=1.2, b=0.75, avgdl=18.1):
    """BM25 (수정): 의료 진단 적응.

    수정사항:
    - k1=1.2 (표준값), b=0.75 (표준값)
    - avgdl=18.1 (DDXPlus 질환당 평균 증상 수)
    - IDF: log((N - n + 0.5) / (n + 0.5)) where N=49, n=t (질환의 총 증상 수)
      → 증상이 적은(특이적인) 질환에 높은 가중치
    - denied 증상은 별도의 negative TF로 처리
    """
    c, d, t = float(c), float(d), float(t)
    if c == 0:
        return 0.0
    dl = t
    # TF normalization (confirmed)
    tf_pos = (c * (k1 + 1)) / (c + k1 * (1 - b + b * dl / avgdl))
    # IDF: 증상이 적은 질환 = 더 특이적 = 높은 가중치
    # n = t (이 질환의 총 증상 수), N = 49 (전체 질환 수)
    # 증상이 많은 질환은 일반적 → 낮은 IDF
    idf = math.log(1.0 + (49.0 - t + 0.5) / (t + 0.5))
    # Denied penalty: confirmed에서 denied 비율만큼 감산
    denied_ratio = d / (c + d) if (c + d) > 0 else 0
    return tf_pos * idf * (1 - 0.5 * denied_ratio)


def score_log_likelihood_v2(c, d, t):
    """Log-Likelihood Ratio (수정): 클리핑 제거.

    LLR = Σ log(P(evidence|disease) / P(evidence|background))
    이진 KG 근사:
    - P(confirmed_s | d) = c/t (질환 증상 중 확인 비율)
    - P(confirmed_s | bg) = c/223 (전체 증상 중 확인 비율)
    - P(denied_s | d) = d/t
    - P(denied_s | bg) = d/223
    클리핑 없이 양수/음수 모두 유지.
    """
    c, d, t = float(c), float(d), float(t)
    if c == 0 or t == 0:
        return -999.0  # 증거 없음 → 매우 낮은 점수
    eps = 1e-10
    # Positive evidence ratio
    p_c_disease = c / (t + eps)
    p_c_bg = c / 223.0
    # Negative evidence ratio
    p_d_disease = d / (t + eps) if d > 0 else eps
    p_d_bg = d / 223.0 if d > 0 else eps
    # LLR (no clipping)
    llr = c * math.log((p_c_disease + eps) / (p_c_bg + eps))
    if d > 0:
        llr += d * math.log((p_d_disease + eps) / (p_d_bg + eps))
    return llr


# 기존 함수들 (비교용)
def score_v15_ratio(c, d, t):
    return (float(c) / (float(c) + float(d) + 1.0)) * float(c)


def score_idf_only(c, d, t):
    """IDF-only: BM25에서 TF/length를 제거하고 IDF만 사용.

    IDF(disease) = log((N - t + 0.5) / (t + 0.5)), N=49
    → 증상이 적은(특이적인) 질환에 높은 가중치
    Score = c × IDF - penalty × d
    """
    c, d, t = float(c), float(d), float(t)
    if c == 0:
        return 0.0
    # IDF: 질환의 총 증상 수가 적을수록(특이적) 높은 값
    idf = math.log(1.0 + (49.0 - t + 0.5) / (t + 0.5))
    # denied penalty: 부정 증상 비율만큼 감산
    denied_penalty = d / (c + d) if (c + d) > 0 else 0
    return c * idf * (1.0 - 0.3 * denied_penalty)


def score_noisy_or(c, d, t):
    """Noisy-OR 모델 (Shwe et al. 1991, Rotmensch et al. 2017).

    P(symptom present | disease) = 1 - (1-leak) × (1-p_cause)^{linked}
    이진 KG 근사:
    - p_cause = 0.7 (질환이 연결된 증상을 유발할 확률)
    - leak = 0.05 (배경 노이즈)
    - 확인 증상: log P(s=1|d) 기여
    - 부정 증상: log P(s=0|d) 기여 (클리핑 없음)

    Score = Σ_confirmed log(p_cause) + Σ_denied log(1-p_cause)
    + log prior (uniform → 무시)
    """
    c, d = float(c), float(d)
    p_cause = 0.7
    leak = 0.05

    # 확인 증상: 질환이 이 증상을 가짐 → P(s=1|d) = 1-(1-leak)×(1-p_cause) = p_cause + leak - p_cause×leak
    p_present = 1.0 - (1.0 - leak) * (1.0 - p_cause)  # ≈ 0.715
    # 부정 증상: 질환이 이 증상을 가지지만 환자에게 없음 → P(s=0|d) = 1 - p_present
    p_absent = 1.0 - p_present  # ≈ 0.285

    # 배경 확률 (질환 무관)
    p_bg_present = leak  # 0.05
    p_bg_absent = 1.0 - leak  # 0.95

    # Log-likelihood ratio 형태
    if c == 0:
        return d * math.log(p_absent / p_bg_absent)

    score = c * math.log(p_present / p_bg_present)  # 확인 → 양의 기여
    score += d * math.log(p_absent / p_bg_absent)    # 부정 → 음의 기여
    return score


SCORING_FUNCTIONS = {
    "naive_bayes_v2": score_naive_bayes_v2,
    "bm25_v2": score_bm25_v2,
    "log_likelihood_v2": score_log_likelihood_v2,
    "idf_only": score_idf_only,
    "noisy_or": score_noisy_or,
}


# ============================================================
# KG 탐색 (experiment_final_75.py와 동일)
# ============================================================

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
            return [SymptomCandidate(cui=r["cui"], name=r["name"],
                                     disease_coverage=r["disease_coverage"]) for r in result]

    deny_filter = f"WHERE denied_count < {DENY_THRESHOLD}"

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


def get_custom_diagnosis(kg, scoring_fn, confirmed_cuis, denied_cuis, top_k=10):
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
            candidates.append({"cui": r["cui"], "name": r["name"], "score": raw,
                               "confirmed_count": r["confirmed_count"],
                               "total_symptoms": r["total_symptoms"]})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def run_single_patient(args):
    patient_data, loader_data, scoring_name, neo4j_port = args

    from src.data_loader import DDXPlusLoader, Patient
    from src.umls_kg import UMLSKG

    scoring_fn = SCORING_FUNCTIONS[scoring_name]

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
                kg, initial_cui,
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

            # Top-3 Stability stopping (v15_ratio for stopping判定)
            kg_diag = kg.get_diagnosis_candidates(top_k=10)
            kg_dist = [(c.cui, c.score) for c in kg_diag] if kg_diag else []
            current_ranks = tuple(cui for cui, _ in kg_dist[:3])
            rank_history.append(current_ranks)

            if len(rank_history) >= 5:
                recent = list(rank_history)[-5:]
                if all(r == recent[0] for r in recent):
                    break

        # 최종 진단: 수정된 scoring 함수 사용
        final = get_custom_diagnosis(kg, scoring_fn,
                                      kg.state.confirmed_cuis, kg.state.denied_cuis, top_k=10)
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


def run_scoring(scoring_name, patients_data, loader_data, ports, workers):
    tasks = [(pd, loader_data, scoring_name, ports[i % len(ports)])
             for i, pd in enumerate(patients_data)]

    start = time.time()
    results, errors = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_single_patient, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc=scoring_name, leave=True) as pbar:
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
        return None

    return {
        "scoring": scoring_name,
        "count": count,
        "errors": errors,
        "gtpa_1": sum(r["correct_at_1"] for r in results) / count,
        "avg_il": float(np.mean([r["il"] for r in results])),
        "avg_confirmed": float(np.mean([r["confirmed"] for r in results])),
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ports", type=str, default="7687,7688,7689,7690,7691,7692,7693,7694")
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",")]

    from src.data_loader import DDXPlusLoader
    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test")
    print(f"Test set: {len(patients):,} cases")

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
    for scoring_name in SCORING_FUNCTIONS:
        print(f"\n=== {scoring_name} ===")
        result = run_scoring(scoring_name, patients_data, loader_data, ports, args.workers)
        if result:
            all_results.append(result)
            print(f"  GTPA@1: {result['gtpa_1']:.2%}, Avg IL: {result['avg_il']:.1f}")

    print(f"\n{'='*60}")
    print("SUMMARY (Threshold=6, Top-3 Stability)")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['scoring']:25s}: GTPA@1={r['gtpa_1']:.2%}, Avg IL={r['avg_il']:.1f}")
    print(f"  {'Evidence Ratio (paper)':25s}: GTPA@1=91.05%, Avg IL=23.1")

    path = Path("results") / "scoring_recheck.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
