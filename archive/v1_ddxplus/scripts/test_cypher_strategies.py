"""Cypher 전략 비교 테스트.

DDF1 향상을 위한 다양한 스코어링 전략 테스트.
"""

import json
import random
from dataclasses import dataclass

from neo4j import GraphDatabase


@dataclass
class DiagnosisCandidate:
    cui: str
    name: str
    score: float


# Cypher 전략들
STRATEGIES = {
    "v4_coverage": """
    // v4: dcov × scov × (1 - den_ratio)²
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied,
         size($confirmed_cuis) AS total_confirmed
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed_count,
         size(matched_denied) AS denied_count,
         toFloat(size(matched_confirmed)) / (total_symptoms + 0.1) AS dcov,
         toFloat(size(matched_confirmed)) / (size($confirmed_cuis) + 0.1) AS scov
    WITH d, confirmed_count, total_symptoms,
         dcov, scov,
         CASE WHEN total_symptoms > 0 THEN toFloat(denied_count) / total_symptoms ELSE 0.0 END AS den_ratio
    WITH d, confirmed_count, total_symptoms,
         dcov * scov * (1.0 - den_ratio) * (1.0 - den_ratio) AS raw_score
    WHERE raw_score > 0
    WITH collect({cui: d.cui, name: d.name, raw_score: raw_score, confirmed_count: confirmed_count, total_symptoms: total_symptoms}) AS all_candidates
    WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count, c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,

    "v5_relaxed_denied": """
    // v5: dcov × scov × (1 - 0.3 × den_ratio) - 완화된 denied 페널티
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied,
         size($confirmed_cuis) AS total_confirmed
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed_count,
         size(matched_denied) AS denied_count,
         toFloat(size(matched_confirmed)) / (total_symptoms + 0.1) AS dcov,
         toFloat(size(matched_confirmed)) / (size($confirmed_cuis) + 0.1) AS scov
    WITH d, confirmed_count, total_symptoms,
         dcov, scov,
         CASE WHEN total_symptoms > 0 THEN toFloat(denied_count) / total_symptoms ELSE 0.0 END AS den_ratio
    WITH d, confirmed_count, total_symptoms,
         dcov * scov * (1.0 - 0.3 * den_ratio) AS raw_score
    WHERE raw_score > 0
    WITH collect({cui: d.cui, name: d.name, raw_score: raw_score, confirmed_count: confirmed_count, total_symptoms: total_symptoms}) AS all_candidates
    WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count, c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,

    "v6_confirmed_only": """
    // v6: dcov × scov - denied 페널티 없음
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         size($confirmed_cuis) AS total_confirmed
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed_count,
         toFloat(size(matched_confirmed)) / (total_symptoms + 0.1) AS dcov,
         toFloat(size(matched_confirmed)) / (size($confirmed_cuis) + 0.1) AS scov
    WITH d, confirmed_count, total_symptoms,
         dcov * scov AS raw_score
    WHERE raw_score > 0
    WITH collect({cui: d.cui, name: d.name, raw_score: raw_score, confirmed_count: confirmed_count, total_symptoms: total_symptoms}) AS all_candidates
    WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count, c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,

    "v7_additive": """
    // v7: confirmed_count - 0.5 × denied_count - 단순 가산 점수
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed_count,
         size(matched_denied) AS denied_count
    WITH d, confirmed_count, total_symptoms,
         toFloat(confirmed_count) - 0.5 * toFloat(denied_count) AS raw_score
    WHERE raw_score > 0
    WITH collect({cui: d.cui, name: d.name, raw_score: raw_score, confirmed_count: confirmed_count, total_symptoms: total_symptoms}) AS all_candidates
    WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count, c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,

    "v8_bayesian": """
    // v8: Bayesian-like - P(D|S) ∝ matched² / (total_symptoms × total_confirmed)
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied,
         size($confirmed_cuis) AS total_confirmed
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed_count,
         size(matched_denied) AS denied_count,
         size($confirmed_cuis) AS total_confirmed
    WITH d, confirmed_count, total_symptoms,
         // Bayesian score: matched² / (disease_symptoms × patient_symptoms) × (1 - denied_ratio)
         toFloat(confirmed_count * confirmed_count) / ((total_symptoms + 1.0) * (total_confirmed + 1.0)) *
         (1.0 - toFloat(denied_count) / (total_symptoms + 1.0)) AS raw_score
    WHERE raw_score > 0
    WITH collect({cui: d.cui, name: d.name, raw_score: raw_score, confirmed_count: confirmed_count, total_symptoms: total_symptoms}) AS all_candidates
    WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count, c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,

    "v9_log_likelihood": """
    // v9: Log likelihood ratio
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed_count,
         size(matched_denied) AS denied_count
    WITH d, confirmed_count, total_symptoms,
         // Log likelihood: log(confirmed + 1) - 0.5 × log(denied + 1)
         log(toFloat(confirmed_count) + 1.0) - 0.5 * log(toFloat(denied_count) + 1.0) AS raw_score
    WHERE raw_score > 0
    WITH collect({cui: d.cui, name: d.name, raw_score: raw_score, confirmed_count: confirmed_count, total_symptoms: total_symptoms}) AS all_candidates
    WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count, c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,

    "v10_dcov_only": """
    // v10: Disease coverage만 사용 (질환 증상 중 매칭 비율)
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed_count,
         size(matched_denied) AS denied_count
    WITH d, confirmed_count, total_symptoms,
         // dcov × (1 - den_ratio)
         (toFloat(confirmed_count) / (total_symptoms + 0.1)) *
         (1.0 - toFloat(denied_count) / (total_symptoms + 0.1)) AS raw_score
    WHERE raw_score > 0
    WITH collect({cui: d.cui, name: d.name, raw_score: raw_score, confirmed_count: confirmed_count, total_symptoms: total_symptoms}) AS all_candidates
    WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count, c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,

    "v11_high_recall": """
    // v11: High recall - 매칭 1개 이상이면 모두 포함, 약한 페널티
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed_count,
         size(matched_denied) AS denied_count
    WITH d, confirmed_count, total_symptoms,
         // 매칭 수 기반 + 매우 약한 denied 페널티
         toFloat(confirmed_count) * (1.0 - 0.1 * toFloat(denied_count)) AS raw_score
    WHERE raw_score > 0
    WITH collect({cui: d.cui, name: d.name, raw_score: raw_score, confirmed_count: confirmed_count, total_symptoms: total_symptoms}) AS all_candidates
    WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count, c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,

    "v12_sqrt_match": """
    // v12: sqrt(matched) - 매칭 많을수록 점진적 증가
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed_count,
         size(matched_denied) AS denied_count
    WITH d, confirmed_count, total_symptoms,
         // sqrt(matched) × (1 - 0.2 × denied/total)
         sqrt(toFloat(confirmed_count)) *
         (1.0 - 0.2 * toFloat(denied_count) / (total_symptoms + 1.0)) AS raw_score
    WHERE raw_score > 0
    WITH collect({cui: d.cui, name: d.name, raw_score: raw_score, confirmed_count: confirmed_count, total_symptoms: total_symptoms}) AS all_candidates
    WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count, c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,

    "v13_balanced": """
    // v13: Balanced - dcov^0.5 × scov^0.5 × penalty
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied,
         size($confirmed_cuis) AS total_confirmed
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed_count,
         size(matched_denied) AS denied_count,
         total_confirmed,
         toFloat(size(matched_confirmed)) / (total_symptoms + 0.1) AS dcov,
         toFloat(size(matched_confirmed)) / (total_confirmed + 0.1) AS scov
    WITH d, confirmed_count, total_symptoms,
         // sqrt(dcov) × sqrt(scov) × (1 - 0.3 × denied_ratio)
         sqrt(dcov) * sqrt(scov) *
         (1.0 - 0.3 * toFloat(denied_count) / (total_symptoms + 1.0)) AS raw_score
    WHERE raw_score > 0
    WITH collect({cui: d.cui, name: d.name, raw_score: raw_score, confirmed_count: confirmed_count, total_symptoms: total_symptoms}) AS all_candidates
    WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count, c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,
}


def load_patients(n: int = 300) -> list[dict]:
    """테스트 환자 로드."""
    import pandas as pd

    df = pd.read_csv("data/ddxplus/release_test_patients.csv", nrows=5000)
    # 랜덤 샘플링
    random.seed(42)
    indices = random.sample(range(len(df)), min(n, len(df)))

    patients = []
    for idx in indices:
        row = df.iloc[idx]
        patients.append({
            "initial_evidence": row["INITIAL_EVIDENCE"],
            "evidences": eval(row["EVIDENCES"]) if isinstance(row["EVIDENCES"], str) else [],
            "pathology": row["PATHOLOGY"],
            "differential": eval(row["DIFFERENTIAL_DIAGNOSIS"]) if isinstance(row["DIFFERENTIAL_DIAGNOSIS"], str) else [],
        })
    return patients


def load_mappings() -> tuple[dict, dict, dict]:
    """매핑 데이터 로드."""
    with open("data/ddxplus/umls_mapping.json") as f:
        symptom_mapping = json.load(f)["mapping"]

    with open("data/ddxplus/disease_umls_mapping.json") as f:
        raw_disease_mapping = json.load(f)["mapping"]

    # French name -> CUI 매핑 생성
    disease_mapping = {}
    cui_to_disease = {}
    for key, info in raw_disease_mapping.items():
        name_fr = info.get("name_fr")
        cui = info.get("umls_cui")
        if name_fr and cui:
            disease_mapping[name_fr] = {"cui": cui, "name": info.get("umls_name", key)}
            cui_to_disease[cui] = name_fr

    return symptom_mapping, disease_mapping, cui_to_disease


def get_patient_symptoms(patient: dict, symptom_mapping: dict) -> tuple[set, set]:
    """환자의 confirmed/denied 증상 CUI 추출."""
    confirmed = set()
    denied = set()

    # Initial evidence
    initial = patient["initial_evidence"]
    if initial in symptom_mapping:
        confirmed.add(symptom_mapping[initial]["cui"])

    # All evidences (simulating full knowledge)
    for ev in patient["evidences"]:
        base_code = ev.split("_@_")[0]
        if base_code in symptom_mapping:
            confirmed.add(symptom_mapping[base_code]["cui"])

    return confirmed, denied


def get_gt_dd(patient: dict, disease_mapping: dict) -> set:
    """Ground truth differential diagnosis CUI set."""
    gt_dd = set()
    for diag, prob in patient["differential"]:
        if diag in disease_mapping:
            cui = disease_mapping[diag].get("cui")
            if cui:
                gt_dd.add(cui)
    return gt_dd


def run_strategy(
    driver,
    strategy_name: str,
    query: str,
    confirmed_cuis: list,
    denied_cuis: list,
    top_k: int = 15,
) -> list[DiagnosisCandidate]:
    """전략 실행."""
    with driver.session() as session:
        try:
            result = session.run(
                query,
                confirmed_cuis=confirmed_cuis,
                denied_cuis=denied_cuis,
                top_k=top_k,
            )
            return [
                DiagnosisCandidate(
                    cui=r["cui"],
                    name=r["name"],
                    score=r["score"],
                )
                for r in result
            ]
        except Exception as e:
            print(f"  Error in {strategy_name}: {e}")
            return []


def calculate_dd_metrics(
    predicted_cuis: set,
    gt_cuis: set,
) -> tuple[float, float, float]:
    """DDR, DDP, DDF1 계산."""
    if not gt_cuis:
        return 0.0, 0.0, 0.0

    intersection = predicted_cuis & gt_cuis

    ddr = len(intersection) / len(gt_cuis) if gt_cuis else 0.0
    ddp = len(intersection) / len(predicted_cuis) if predicted_cuis else 0.0
    ddf1 = 2 * ddr * ddp / (ddr + ddp) if (ddr + ddp) > 0 else 0.0

    return ddr, ddp, ddf1


def test_top_k_impact(driver, patients, symptom_mapping, disease_mapping):
    """Test impact of different top_k values."""
    print("\n" + "=" * 60)
    print("Top-K Impact Test (using v7_additive)")
    print("=" * 60)

    query = STRATEGIES["v7_additive"]
    top_k_values = [5, 8, 10, 12, 15, 20, 25, 30]

    for top_k in top_k_values:
        ddr_list = []
        ddp_list = []
        ddf1_list = []

        for patient in patients:
            confirmed_cuis, denied_cuis = get_patient_symptoms(patient, symptom_mapping)
            gt_dd_cuis = get_gt_dd(patient, disease_mapping)

            if not confirmed_cuis or not gt_dd_cuis:
                continue

            candidates = run_strategy(driver, "v7", query, list(confirmed_cuis), list(denied_cuis), top_k=top_k)
            if not candidates:
                continue

            predicted_cuis = {c.cui for c in candidates}
            ddr, ddp, ddf1 = calculate_dd_metrics(predicted_cuis, gt_dd_cuis)
            ddr_list.append(ddr)
            ddp_list.append(ddp)
            ddf1_list.append(ddf1)

        if ddr_list:
            avg_ddr = sum(ddr_list) / len(ddr_list) * 100
            avg_ddp = sum(ddp_list) / len(ddp_list) * 100
            avg_ddf1 = sum(ddf1_list) / len(ddf1_list) * 100
            print(f"  top_k={top_k:2d}: DDR={avg_ddr:5.1f}%, DDP={avg_ddp:5.1f}%, DDF1={avg_ddf1:5.1f}%")


def test_probability_cutoff(driver, patients, symptom_mapping, disease_mapping):
    """Test probability-based cutoff strategies."""
    print("\n" + "=" * 60)
    print("Probability Cutoff Test")
    print("=" * 60)

    query = STRATEGIES["v7_additive"]

    # 여러 cutoff 전략
    cutoff_strategies = [
        ("min_prob=0.02", lambda candidates: [c for c in candidates if c.score >= 0.02]),
        ("min_prob=0.025", lambda candidates: [c for c in candidates if c.score >= 0.025]),
        ("min_prob=0.03", lambda candidates: [c for c in candidates if c.score >= 0.03]),
        ("min_prob=0.035", lambda candidates: [c for c in candidates if c.score >= 0.035]),
        ("min_prob=0.04", lambda candidates: [c for c in candidates if c.score >= 0.04]),
        ("top_half_score", lambda candidates: top_half_score(candidates)),
        ("top_third", lambda candidates: top_third_score(candidates)),
    ]

    for name, cutoff_fn in cutoff_strategies:
        ddr_list = []
        ddp_list = []
        ddf1_list = []
        avg_n_list = []

        for patient in patients:
            confirmed_cuis, denied_cuis = get_patient_symptoms(patient, symptom_mapping)
            gt_dd_cuis = get_gt_dd(patient, disease_mapping)

            if not confirmed_cuis or not gt_dd_cuis:
                continue

            candidates = run_strategy(driver, "v7", query, list(confirmed_cuis), list(denied_cuis), top_k=49)
            if not candidates:
                continue

            # Apply cutoff
            filtered = cutoff_fn(candidates)
            if not filtered:
                filtered = candidates[:1]  # At least 1

            avg_n_list.append(len(filtered))
            predicted_cuis = {c.cui for c in filtered}
            ddr, ddp, ddf1 = calculate_dd_metrics(predicted_cuis, gt_dd_cuis)
            ddr_list.append(ddr)
            ddp_list.append(ddp)
            ddf1_list.append(ddf1)

        if ddr_list:
            avg_ddr = sum(ddr_list) / len(ddr_list) * 100
            avg_ddp = sum(ddp_list) / len(ddp_list) * 100
            avg_ddf1 = sum(ddf1_list) / len(ddf1_list) * 100
            avg_n = sum(avg_n_list) / len(avg_n_list)
            print(f"  {name:<15}: DDR={avg_ddr:5.1f}%, DDP={avg_ddp:5.1f}%, DDF1={avg_ddf1:5.1f}%, avg_n={avg_n:.1f}")


def cumulative_cutoff(candidates: list, threshold: float) -> list:
    """Cumulative probability cutoff."""
    result = []
    cumul = 0.0
    for c in candidates:
        result.append(c)
        cumul += c.score
        if cumul >= threshold:
            break
    return result


def top_half_score(candidates: list) -> list:
    """Include candidates with score >= half of top score."""
    if not candidates:
        return []
    top_score = candidates[0].score
    threshold = top_score / 2
    return [c for c in candidates if c.score >= threshold]


def top_third_score(candidates: list) -> list:
    """Include candidates with score >= 1/3 of top score."""
    if not candidates:
        return []
    top_score = candidates[0].score
    threshold = top_score / 3
    return [c for c in candidates if c.score >= threshold]


def main():
    print("=" * 60)
    print("Cypher Strategy Comparison for DDF1 Improvement")
    print("=" * 60)

    # 데이터 로드
    print("\nLoading data...")
    patients = load_patients(n=100)
    symptom_mapping, disease_mapping, cui_to_disease = load_mappings()

    # Neo4j 연결
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))

    # 결과 저장
    results = {name: {"ddr": [], "ddp": [], "ddf1": [], "gtpa1": []} for name in STRATEGIES}

    print(f"\nTesting {len(STRATEGIES)} strategies on {len(patients)} patients...")

    skip_count = 0

    # Debug first patient before loop
    if patients:
        p = patients[0]
        print(f"\n  DEBUG: First patient raw data")
        print(f"    initial_evidence: {p['initial_evidence']}")
        print(f"    evidences[:3]: {p['evidences'][:3] if p['evidences'] else []}")
        print(f"    pathology: {p['pathology']}")
        print(f"    differential[:3]: {p['differential'][:3] if p['differential'] else []}")

        # Check mappings
        initial = p["initial_evidence"]
        print(f"    initial in symptom_mapping: {initial in symptom_mapping}")
        if initial in symptom_mapping:
            print(f"    initial CUI: {symptom_mapping[initial]}")

        print(f"    pathology in disease_mapping: {p['pathology'] in disease_mapping}")
        if p['pathology'] in disease_mapping:
            print(f"    pathology CUI: {disease_mapping[p['pathology']]}")

    for i, patient in enumerate(patients):
        if (i + 1) % 20 == 0:
            print(f"  Processing patient {i + 1}/{len(patients)}...")

        # 환자 데이터 준비
        confirmed_cuis, denied_cuis = get_patient_symptoms(patient, symptom_mapping)
        gt_dd_cuis = get_gt_dd(patient, disease_mapping)
        gt_pathology_cui = disease_mapping.get(patient["pathology"], {}).get("cui")

        if not confirmed_cuis or not gt_dd_cuis:
            skip_count += 1
            if i < 3:
                print(f"  Skipping patient {i}: confirmed={len(confirmed_cuis)}, gt_dd={len(gt_dd_cuis)}")
            continue

        # Debug first patient
        if i == 0:
            print(f"\n  DEBUG: First patient")
            print(f"    confirmed_cuis: {list(confirmed_cuis)[:3]}...")
            print(f"    gt_dd_cuis: {list(gt_dd_cuis)[:3]}...")
            print(f"    gt_pathology_cui: {gt_pathology_cui}")

        # 각 전략 실행
        for name, query in STRATEGIES.items():
            candidates = run_strategy(
                driver, name, query,
                list(confirmed_cuis), list(denied_cuis),
                top_k=15,
            )

            # Debug first strategy of first patient
            if i == 0 and name == "v4_coverage":
                print(f"    Strategy {name}: {len(candidates)} candidates")
                if candidates:
                    print(f"      Top 3: {[(c.cui, c.name, c.score) for c in candidates[:3]]}")

            if not candidates:
                continue

            # 예측 DD CUI set
            predicted_cuis = {c.cui for c in candidates}

            # DD 메트릭 계산
            ddr, ddp, ddf1 = calculate_dd_metrics(predicted_cuis, gt_dd_cuis)
            results[name]["ddr"].append(ddr)
            results[name]["ddp"].append(ddp)
            results[name]["ddf1"].append(ddf1)

            # GTPA@1 계산
            top1_correct = 1 if candidates and candidates[0].cui == gt_pathology_cui else 0
            results[name]["gtpa1"].append(top1_correct)

    # 결과 출력
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"{'Strategy':<25} {'DDR':>8} {'DDP':>8} {'DDF1':>8} {'GTPA@1':>8}")
    print("-" * 80)

    best_ddf1 = 0
    best_strategy = None

    for name in STRATEGIES:
        r = results[name]
        if r["ddf1"]:
            avg_ddr = sum(r["ddr"]) / len(r["ddr"]) * 100
            avg_ddp = sum(r["ddp"]) / len(r["ddp"]) * 100
            avg_ddf1 = sum(r["ddf1"]) / len(r["ddf1"]) * 100
            avg_gtpa1 = sum(r["gtpa1"]) / len(r["gtpa1"]) * 100

            print(f"{name:<25} {avg_ddr:>7.1f}% {avg_ddp:>7.1f}% {avg_ddf1:>7.1f}% {avg_gtpa1:>7.1f}%")

            if avg_ddf1 > best_ddf1:
                best_ddf1 = avg_ddf1
                best_strategy = name

    print("-" * 80)
    print(f"Skipped patients: {skip_count}")
    print(f"\nBest strategy for DDF1: {best_strategy} ({best_ddf1:.1f}%)")

    # AARLC 비교
    if best_strategy:
        print("\n" + "=" * 80)
        print("vs AARLC Baseline")
        print("=" * 80)
        print(f"AARLC DDR:  97.73%")
        print(f"AARLC DDF1: 78.24%")
        print(f"Best DDR:   {sum(results[best_strategy]['ddr']) / len(results[best_strategy]['ddr']) * 100:.1f}%")
        print(f"Best DDF1:  {best_ddf1:.1f}%")

    # Top-K 영향 테스트
    test_top_k_impact(driver, patients, symptom_mapping, disease_mapping)

    # 확률 기반 cutoff 테스트
    test_probability_cutoff(driver, patients, symptom_mapping, disease_mapping)

    driver.close()


if __name__ == "__main__":
    main()
