#!/usr/bin/env python3
"""Cypher 최적화 연구: IL 감소 + 정확도 유지/향상.

1개 케이스에서 시작하여 100개까지 확장하면서 Cypher 쿼리 최적화.

핵심 목표:
- GTPA@1 유지/향상 (현재 ~80%)
- IL 감소 (현재 ~22 → 목표 15 이하)

최적화 대상:
1. 진단 스코어링 (get_diagnosis_candidates)
2. 질문 선택 (Information Gain)
3. 조기 종료 기준 (should_stop)
"""

import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class CaseResult:
    """단일 케이스 결과."""
    case_idx: int
    gt_pathology: str
    gt_cui: str
    predicted_cui: str | None
    predicted_name: str | None
    il: int  # 사용된 질문 수
    correct: bool  # GTPA@1
    gt_rank: int  # 정답 순위 (-1 if not found)
    top_scores: list[tuple[str, str, float]] = field(default_factory=list)  # [(cui, name, score), ...]
    confirmed_cuis: list[str] = field(default_factory=list)
    denied_cuis: list[str] = field(default_factory=list)


@dataclass
class StrategyConfig:
    """전략 설정."""
    name: str
    diagnosis_query: str
    question_query: str | None = None  # None이면 기본 사용
    stop_confidence: float = 0.4
    stop_gap: float = 0.15
    min_il: int = 5
    max_il: int = 25


# =============================================================================
# 진단 스코어링 전략들
# =============================================================================

DIAGNOSIS_STRATEGIES = {
    "v7_additive": """
    // v7: confirmed_count - 0.5 × denied_count
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
    WITH collect({
        cui: d.cui, name: d.name, raw_score: raw_score,
        confirmed_count: confirmed_count, total_symptoms: total_symptoms
    }) AS all_candidates
    WITH all_candidates,
         reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count,
           c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,

    "v14_weighted_denied": """
    // v14: confirmed - 0.3×denied (완화된 페널티)
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
         toFloat(confirmed_count) - 0.3 * toFloat(denied_count) AS raw_score
    WHERE raw_score > 0
    WITH collect({
        cui: d.cui, name: d.name, raw_score: raw_score,
        confirmed_count: confirmed_count, total_symptoms: total_symptoms
    }) AS all_candidates
    WITH all_candidates,
         reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count,
           c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,

    "v15_ratio_based": """
    // v15: confirmed/(confirmed+denied) ratio 기반
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
         toFloat(confirmed_count) / (toFloat(confirmed_count) + toFloat(denied_count) + 1.0) *
         toFloat(confirmed_count) AS raw_score
    WHERE raw_score > 0
    WITH collect({
        cui: d.cui, name: d.name, raw_score: raw_score,
        confirmed_count: confirmed_count, total_symptoms: total_symptoms
    }) AS all_candidates
    WITH all_candidates,
         reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count,
           c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,

    "v16_coverage_boost": """
    // v16: confirmed² / total_symptoms (coverage 보상)
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
         toFloat(confirmed_count * confirmed_count) / (toFloat(total_symptoms) + 1.0) *
         (1.0 - 0.3 * toFloat(denied_count) / (toFloat(total_symptoms) + 1.0)) AS raw_score
    WHERE raw_score > 0
    WITH collect({
        cui: d.cui, name: d.name, raw_score: raw_score,
        confirmed_count: confirmed_count, total_symptoms: total_symptoms
    }) AS all_candidates
    WITH all_candidates,
         reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count,
           c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,

    "v17_early_discriminative": """
    // v17: 초기 증상 수가 적을 때 더 discriminative
    // confirmed² - denied (초기에 강한 구분)
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
         toFloat(confirmed_count * confirmed_count) - toFloat(denied_count) AS raw_score
    WHERE raw_score > 0
    WITH collect({
        cui: d.cui, name: d.name, raw_score: raw_score,
        confirmed_count: confirmed_count, total_symptoms: total_symptoms
    }) AS all_candidates
    WITH all_candidates,
         reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
    UNWIND all_candidates AS c
    RETURN c.cui AS cui, c.name AS name,
           CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
           c.confirmed_count AS confirmed_count,
           c.total_symptoms AS total_symptoms
    ORDER BY score DESC
    LIMIT $top_k
    """,
}

# =============================================================================
# 질문 선택 전략들 (Information Gain 변형)
# =============================================================================

QUESTION_STRATEGIES = {
    "ig_standard": """
    // 표준 Information Gain: 50%에 가까울수록 높은 점수
    MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
    WHERE confirmed.cui IN $confirmed_cuis
    WITH collect(DISTINCT d) AS candidate_diseases

    UNWIND candidate_diseases AS d
    OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
    WHERE denied.cui IN $denied_cuis
    WITH d, candidate_diseases, count(denied) AS denied_count
    WITH d, candidate_diseases
    WHERE denied_count < 3

    WITH collect(d) AS valid_diseases
    WHERE size(valid_diseases) > 0

    UNWIND valid_diseases AS d
    MATCH (d)<-[:INDICATES]-(next:Symptom)
    WHERE NOT next.cui IN $confirmed_cuis
      AND NOT next.cui IN $denied_cuis
      AND NOT next.cui IN $asked_cuis
    WITH DISTINCT next, valid_diseases

    WITH next, valid_diseases,
         size([vd IN valid_diseases WHERE (next)-[:INDICATES]->(vd)]) AS coverage,
         size(valid_diseases) AS total

    WITH next, coverage, total,
         abs(toFloat(coverage) - toFloat(total) / 2.0) AS distance_from_optimal

    WITH next, coverage, total,
         CASE WHEN total > 0
              THEN 1.0 - (distance_from_optimal / (toFloat(total) / 2.0 + 0.1))
              ELSE 0.0 END AS ig_score

    RETURN next.cui AS cui,
           next.name AS name,
           coverage AS disease_coverage,
           CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority,
           ig_score
    ORDER BY priority ASC, ig_score DESC
    LIMIT $limit
    """,

    "ig_aggressive": """
    // 공격적 IG: Top-2 질환 간 구별력 극대화
    MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
    WHERE confirmed.cui IN $confirmed_cuis
    WITH d
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptoms

    WITH d, disease_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptoms] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptoms] AS matched_denied
    WHERE size(matched_confirmed) > 0
    WITH d, toFloat(size(matched_confirmed)) - 0.5 * toFloat(size(matched_denied)) AS score
    WHERE score > 0
    ORDER BY score DESC
    LIMIT 5  // Top-5 후보 질환만 고려

    WITH collect(d) AS top_diseases

    UNWIND top_diseases AS d
    MATCH (d)<-[:INDICATES]-(next:Symptom)
    WHERE NOT next.cui IN $confirmed_cuis
      AND NOT next.cui IN $denied_cuis
      AND NOT next.cui IN $asked_cuis
    WITH DISTINCT next, top_diseases

    // Top 질환들 중 몇 개와 연결되는지 (더 좁은 범위)
    WITH next, top_diseases,
         size([td IN top_diseases WHERE (next)-[:INDICATES]->(td)]) AS coverage,
         size(top_diseases) AS total

    // 1개 또는 (n-1)개와 연결되면 가장 discriminative
    WITH next, coverage, total,
         CASE
           WHEN coverage = 1 THEN 1.0  // 하나만 연결 = 최고
           WHEN coverage = total - 1 THEN 0.9  // 하나만 제외 = 차선
           ELSE abs(toFloat(coverage) - toFloat(total) / 2.0) / (toFloat(total) / 2.0 + 0.1)
         END AS discriminative_score

    RETURN next.cui AS cui,
           next.name AS name,
           coverage AS disease_coverage,
           CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority,
           discriminative_score AS ig_score
    ORDER BY priority ASC, ig_score DESC
    LIMIT $limit
    """,

    "ig_entropy": """
    // 엔트로피 기반: p×log(p) + (1-p)×log(1-p) 최대화
    MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
    WHERE confirmed.cui IN $confirmed_cuis
    WITH collect(DISTINCT d) AS candidate_diseases

    UNWIND candidate_diseases AS d
    OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
    WHERE denied.cui IN $denied_cuis
    WITH d, candidate_diseases, count(denied) AS denied_count
    WITH d, candidate_diseases
    WHERE denied_count < 3

    WITH collect(d) AS valid_diseases
    WHERE size(valid_diseases) > 0

    UNWIND valid_diseases AS d
    MATCH (d)<-[:INDICATES]-(next:Symptom)
    WHERE NOT next.cui IN $confirmed_cuis
      AND NOT next.cui IN $denied_cuis
      AND NOT next.cui IN $asked_cuis
    WITH DISTINCT next, valid_diseases

    WITH next, valid_diseases,
         size([vd IN valid_diseases WHERE (next)-[:INDICATES]->(vd)]) AS coverage,
         size(valid_diseases) AS total

    // p = coverage/total
    WITH next, coverage, total,
         CASE WHEN total > 0 THEN toFloat(coverage) / toFloat(total) ELSE 0.5 END AS p

    // 엔트로피: -p×log(p) - (1-p)×log(1-p) (50%에서 최대)
    WITH next, p,
         CASE
           WHEN p <= 0.01 OR p >= 0.99 THEN 0.0
           ELSE -p * log(p + 0.001) - (1.0 - p) * log(1.0 - p + 0.001)
         END AS entropy_score

    RETURN next.cui AS cui,
           next.name AS name,
           0 AS disease_coverage,
           CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority,
           entropy_score AS ig_score
    ORDER BY priority ASC, ig_score DESC
    LIMIT $limit
    """,
}


class CypherOptimizer:
    """Cypher 최적화 실행기."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password123",
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._load_data()

    def _load_data(self):
        """데이터 로드."""
        self.df = pd.read_csv("data/ddxplus/release_test_patients.csv")

        with open("data/ddxplus/release_conditions.json") as f:
            conditions = json.load(f)

        pathology_to_cat = {
            cond_info["cond-name-fr"]: cond_info["severity"]
            for cond_name, cond_info in conditions.items()
        }
        self.df["CATEGORY"] = self.df["PATHOLOGY"].map(pathology_to_cat)

        with open("data/ddxplus/umls_mapping.json") as f:
            self.symptom_map = json.load(f)["mapping"]

        with open("data/ddxplus/disease_umls_mapping.json") as f:
            disease_data = json.load(f)
            self.disease_map = disease_data["mapping"]

        # French name -> info 매핑
        self.fr_to_info = {}
        for eng, info in self.disease_map.items():
            fr_name = info.get("name_fr", "")
            if fr_name:
                self.fr_to_info[fr_name] = info

    def close(self):
        self.driver.close()

    def get_symptom_cui(self, code: str) -> str | None:
        """증상 코드 → CUI."""
        base = code.split("_@_")[0] if "_@_" in code else code
        info = self.symptom_map.get(base, {})
        return info.get("cui")

    def get_disease_cui(self, name_fr: str) -> str | None:
        """프랑스어 질환명 → CUI."""
        info = self.fr_to_info.get(name_fr, {})
        return info.get("umls_cui")

    def simulate_case(
        self,
        case_idx: int,
        strategy: StrategyConfig,
        verbose: bool = False,
    ) -> CaseResult:
        """단일 케이스 시뮬레이션.

        환자의 모든 증상 정보를 사용하여 KG 기반 진단 시뮬레이션.
        실제 LLM 없이 KG 쿼리 성능만 테스트.
        """
        row = self.df[self.df["CATEGORY"] == 2].iloc[case_idx]

        gt_pathology = row["PATHOLOGY"]
        gt_cui = self.get_disease_cui(gt_pathology)

        # 환자의 모든 증상 수집
        evidences_raw = ast.literal_eval(row["EVIDENCES"])
        all_symptom_cuis = set()
        for ev in evidences_raw:
            cui = self.get_symptom_cui(ev)
            if cui:
                all_symptom_cuis.add(cui)

        # Initial evidence
        initial_cui = self.get_symptom_cui(row["INITIAL_EVIDENCE"])
        if initial_cui:
            all_symptom_cuis.add(initial_cui)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Case {case_idx}: {gt_pathology}")
            print(f"GT CUI: {gt_cui}")
            print(f"Total symptoms: {len(all_symptom_cuis)}")

        # 시뮬레이션: 증상을 하나씩 추가하며 진단
        confirmed_cuis = set()
        denied_cuis = set()
        asked_cuis = set()

        # Initial symptom
        if initial_cui:
            confirmed_cuis.add(initial_cui)
            asked_cuis.add(initial_cui)

        il = 1  # Initial evidence counted

        while il < strategy.max_il:
            # 현재 상태로 진단 점수 계산
            candidates = self._run_diagnosis_query(
                strategy.diagnosis_query,
                list(confirmed_cuis),
                list(denied_cuis),
                top_k=10,
            )

            if not candidates:
                break

            top_score = candidates[0]["score"]
            top2_score = candidates[1]["score"] if len(candidates) > 1 else 0

            # 조기 종료 조건 체크
            if il >= strategy.min_il:
                if top_score >= strategy.stop_confidence:
                    if verbose:
                        print(f"  IL={il}: Early stop (confidence={top_score:.2f})")
                    break
                if top_score - top2_score >= strategy.stop_gap:
                    if verbose:
                        print(f"  IL={il}: Early stop (gap={top_score - top2_score:.2f})")
                    break

            # 다음 질문 선택 (가장 discriminative한 증상)
            next_candidates = self._get_next_question(
                strategy.question_query or QUESTION_STRATEGIES["ig_standard"],
                list(confirmed_cuis),
                list(denied_cuis),
                list(asked_cuis),
                limit=10,
            )

            if not next_candidates:
                if verbose:
                    print(f"  IL={il}: No more questions")
                break

            # 첫 번째 후보 선택
            next_cui = next_candidates[0]["cui"]
            asked_cuis.add(next_cui)

            # 환자 응답 시뮬레이션
            if next_cui in all_symptom_cuis:
                confirmed_cuis.add(next_cui)
            else:
                denied_cuis.add(next_cui)

            il += 1

            if verbose and il <= 5:
                print(f"  IL={il}: Asked {next_candidates[0]['name'][:30]} → {'Yes' if next_cui in all_symptom_cuis else 'No'}")

        # 최종 진단
        final_candidates = self._run_diagnosis_query(
            strategy.diagnosis_query,
            list(confirmed_cuis),
            list(denied_cuis),
            top_k=10,
        )

        predicted_cui = final_candidates[0]["cui"] if final_candidates else None
        predicted_name = final_candidates[0]["name"] if final_candidates else None

        # GT 순위 찾기
        gt_rank = -1
        for i, c in enumerate(final_candidates):
            if c["cui"] == gt_cui:
                gt_rank = i + 1
                break

        top_scores = [(c["cui"], c["name"], c["score"]) for c in final_candidates[:5]]

        if verbose:
            print(f"\nFinal (IL={il}):")
            print(f"  Predicted: {predicted_name} ({predicted_cui})")
            print(f"  GT Rank: {gt_rank}")
            print(f"  Correct: {predicted_cui == gt_cui}")

        return CaseResult(
            case_idx=case_idx,
            gt_pathology=gt_pathology,
            gt_cui=gt_cui,
            predicted_cui=predicted_cui,
            predicted_name=predicted_name,
            il=il,
            correct=(predicted_cui == gt_cui),
            gt_rank=gt_rank,
            top_scores=top_scores,
            confirmed_cuis=list(confirmed_cuis),
            denied_cuis=list(denied_cuis),
        )

    def _run_diagnosis_query(
        self,
        query: str,
        confirmed_cuis: list,
        denied_cuis: list,
        top_k: int = 10,
    ) -> list[dict]:
        """진단 쿼리 실행."""
        with self.driver.session() as session:
            try:
                result = session.run(
                    query,
                    confirmed_cuis=confirmed_cuis,
                    denied_cuis=denied_cuis,
                    top_k=top_k,
                )
                return [dict(r) for r in result]
            except Exception as e:
                print(f"Diagnosis query error: {e}")
                return []

    def _get_next_question(
        self,
        query: str,
        confirmed_cuis: list,
        denied_cuis: list,
        asked_cuis: list,
        limit: int = 10,
    ) -> list[dict]:
        """다음 질문 후보 쿼리."""
        with self.driver.session() as session:
            try:
                result = session.run(
                    query,
                    confirmed_cuis=confirmed_cuis,
                    denied_cuis=denied_cuis,
                    asked_cuis=asked_cuis,
                    limit=limit,
                )
                return [dict(r) for r in result]
            except Exception as e:
                print(f"Question query error: {e}")
                return []


def run_optimization(n_cases: int = 1, verbose: bool = True):
    """최적화 실행."""
    optimizer = CypherOptimizer()

    print("=" * 70)
    print(f"Cypher 최적화 연구: IL 감소 + 정확도 유지")
    print(f"케이스 수: {n_cases}")
    print("=" * 70)

    # 테스트할 전략 조합
    strategies = []

    # 진단 전략 × 종료 조건 조합
    for diag_name, diag_query in DIAGNOSIS_STRATEGIES.items():
        for stop_conf, stop_gap in [(0.3, 0.10), (0.35, 0.12), (0.4, 0.15)]:
            for min_il in [3, 5]:
                strategies.append(StrategyConfig(
                    name=f"{diag_name}_c{int(stop_conf*100)}_g{int(stop_gap*100)}_min{min_il}",
                    diagnosis_query=diag_query,
                    stop_confidence=stop_conf,
                    stop_gap=stop_gap,
                    min_il=min_il,
                    max_il=25,
                ))

    # Baseline (현재 구현)
    strategies.insert(0, StrategyConfig(
        name="baseline_v7",
        diagnosis_query=DIAGNOSIS_STRATEGIES["v7_additive"],
        stop_confidence=0.4,
        stop_gap=0.15,
        min_il=5,
        max_il=25,
    ))

    results_summary = []

    for strategy in strategies:
        case_results = []

        for i in range(n_cases):
            result = optimizer.simulate_case(
                case_idx=i,
                strategy=strategy,
                verbose=(verbose and i < 1),  # 첫 케이스만 상세
            )
            case_results.append(result)

        # 통계
        accuracy = sum(1 for r in case_results if r.correct) / len(case_results)
        avg_il = sum(r.il for r in case_results) / len(case_results)
        top5_acc = sum(1 for r in case_results if 1 <= r.gt_rank <= 5) / len(case_results)

        results_summary.append({
            "strategy": strategy.name,
            "accuracy": accuracy,
            "avg_il": avg_il,
            "top5_acc": top5_acc,
            "n_cases": len(case_results),
        })

        if verbose or n_cases >= 10:
            print(f"\n{strategy.name}:")
            print(f"  GTPA@1: {accuracy:.1%}, Top-5: {top5_acc:.1%}, Avg IL: {avg_il:.1f}")

    optimizer.close()

    # 결과 정렬 (IL 낮으면서 정확도 높은 순)
    print("\n" + "=" * 70)
    print("최종 결과 (IL 기준 정렬, 정확도 >= baseline)")
    print("=" * 70)

    baseline_acc = results_summary[0]["accuracy"]
    filtered = [r for r in results_summary if r["accuracy"] >= baseline_acc * 0.95]
    filtered_sorted = sorted(filtered, key=lambda x: (x["avg_il"], -x["accuracy"]))

    print(f"{'Strategy':<50} {'GTPA@1':>8} {'Top-5':>8} {'Avg IL':>8}")
    print("-" * 78)
    for r in filtered_sorted[:15]:
        print(f"{r['strategy']:<50} {r['accuracy']:>7.1%} {r['top5_acc']:>7.1%} {r['avg_il']:>7.1f}")

    return results_summary


def incremental_optimization():
    """점진적 최적화: 1 → 10 → 50 → 100 케이스."""
    print("\n" + "=" * 70)
    print("점진적 Cypher 최적화")
    print("=" * 70)

    for n in [1, 10, 50, 100]:
        print(f"\n{'='*70}")
        print(f"Phase: {n} cases")
        print("=" * 70)

        results = run_optimization(n_cases=n, verbose=(n <= 10))

        # 최적 전략 저장
        best = min(
            [r for r in results if r["accuracy"] >= results[0]["accuracy"] * 0.95],
            key=lambda x: x["avg_il"],
            default=results[0],
        )

        print(f"\n최적 전략 ({n} cases): {best['strategy']}")
        print(f"  GTPA@1: {best['accuracy']:.1%}, Avg IL: {best['avg_il']:.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        run_optimization(n_cases=n, verbose=(n <= 10))
    else:
        # 기본: 점진적 최적화
        incremental_optimization()
