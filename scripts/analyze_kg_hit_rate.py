#!/usr/bin/env python3
"""KG 증상 제안의 Hit Rate 분석.

KG가 제시하는 Top-N 증상 중 실제 유효한 증상의 비율 측정.

예: KG가 10개 증상 제시, 환자가 1,3,7번 증상만 있음
- Top-1: 1/1 = 100%
- Top-2: 1/2 = 50%
- Top-3: 2/3 = 66.7%
- Top-7: 3/7 = 42.9%
- Top-10: 3/10 = 30%

이를 통해 LLM에게 몇 개의 증상을 제시해야 최적인지 결정.
"""

import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class HitRateResult:
    """각 라운드의 Hit Rate 결과."""
    case_idx: int
    round_num: int
    kg_suggestions: list[str]  # KG가 제시한 CUI 순서대로
    valid_positions: list[int]  # 유효한 증상의 위치 (0-indexed)
    hit_rates: dict[int, float]  # {N: hit_rate}


# Information Gain 기반 질문 선택 쿼리
QUESTION_QUERY = """
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
"""

# 진단 쿼리
DIAGNOSIS_QUERY = """
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
WITH collect({cui: d.cui, raw_score: raw_score}) AS all_candidates
WITH all_candidates,
     reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
UNWIND all_candidates AS c
RETURN c.cui AS cui,
       CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score
ORDER BY score DESC
LIMIT 10
"""


class HitRateAnalyzer:
    """KG Hit Rate 분석기."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "password123")
        )
        self._load_data()

    def _load_data(self):
        self.df = pd.read_csv("data/ddxplus/release_test_patients.csv")

        with open("data/ddxplus/release_conditions.json") as f:
            conditions = json.load(f)
        pathology_to_cat = {
            cond_info["cond-name-fr"]: cond_info["severity"]
            for cond_info in conditions.values()
        }
        self.df["CATEGORY"] = self.df["PATHOLOGY"].map(pathology_to_cat)

        with open("data/ddxplus/umls_mapping.json") as f:
            self.symptom_map = json.load(f)["mapping"]

        with open("data/ddxplus/disease_umls_mapping.json") as f:
            self.disease_map = json.load(f)["mapping"]

        self.fr_to_info = {}
        for eng, info in self.disease_map.items():
            fr_name = info.get("name_fr", "")
            if fr_name:
                self.fr_to_info[fr_name] = info

    def close(self):
        self.driver.close()

    def get_symptom_cui(self, code: str) -> str | None:
        base = code.split("_@_")[0] if "_@_" in code else code
        info = self.symptom_map.get(base, {})
        return info.get("cui")

    def get_disease_cui(self, name_fr: str) -> str | None:
        info = self.fr_to_info.get(name_fr, {})
        return info.get("umls_cui")

    def analyze_case(self, case_idx: int, max_rounds: int = 20) -> list[HitRateResult]:
        """단일 케이스의 각 라운드별 Hit Rate 분석."""
        row = self.df[self.df["CATEGORY"] == 2].iloc[case_idx]

        # 환자의 모든 유효 증상
        evidences_raw = ast.literal_eval(row["EVIDENCES"])
        all_valid_cuis = set()
        for ev in evidences_raw:
            cui = self.get_symptom_cui(ev)
            if cui:
                all_valid_cuis.add(cui)

        initial_cui = self.get_symptom_cui(row["INITIAL_EVIDENCE"])
        if initial_cui:
            all_valid_cuis.add(initial_cui)

        # 시뮬레이션 시작
        confirmed_cuis = set()
        denied_cuis = set()
        asked_cuis = set()

        if initial_cui:
            confirmed_cuis.add(initial_cui)
            asked_cuis.add(initial_cui)

        results = []

        for round_num in range(1, max_rounds + 1):
            # KG에서 다음 질문 후보 가져오기
            candidates = self._run_query(
                QUESTION_QUERY,
                confirmed_cuis=list(confirmed_cuis),
                denied_cuis=list(denied_cuis),
                asked_cuis=list(asked_cuis),
                limit=10,
            )

            if not candidates:
                break

            # KG 제안 순서대로 CUI 리스트
            kg_suggestions = [c["cui"] for c in candidates]

            # 유효한 증상의 위치 찾기
            valid_positions = []
            for i, cui in enumerate(kg_suggestions):
                if cui in all_valid_cuis:
                    valid_positions.append(i)

            # Top-N별 Hit Rate 계산
            hit_rates = {}
            for n in range(1, len(kg_suggestions) + 1):
                valid_in_topn = sum(1 for pos in valid_positions if pos < n)
                hit_rates[n] = valid_in_topn / n

            results.append(HitRateResult(
                case_idx=case_idx,
                round_num=round_num,
                kg_suggestions=kg_suggestions,
                valid_positions=valid_positions,
                hit_rates=hit_rates,
            ))

            # 다음 라운드를 위해 Top-1 선택 (KG-only 방식)
            next_cui = kg_suggestions[0]
            asked_cuis.add(next_cui)

            if next_cui in all_valid_cuis:
                confirmed_cuis.add(next_cui)
            else:
                denied_cuis.add(next_cui)

            # 조기 종료 체크
            diag = self._run_query(
                DIAGNOSIS_QUERY,
                confirmed_cuis=list(confirmed_cuis),
                denied_cuis=list(denied_cuis),
            )
            if diag and diag[0]["score"] >= 0.3:
                break

        return results

    def _run_query(self, query: str, **params) -> list[dict]:
        with self.driver.session() as session:
            try:
                result = session.run(query, **params)
                return [dict(r) for r in result]
            except Exception as e:
                print(f"Query error: {e}")
                return []


def run_analysis(n_cases: int = 100):
    """Hit Rate 분석 실행."""
    analyzer = HitRateAnalyzer()

    print("=" * 70)
    print(f"KG Hit Rate 분석 (cases={n_cases})")
    print("=" * 70)

    # 모든 라운드의 Hit Rate 수집
    all_hit_rates = {n: [] for n in range(1, 11)}  # Top-1 to Top-10

    for case_idx in range(n_cases):
        if (case_idx + 1) % 20 == 0:
            print(f"  Processing case {case_idx + 1}/{n_cases}...")

        results = analyzer.analyze_case(case_idx)

        for r in results:
            for n, rate in r.hit_rates.items():
                if n <= 10:
                    all_hit_rates[n].append(rate)

    analyzer.close()

    # 결과 집계
    print("\n" + "=" * 70)
    print("Top-N별 평균 Hit Rate")
    print("=" * 70)
    print(f"{'Top-N':<8} {'Hit Rate':>12} {'Samples':>10}")
    print("-" * 35)

    summary = []
    for n in range(1, 11):
        rates = all_hit_rates[n]
        if rates:
            mean_rate = sum(rates) / len(rates)
            print(f"Top-{n:<5} {mean_rate:>11.1%} {len(rates):>10}")
            summary.append({"n": n, "hit_rate": mean_rate, "samples": len(rates)})

    # 분석
    print("\n" + "=" * 70)
    print("분석")
    print("=" * 70)

    if len(summary) >= 2:
        print("\n[해석]")
        print("- Hit Rate = KG가 제시한 Top-N 중 실제 환자가 가진 증상의 비율")
        print("- Hit Rate가 높을수록 LLM이 유효한 증상을 선택할 확률이 높음")

        print("\n[최적 N 찾기]")
        # Hit Rate * N을 최대화하는 N 찾기 (expected valid symptoms)
        best_n = 1
        best_expected = 0
        for s in summary:
            expected = s["hit_rate"] * s["n"]
            if expected > best_expected:
                best_expected = expected
                best_n = s["n"]
            print(f"  Top-{s['n']}: Hit Rate {s['hit_rate']:.1%} × {s['n']} = 기대 유효 증상 {expected:.2f}개")

        print(f"\n[결론]")
        print(f"  - LLM에게 Top-{best_n} 제시 시 기대 유효 증상 {best_expected:.2f}개")

        # 50% 이상 Hit Rate 유지되는 최대 N
        max_n_50 = 1
        for s in summary:
            if s["hit_rate"] >= 0.5:
                max_n_50 = s["n"]
        print(f"  - Hit Rate ≥ 50% 유지: Top-{max_n_50}까지")

    # JSON 저장
    output_file = Path("results/kg_hit_rate_analysis.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n결과 저장: {output_file}")

    return summary


if __name__ == "__main__":
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_analysis(n_cases=n_cases)
