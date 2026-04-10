#!/usr/bin/env python3
"""최적 Top-N 학술적 분석.

Expected Information Gain 최적화를 통한 최적 N 도출.

E[IG | Top-N] = Σ P(LLM이 i번째 선택) × IG(i번째 증상)
최적 N* = argmax E[IG | Top-N]

측정 항목:
1. IG(position): 각 위치별 평균 Information Gain
2. P(position | N): LLM의 선택 분포 (기존 로그 분석)
3. E[IG | N]: 기대 Information Gain
"""

import ast
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Cypher 쿼리
# =============================================================================

# 후보 질환 가져오기
CANDIDATE_DISEASES_QUERY = """
MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
WHERE confirmed.cui IN $confirmed_cuis
WITH collect(DISTINCT d) AS candidate_diseases

UNWIND candidate_diseases AS d
OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
WHERE denied.cui IN $denied_cuis
WITH d, candidate_diseases, count(denied) AS denied_count
WHERE denied_count < 3

RETURN d.cui AS cui, d.name AS name
"""

# 다음 질문 후보 (IG 점수 포함)
QUESTION_QUERY_WITH_IG = """
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

// Information Gain: 50%에 가까울수록 높음
WITH next, coverage, total,
     toFloat(coverage) / toFloat(total) AS p

// Binary Entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
// IG = H(prior) - H(posterior) ≈ H(p) for uniform prior
WITH next, coverage, total, p,
     CASE
       WHEN p <= 0.01 OR p >= 0.99 THEN 0.0
       ELSE -p * log(p + 0.001) - (1.0 - p) * log(1.0 - p + 0.001)
     END AS entropy

RETURN next.cui AS cui,
       next.name AS name,
       coverage,
       total,
       p AS coverage_ratio,
       entropy AS ig_score
ORDER BY ig_score DESC
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


@dataclass
class PositionIG:
    """각 위치의 IG 통계."""
    position: int
    ig_values: list
    coverage_ratios: list

    @property
    def mean_ig(self) -> float:
        return sum(self.ig_values) / len(self.ig_values) if self.ig_values else 0

    @property
    def mean_coverage(self) -> float:
        return sum(self.coverage_ratios) / len(self.coverage_ratios) if self.coverage_ratios else 0


class OptimalTopNAnalyzer:
    """최적 Top-N 분석기."""

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

    def measure_position_ig(self, n_cases: int = 100, max_rounds: int = 15) -> dict[int, PositionIG]:
        """각 위치별 평균 IG 측정."""
        position_data = {i: PositionIG(i, [], []) for i in range(1, 11)}

        for case_idx in range(n_cases):
            if (case_idx + 1) % 20 == 0:
                print(f"  Measuring IG: case {case_idx + 1}/{n_cases}...")

            row = self.df[self.df["CATEGORY"] == 2].iloc[case_idx]

            # 환자 증상
            evidences_raw = ast.literal_eval(row["EVIDENCES"])
            all_valid_cuis = set()
            for ev in evidences_raw:
                cui = self.get_symptom_cui(ev)
                if cui:
                    all_valid_cuis.add(cui)

            initial_cui = self.get_symptom_cui(row["INITIAL_EVIDENCE"])
            if initial_cui:
                all_valid_cuis.add(initial_cui)

            confirmed_cuis = set()
            denied_cuis = set()
            asked_cuis = set()

            if initial_cui:
                confirmed_cuis.add(initial_cui)
                asked_cuis.add(initial_cui)

            for round_num in range(1, max_rounds + 1):
                # IG 점수 포함한 후보 가져오기
                candidates = self._run_query(
                    QUESTION_QUERY_WITH_IG,
                    confirmed_cuis=list(confirmed_cuis),
                    denied_cuis=list(denied_cuis),
                    asked_cuis=list(asked_cuis),
                    limit=10,
                )

                if not candidates:
                    break

                # 각 위치의 IG 기록
                for i, cand in enumerate(candidates):
                    pos = i + 1
                    if pos <= 10:
                        position_data[pos].ig_values.append(cand["ig_score"])
                        position_data[pos].coverage_ratios.append(cand["coverage_ratio"])

                # Top-1 선택으로 진행
                next_cui = candidates[0]["cui"]
                asked_cuis.add(next_cui)

                if next_cui in all_valid_cuis:
                    confirmed_cuis.add(next_cui)
                else:
                    denied_cuis.add(next_cui)

                # 조기 종료
                diag = self._run_query(
                    DIAGNOSIS_QUERY,
                    confirmed_cuis=list(confirmed_cuis),
                    denied_cuis=list(denied_cuis),
                )
                if diag and diag[0]["score"] >= 0.3:
                    break

        return position_data

    def analyze_llm_selection_distribution(self, log_dir: str = "results") -> dict[int, float]:
        """기존 벤치마크 로그에서 LLM 선택 분포 분석."""
        selection_counts = defaultdict(int)
        total_selections = 0

        log_path = Path(log_dir)

        # 가장 최근 결과 디렉토리 찾기
        result_dirs = sorted(log_path.glob("20*"), reverse=True)

        for result_dir in result_dirs[:3]:  # 최근 3개 디렉토리
            for log_file in result_dir.glob("interaction_logs_*.jsonl"):
                try:
                    with open(log_file) as f:
                        for line in f:
                            data = json.loads(line)
                            interactions = data.get("interactions", [])

                            for interaction in interactions:
                                # KG 후보 증상과 LLM 선택 비교
                                kg_candidates = interaction.get("kg_candidate_symptoms", [])
                                llm_selected_cui = interaction.get("llm_selected_cui", "")

                                if kg_candidates and llm_selected_cui:
                                    # LLM이 몇 번째를 선택했는지 찾기
                                    for i, cand in enumerate(kg_candidates):
                                        if cand.get("cui") == llm_selected_cui:
                                            selection_counts[i + 1] += 1
                                            total_selections += 1
                                            break
                except Exception as e:
                    print(f"  Warning: Error reading {log_file}: {e}")
                    continue

        # 확률 분포로 변환
        if total_selections > 0:
            distribution = {pos: count / total_selections
                          for pos, count in selection_counts.items()}
        else:
            # 로그가 없으면 균등 분포 가정
            print("  Warning: No interaction logs found. Using uniform distribution.")
            distribution = {i: 0.1 for i in range(1, 11)}

        return distribution, total_selections

    def calculate_expected_ig(
        self,
        position_ig: dict[int, PositionIG],
        llm_distribution: dict[int, float],
        top_n: int,
    ) -> float:
        """Top-N에서의 기대 Information Gain 계산.

        E[IG | Top-N] = Σ P(i|N) × IG(i) for i=1..N

        P(i|N): Top-N 중에서 i번째를 선택할 확률 (정규화)
        """
        # Top-N 내에서 확률 재정규화
        total_prob = sum(llm_distribution.get(i, 0) for i in range(1, top_n + 1))

        if total_prob == 0:
            # 균등 분포 가정
            normalized = {i: 1/top_n for i in range(1, top_n + 1)}
        else:
            normalized = {i: llm_distribution.get(i, 0) / total_prob
                         for i in range(1, top_n + 1)}

        # E[IG] 계산
        expected_ig = 0
        for pos in range(1, top_n + 1):
            p = normalized.get(pos, 0)
            ig = position_ig[pos].mean_ig if pos in position_ig else 0
            expected_ig += p * ig

        return expected_ig

    def _run_query(self, query: str, **params) -> list[dict]:
        with self.driver.session() as session:
            try:
                result = session.run(query, **params)
                return [dict(r) for r in result]
            except Exception as e:
                print(f"Query error: {e}")
                return []


def run_analysis(n_cases: int = 100):
    """전체 분석 실행."""
    analyzer = OptimalTopNAnalyzer()

    print("=" * 70)
    print(f"최적 Top-N 학술적 분석 (cases={n_cases})")
    print("=" * 70)

    # ==========================================================================
    # 1단계: 각 위치별 평균 IG 측정
    # ==========================================================================
    print("\n[1단계] 각 위치별 평균 Information Gain 측정...")
    position_ig = analyzer.measure_position_ig(n_cases)

    print("\n" + "=" * 70)
    print("위치별 평균 Information Gain")
    print("=" * 70)
    print(f"{'Position':<10} {'Mean IG':>12} {'Mean Coverage':>15} {'Samples':>10}")
    print("-" * 50)

    for pos in range(1, 11):
        data = position_ig[pos]
        print(f"  {pos:<8} {data.mean_ig:>11.4f} {data.mean_coverage:>14.1%} {len(data.ig_values):>10}")

    # ==========================================================================
    # 2단계: LLM 선택 분포 분석
    # ==========================================================================
    print("\n[2단계] LLM 선택 분포 분석 (기존 로그)...")
    llm_distribution, total_samples = analyzer.analyze_llm_selection_distribution()

    print("\n" + "=" * 70)
    print(f"LLM 선택 분포 (총 {total_samples} 샘플)")
    print("=" * 70)
    print(f"{'Position':<10} {'P(select)':>12}")
    print("-" * 25)

    for pos in range(1, 11):
        prob = llm_distribution.get(pos, 0)
        bar = "█" * int(prob * 50)
        print(f"  {pos:<8} {prob:>11.1%} {bar}")

    # ==========================================================================
    # 3단계: Expected IG 계산
    # ==========================================================================
    print("\n[3단계] Top-N별 Expected Information Gain 계산...")

    print("\n" + "=" * 70)
    print("Top-N별 Expected Information Gain")
    print("=" * 70)
    print(f"{'Top-N':<10} {'E[IG]':>12} {'vs Top-1':>12} {'vs Top-10':>12}")
    print("-" * 50)

    results = []
    baseline_ig = None
    top10_ig = None

    for n in range(1, 11):
        expected_ig = analyzer.calculate_expected_ig(position_ig, llm_distribution, n)

        if n == 1:
            baseline_ig = expected_ig
        if n == 10:
            top10_ig = expected_ig

        vs_top1 = ((expected_ig - baseline_ig) / baseline_ig * 100) if baseline_ig else 0
        vs_top10 = ((expected_ig - top10_ig) / top10_ig * 100) if top10_ig else 0

        results.append({
            "n": n,
            "expected_ig": expected_ig,
            "vs_top1_pct": vs_top1,
        })

        print(f"  Top-{n:<5} {expected_ig:>11.4f} {vs_top1:>+11.1f}%")

    # ==========================================================================
    # 4단계: 최적 N 도출
    # ==========================================================================
    print("\n" + "=" * 70)
    print("분석 결과")
    print("=" * 70)

    # 최대 E[IG]를 가지는 N
    optimal = max(results, key=lambda x: x["expected_ig"])

    print(f"\n최적 N* = {optimal['n']}")
    print(f"  E[IG | Top-{optimal['n']}] = {optimal['expected_ig']:.4f}")

    if optimal["n"] > 1:
        print(f"  vs Top-1: {optimal['vs_top1_pct']:+.1f}%")

    # IG 감소율 분석
    print("\n[IG 감소 패턴]")
    for i in range(1, 10):
        ig_i = position_ig[i].mean_ig
        ig_next = position_ig[i + 1].mean_ig
        if ig_i > 0:
            decay = (ig_i - ig_next) / ig_i * 100
            print(f"  Position {i} → {i+1}: IG {decay:+.1f}% 감소")

    # ==========================================================================
    # 결론
    # ==========================================================================
    print("\n" + "=" * 70)
    print("결론")
    print("=" * 70)

    print(f"""
[학술적 근거]

1. Information Gain 분포:
   - Position 1의 평균 IG: {position_ig[1].mean_ig:.4f}
   - Position 10의 평균 IG: {position_ig[10].mean_ig:.4f}
   - IG 감소율: {(1 - position_ig[10].mean_ig/position_ig[1].mean_ig)*100:.1f}%

2. LLM 선택 분포:
   - 총 분석 샘플: {total_samples}
   - Top-1 선택 확률: {llm_distribution.get(1, 0):.1%}
   - Top-3 내 선택 확률: {sum(llm_distribution.get(i, 0) for i in range(1,4)):.1%}

3. Expected Information Gain:
   - E[IG | Top-1] = {results[0]['expected_ig']:.4f}
   - E[IG | Top-{optimal['n']}] = {optimal['expected_ig']:.4f} (최적)
   - E[IG | Top-10] = {results[9]['expected_ig']:.4f}

4. 최적 Top-N:
   - N* = {optimal['n']}
   - 이론적 근거: E[IG] 최대화
""")

    analyzer.close()

    # JSON 저장
    output = {
        "n_cases": n_cases,
        "position_ig": {pos: {"mean_ig": data.mean_ig, "mean_coverage": data.mean_coverage, "samples": len(data.ig_values)}
                       for pos, data in position_ig.items()},
        "llm_distribution": llm_distribution,
        "total_llm_samples": total_samples,
        "expected_ig": {r["n"]: r["expected_ig"] for r in results},
        "optimal_n": optimal["n"],
        "optimal_expected_ig": optimal["expected_ig"],
    }

    output_file = Path("results/optimal_topn_analysis.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"결과 저장: {output_file}")

    return output


if __name__ == "__main__":
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_analysis(n_cases=n_cases)
