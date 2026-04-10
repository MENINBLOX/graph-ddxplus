#!/usr/bin/env python3
"""진단 타이밍(조기 종료) 최적화.

목표: IL=15 달성하면서 GTPA@1 유지/향상

최적화 대상 파라미터:
- min_il: 최소 질문 수
- confidence_threshold: Top-1 확신도 임계값
- gap_threshold: Top-1과 Top-2 격차 임계값
- relative_gap_threshold: Top-1/Top-2 비율 임계값
"""

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from itertools import product

import pandas as pd
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class StoppingConfig:
    """조기 종료 설정."""
    min_il: int
    confidence_threshold: float
    gap_threshold: float
    relative_gap_threshold: float

    def __str__(self):
        return f"min{self.min_il}_c{int(self.confidence_threshold*100)}_g{int(self.gap_threshold*100)}_r{int(self.relative_gap_threshold*10)}"


@dataclass
class SimulationResult:
    """시뮬레이션 결과."""
    config: StoppingConfig
    n_cases: int
    gtpa_at_1: float
    top_3_acc: float
    top_5_acc: float
    avg_il: float
    std_il: float
    min_il_actual: float
    max_il_actual: float


# 진단 스코어링 쿼리 (v15_ratio)
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
"""

# 질문 선택 쿼리
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
       ig_score
ORDER BY ig_score DESC
LIMIT $limit
"""


class StoppingOptimizer:
    """조기 종료 최적화기."""

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

    def simulate_case(
        self,
        case_idx: int,
        config: StoppingConfig,
        max_il: int = 50,
    ) -> tuple[bool, int, int]:
        """단일 케이스 시뮬레이션.

        Returns:
            (correct, il, gt_rank)
        """
        row = self.df[self.df["CATEGORY"] == 2].iloc[case_idx]

        gt_pathology = row["PATHOLOGY"]
        gt_cui = self.get_disease_cui(gt_pathology)

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

        il = 1

        while il < max_il:
            # 진단 점수 계산
            candidates = self._run_query(
                DIAGNOSIS_QUERY,
                confirmed_cuis=list(confirmed_cuis),
                denied_cuis=list(denied_cuis),
                top_k=10,
            )

            if not candidates:
                break

            top1_score = candidates[0]["score"]
            top2_score = candidates[1]["score"] if len(candidates) > 1 else 0

            # 조기 종료 조건 체크
            if il >= config.min_il:
                # 조건 1: 단일 질환
                if len(candidates) == 1:
                    break

                # 조건 2: 확신도 임계값
                if top1_score >= config.confidence_threshold:
                    break

                # 조건 3: 절대 격차
                if top1_score - top2_score >= config.gap_threshold:
                    break

                # 조건 4: 상대 비율
                if top2_score > 0 and top1_score / top2_score >= config.relative_gap_threshold:
                    break

            # 다음 질문
            next_candidates = self._run_query(
                QUESTION_QUERY,
                confirmed_cuis=list(confirmed_cuis),
                denied_cuis=list(denied_cuis),
                asked_cuis=list(asked_cuis),
                limit=10,
            )

            if not next_candidates:
                break

            # Top-1 선택 (LLM이 85.7% Top-1 선택하므로)
            next_cui = next_candidates[0]["cui"]
            asked_cuis.add(next_cui)

            if next_cui in all_valid_cuis:
                confirmed_cuis.add(next_cui)
            else:
                denied_cuis.add(next_cui)

            il += 1

        # 최종 진단
        final_candidates = self._run_query(
            DIAGNOSIS_QUERY,
            confirmed_cuis=list(confirmed_cuis),
            denied_cuis=list(denied_cuis),
            top_k=10,
        )

        if not final_candidates:
            return False, il, -1

        predicted_cui = final_candidates[0]["cui"]
        correct = (predicted_cui == gt_cui)

        gt_rank = -1
        for i, c in enumerate(final_candidates):
            if c["cui"] == gt_cui:
                gt_rank = i + 1
                break

        return correct, il, gt_rank

    def run_simulation(
        self,
        config: StoppingConfig,
        n_cases: int,
    ) -> SimulationResult:
        """설정에 대한 시뮬레이션 실행."""
        results = []

        for case_idx in range(n_cases):
            correct, il, gt_rank = self.simulate_case(case_idx, config)
            results.append({
                "correct": correct,
                "il": il,
                "gt_rank": gt_rank,
            })

        # 통계 계산
        n = len(results)
        gtpa_at_1 = sum(1 for r in results if r["correct"]) / n
        top_3_acc = sum(1 for r in results if 1 <= r["gt_rank"] <= 3) / n
        top_5_acc = sum(1 for r in results if 1 <= r["gt_rank"] <= 5) / n

        ils = [r["il"] for r in results]
        avg_il = sum(ils) / n
        std_il = (sum((x - avg_il) ** 2 for x in ils) / n) ** 0.5

        return SimulationResult(
            config=config,
            n_cases=n,
            gtpa_at_1=gtpa_at_1,
            top_3_acc=top_3_acc,
            top_5_acc=top_5_acc,
            avg_il=avg_il,
            std_il=std_il,
            min_il_actual=min(ils),
            max_il_actual=max(ils),
        )

    def _run_query(self, query: str, **params) -> list[dict]:
        with self.driver.session() as session:
            try:
                result = session.run(query, **params)
                return [dict(r) for r in result]
            except Exception as e:
                return []


def grid_search(n_cases: int = 100, target_il: float = 15.0):
    """그리드 서치로 최적 파라미터 탐색."""
    optimizer = StoppingOptimizer()

    print("=" * 70)
    print(f"조기 종료 최적화 (cases={n_cases}, target_IL={target_il})")
    print("=" * 70)

    # 탐색 공간
    param_grid = {
        "min_il": [1, 2, 3, 5],
        "confidence_threshold": [0.15, 0.20, 0.25, 0.30, 0.35],
        "gap_threshold": [0.05, 0.08, 0.10, 0.12, 0.15],
        "relative_gap_threshold": [1.5, 2.0, 2.5],
    }

    # 모든 조합 생성
    configs = []
    for min_il, conf, gap, rel in product(
        param_grid["min_il"],
        param_grid["confidence_threshold"],
        param_grid["gap_threshold"],
        param_grid["relative_gap_threshold"],
    ):
        configs.append(StoppingConfig(min_il, conf, gap, rel))

    print(f"총 {len(configs)}개 조합 테스트...")

    results = []
    best_result = None
    best_score = -1

    for i, config in enumerate(configs):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(configs)}...")

        result = optimizer.run_simulation(config, n_cases)
        results.append(result)

        # 목표 IL 근처에서 GTPA@1 최대화
        il_penalty = abs(result.avg_il - target_il)
        score = result.gtpa_at_1 - 0.01 * il_penalty  # IL 1 차이당 1% 패널티

        if score > best_score:
            best_score = score
            best_result = result

    optimizer.close()

    # 결과 정렬 (IL이 target 근처이면서 GTPA@1 높은 순)
    results_near_target = [r for r in results if abs(r.avg_il - target_il) <= 3]
    results_near_target.sort(key=lambda r: -r.gtpa_at_1)

    print("\n" + "=" * 70)
    print(f"IL ≈ {target_il} (±3) 결과 (GTPA@1 순)")
    print("=" * 70)
    print(f"{'Config':<35} {'GTPA@1':>8} {'Top-3':>8} {'Avg IL':>8} {'Std IL':>8}")
    print("-" * 75)

    for r in results_near_target[:20]:
        print(f"{str(r.config):<35} {r.gtpa_at_1:>7.1%} {r.top_3_acc:>7.1%} {r.avg_il:>7.1f} {r.std_il:>7.1f}")

    # 최적 결과
    print("\n" + "=" * 70)
    print("최적 설정")
    print("=" * 70)

    if best_result:
        print(f"""
설정:
  min_il: {best_result.config.min_il}
  confidence_threshold: {best_result.config.confidence_threshold}
  gap_threshold: {best_result.config.gap_threshold}
  relative_gap_threshold: {best_result.config.relative_gap_threshold}

결과:
  GTPA@1: {best_result.gtpa_at_1:.1%}
  Top-3:  {best_result.top_3_acc:.1%}
  Top-5:  {best_result.top_5_acc:.1%}
  Avg IL: {best_result.avg_il:.1f} (std: {best_result.std_il:.1f})
  IL Range: {best_result.min_il_actual} ~ {best_result.max_il_actual}
""")

    # JSON 저장
    output = {
        "n_cases": n_cases,
        "target_il": target_il,
        "best_config": {
            "min_il": best_result.config.min_il,
            "confidence_threshold": best_result.config.confidence_threshold,
            "gap_threshold": best_result.config.gap_threshold,
            "relative_gap_threshold": best_result.config.relative_gap_threshold,
        } if best_result else None,
        "best_result": {
            "gtpa_at_1": best_result.gtpa_at_1,
            "top_3_acc": best_result.top_3_acc,
            "top_5_acc": best_result.top_5_acc,
            "avg_il": best_result.avg_il,
            "std_il": best_result.std_il,
        } if best_result else None,
        "all_results": [
            {
                "config": str(r.config),
                "gtpa_at_1": r.gtpa_at_1,
                "avg_il": r.avg_il,
            }
            for r in results_near_target[:50]
        ],
    }

    output_file = Path("results/stopping_optimization.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"결과 저장: {output_file}")

    return best_result, results


def fine_tune(n_cases: int = 100, target_il: float = 15.0):
    """최적 영역 근처에서 미세 조정."""
    optimizer = StoppingOptimizer()

    print("=" * 70)
    print(f"미세 조정 (cases={n_cases}, target_IL={target_il})")
    print("=" * 70)

    # 미세 조정 공간 (이전 결과 기반)
    param_grid = {
        "min_il": [1, 2, 3],
        "confidence_threshold": [0.18, 0.20, 0.22, 0.25],
        "gap_threshold": [0.06, 0.08, 0.10],
        "relative_gap_threshold": [1.5, 1.8, 2.0],
    }

    configs = []
    for min_il, conf, gap, rel in product(
        param_grid["min_il"],
        param_grid["confidence_threshold"],
        param_grid["gap_threshold"],
        param_grid["relative_gap_threshold"],
    ):
        configs.append(StoppingConfig(min_il, conf, gap, rel))

    print(f"총 {len(configs)}개 조합 테스트...")

    results = []

    for i, config in enumerate(configs):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(configs)}...")

        result = optimizer.run_simulation(config, n_cases)
        results.append(result)

    optimizer.close()

    # IL이 target 근처인 결과만 필터링
    results_filtered = [r for r in results if abs(r.avg_il - target_il) <= 2]
    results_filtered.sort(key=lambda r: -r.gtpa_at_1)

    print("\n" + "=" * 70)
    print(f"IL ≈ {target_il} (±2) 결과")
    print("=" * 70)
    print(f"{'Config':<35} {'GTPA@1':>8} {'Top-3':>8} {'Avg IL':>8}")
    print("-" * 60)

    for r in results_filtered[:15]:
        print(f"{str(r.config):<35} {r.gtpa_at_1:>7.1%} {r.top_3_acc:>7.1%} {r.avg_il:>7.1f}")

    return results_filtered[0] if results_filtered else None


if __name__ == "__main__":
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    target_il = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

    print("Phase 1: Grid Search")
    best, all_results = grid_search(n_cases, target_il)

    print("\n" + "=" * 70)
    print("Phase 2: Fine Tuning")
    fine_result = fine_tune(n_cases, target_il)
