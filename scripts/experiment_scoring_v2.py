#!/usr/bin/env python3
"""스코어링 + 종료조건 최적화 실험.

목표:
- IL: 15~20 (가능하면 15에 가깝게)
- Denied 패널티: 적당히 (너무 강하면 역효과)
- DDR 개선: min_prob cutoff 조정

Usage:
    uv run python scripts/experiment_scoring_v2.py --n-cases 1000
    uv run python scripts/experiment_scoring_v2.py --n-cases 1000 --full  # 전체 테스트
"""

import argparse
import ast
import json
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# 새로운 스코어링 전략들
# =============================================================================

SCORING_QUERIES = {
    # 기준선: v18_coverage (현재 최고 성능, denied 무시)
    "v18_coverage": """
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed,
         size(matched_denied) AS denied
    WITH d, confirmed, denied, total_symptoms,
         toFloat(confirmed) / (toFloat(total_symptoms) + 1.0) * toFloat(confirmed) AS raw_score
    WHERE raw_score > 0
    RETURN d.cui AS cui, d.name AS name, raw_score, confirmed, denied, total_symptoms
    ORDER BY raw_score DESC
    LIMIT $top_k
    """,

    # v23: v18 + 약한 denied ratio 패널티 (0.1)
    # score = (confirmed / (total + 1) × confirmed) × (1 - 0.1 × denied/total)
    "v23_mild_ratio": """
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed,
         size(matched_denied) AS denied
    WITH d, confirmed, denied, total_symptoms,
         (toFloat(confirmed) / (toFloat(total_symptoms) + 1.0) * toFloat(confirmed))
         * (1.0 - 0.1 * toFloat(denied) / (toFloat(total_symptoms) + 1.0)) AS raw_score
    WHERE raw_score > 0
    RETURN d.cui AS cui, d.name AS name, raw_score, confirmed, denied, total_symptoms
    ORDER BY raw_score DESC
    LIMIT $top_k
    """,

    # v24: v18 + 약한 denied ratio 패널티 (0.2)
    "v24_mild_ratio_02": """
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed,
         size(matched_denied) AS denied
    WITH d, confirmed, denied, total_symptoms,
         (toFloat(confirmed) / (toFloat(total_symptoms) + 1.0) * toFloat(confirmed))
         * (1.0 - 0.2 * toFloat(denied) / (toFloat(total_symptoms) + 1.0)) AS raw_score
    WHERE raw_score > 0
    RETURN d.cui AS cui, d.name AS name, raw_score, confirmed, denied, total_symptoms
    ORDER BY raw_score DESC
    LIMIT $top_k
    """,

    # v25: v18 - 약한 absolute denied 패널티
    # score = (confirmed / (total + 1) × confirmed) - 0.05 × denied
    "v25_mild_abs": """
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed,
         size(matched_denied) AS denied
    WITH d, confirmed, denied, total_symptoms,
         (toFloat(confirmed) / (toFloat(total_symptoms) + 1.0) * toFloat(confirmed))
         - 0.05 * toFloat(denied) AS raw_score
    WHERE raw_score > 0
    RETURN d.cui AS cui, d.name AS name, raw_score, confirmed, denied, total_symptoms
    ORDER BY raw_score DESC
    LIMIT $top_k
    """,

    # v26: v18 hybrid with v21 (balanced)
    # score = confirmed² / (total + 0.3×denied + 1)
    "v26_hybrid": """
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed,
         size(matched_denied) AS denied
    WITH d, confirmed, denied, total_symptoms,
         toFloat(confirmed * confirmed) / (toFloat(total_symptoms) + 0.3 * toFloat(denied) + 1.0) AS raw_score
    WHERE raw_score > 0
    RETURN d.cui AS cui, d.name AS name, raw_score, confirmed, denied, total_symptoms
    ORDER BY raw_score DESC
    LIMIT $top_k
    """,

    # v27: coverage × confirmed ratio
    # score = (confirmed/total) × (confirmed/(confirmed+denied+1))
    "v27_coverage_ratio": """
    MATCH (d:Disease)
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
         count(DISTINCT s) AS total_symptoms
    WITH d, disease_symptom_cuis, total_symptoms,
         [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
         [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied
    WHERE size(matched_confirmed) > 0
    WITH d, total_symptoms,
         size(matched_confirmed) AS confirmed,
         size(matched_denied) AS denied
    WITH d, confirmed, denied, total_symptoms,
         (toFloat(confirmed) / (toFloat(total_symptoms) + 1.0))
         * (toFloat(confirmed) / (toFloat(confirmed) + toFloat(denied) + 1.0))
         * toFloat(confirmed) AS raw_score
    WHERE raw_score > 0
    RETURN d.cui AS cui, d.name AS name, raw_score, confirmed, denied, total_symptoms
    ORDER BY raw_score DESC
    LIMIT $top_k
    """,
}

QUESTION_QUERY = """
MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
WHERE confirmed.cui IN $confirmed_cuis
WITH collect(DISTINCT d) AS candidate_diseases

UNWIND candidate_diseases AS d
OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
WHERE denied.cui IN $denied_cuis
WITH d, candidate_diseases, count(denied) AS denied_count
WITH d, candidate_diseases
WHERE denied_count < $denied_threshold

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


@dataclass
class StoppingConfig:
    """종료 조건 설정."""
    min_il: int = 3
    confidence_threshold: float = 0.25
    gap_threshold: float = 0.06
    relative_gap_threshold: float = 2.0
    denied_threshold: int = 5  # Cypher 쿼리용


# IL ≈ 15 타겟 설정들
STOPPING_CONFIGS = {
    # 현재 설정 (IL ≈ 21)
    "current": StoppingConfig(
        min_il=3, confidence_threshold=0.25, gap_threshold=0.06, relative_gap_threshold=2.0
    ),
    # 더 공격적 (IL ≈ 15 목표)
    "aggressive_1": StoppingConfig(
        min_il=2, confidence_threshold=0.20, gap_threshold=0.05, relative_gap_threshold=1.8
    ),
    "aggressive_2": StoppingConfig(
        min_il=2, confidence_threshold=0.18, gap_threshold=0.04, relative_gap_threshold=1.6
    ),
    "aggressive_3": StoppingConfig(
        min_il=3, confidence_threshold=0.18, gap_threshold=0.05, relative_gap_threshold=1.5
    ),
    # 중간 (IL ≈ 18 목표)
    "balanced_1": StoppingConfig(
        min_il=3, confidence_threshold=0.22, gap_threshold=0.05, relative_gap_threshold=1.8
    ),
    "balanced_2": StoppingConfig(
        min_il=2, confidence_threshold=0.22, gap_threshold=0.06, relative_gap_threshold=1.8
    ),
}


class ScoringExperiment:
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
        self.severe_df = self.df[self.df["CATEGORY"] == 2].reset_index(drop=True)

        with open("data/ddxplus/umls_mapping.json") as f:
            self.symptom_map = json.load(f)["mapping"]

        with open("data/ddxplus/disease_umls_mapping.json") as f:
            self.disease_map = json.load(f)["mapping"]

        self.fr_to_info = {}
        for eng, info in self.disease_map.items():
            fr_name = info.get("name_fr", "")
            if fr_name:
                self.fr_to_info[fr_name] = info

        # DDXPlus 질환 CUI 목록
        self.ddxplus_cuis = set()
        for info in self.disease_map.values():
            cui = info.get("umls_cui")
            if cui:
                self.ddxplus_cuis.add(cui)

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
        scoring_name: str,
        config: StoppingConfig,
        max_il: int = 50,
        min_prob: float = 0.02,  # DDR용 cutoff
    ) -> dict:
        """단일 케이스 시뮬레이션."""
        row = self.severe_df.iloc[case_idx]

        gt_pathology = row["PATHOLOGY"]
        gt_cui = self.get_disease_cui(gt_pathology)

        # Ground truth differential diagnosis
        gt_dd_raw = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
        gt_dd_cuis = set()
        for d, _ in gt_dd_raw:
            cui = self.get_disease_cui(d)
            if cui:
                gt_dd_cuis.add(cui)

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
        scoring_query = SCORING_QUERIES[scoring_name]

        while il < max_il:
            candidates = self._run_query(
                scoring_query,
                confirmed_cuis=list(confirmed_cuis),
                denied_cuis=list(denied_cuis),
                top_k=50,
            )

            if not candidates:
                break

            # DDXPlus 질환만 필터링
            candidates = [c for c in candidates if c["cui"] in self.ddxplus_cuis]
            if not candidates:
                break

            top1_score = candidates[0]["raw_score"]
            top2_score = candidates[1]["raw_score"] if len(candidates) > 1 else 0

            # 점수 정규화
            total_score = sum(c["raw_score"] for c in candidates)
            if total_score > 0:
                top1_norm = top1_score / total_score
                top2_norm = top2_score / total_score
            else:
                top1_norm = top2_norm = 0

            # 종료 조건 체크
            if il >= config.min_il:
                if len(candidates) == 1:
                    break
                if top1_norm >= config.confidence_threshold:
                    break
                if top1_norm - top2_norm >= config.gap_threshold:
                    break
                if top2_norm > 0 and top1_norm / top2_norm >= config.relative_gap_threshold:
                    break

            # 다음 질문
            next_candidates = self._run_query(
                QUESTION_QUERY,
                confirmed_cuis=list(confirmed_cuis),
                denied_cuis=list(denied_cuis),
                asked_cuis=list(asked_cuis),
                denied_threshold=config.denied_threshold,
                limit=10,
            )

            if not next_candidates:
                break

            next_cui = next_candidates[0]["cui"]
            asked_cuis.add(next_cui)

            if next_cui in all_valid_cuis:
                confirmed_cuis.add(next_cui)
            else:
                denied_cuis.add(next_cui)

            il += 1

        # 최종 진단
        final_candidates = self._run_query(
            scoring_query,
            confirmed_cuis=list(confirmed_cuis),
            denied_cuis=list(denied_cuis),
            top_k=100,
        )

        # DDXPlus 질환만 필터링
        final_candidates = [c for c in final_candidates if c["cui"] in self.ddxplus_cuis]

        if not final_candidates:
            return {
                "correct": False, "il": il, "gt_rank": -1,
                "ddr": 0, "ddp": 0, "n_pred": 0
            }

        # min_prob cutoff 적용
        total_score = sum(c["raw_score"] for c in final_candidates)
        if total_score > 0:
            filtered_candidates = [
                c for c in final_candidates
                if c["raw_score"] / total_score >= min_prob
            ]
            if not filtered_candidates:
                filtered_candidates = [final_candidates[0]]
        else:
            filtered_candidates = final_candidates

        predicted_cui = filtered_candidates[0]["cui"]
        correct = (predicted_cui == gt_cui)

        gt_rank = -1
        for i, c in enumerate(filtered_candidates):
            if c["cui"] == gt_cui:
                gt_rank = i + 1
                break

        # DDR/DDP 계산
        pred_dd_cuis = {c["cui"] for c in filtered_candidates}
        intersection = len(gt_dd_cuis & pred_dd_cuis)
        ddr = intersection / len(gt_dd_cuis) if gt_dd_cuis else 0
        ddp = intersection / len(pred_dd_cuis) if pred_dd_cuis else 0

        return {
            "correct": correct,
            "il": il,
            "gt_rank": gt_rank,
            "ddr": ddr,
            "ddp": ddp,
            "n_pred": len(pred_dd_cuis),
        }

    def _run_query(self, query: str, **params) -> list[dict]:
        with self.driver.session() as session:
            try:
                result = session.run(query, **params)
                return [dict(r) for r in result]
            except Exception as e:
                print(f"Query error: {e}")
                return []

    def run_experiment(
        self,
        n_cases: int,
        scoring_names: list[str] | None = None,
        config_names: list[str] | None = None,
        min_prob_values: list[float] | None = None,
    ) -> list[dict]:
        """실험 실행."""
        if scoring_names is None:
            scoring_names = list(SCORING_QUERIES.keys())
        if config_names is None:
            config_names = list(STOPPING_CONFIGS.keys())
        if min_prob_values is None:
            min_prob_values = [0.02]  # 현재 설정

        results = []
        total_combinations = len(scoring_names) * len(config_names) * len(min_prob_values)

        print(f"Running {total_combinations} combinations on {n_cases} cases...")
        print(f"Scorings: {scoring_names}")
        print(f"Configs: {config_names}")
        print(f"min_prob: {min_prob_values}")
        print("=" * 70)

        for scoring_name, config_name, min_prob in tqdm(
            list(product(scoring_names, config_names, min_prob_values)),
            desc="Combinations",
        ):
            config = STOPPING_CONFIGS[config_name]

            case_results = []
            for i in range(n_cases):
                result = self.simulate_case(i, scoring_name, config, min_prob=min_prob)
                case_results.append(result)

            n = len(case_results)
            gtpa_at_1 = sum(1 for r in case_results if r["correct"]) / n
            avg_il = sum(r["il"] for r in case_results) / n
            avg_ddr = sum(r["ddr"] for r in case_results) / n
            avg_ddp = sum(r["ddp"] for r in case_results) / n
            ddf1 = 2 * avg_ddr * avg_ddp / (avg_ddr + avg_ddp) if (avg_ddr + avg_ddp) > 0 else 0
            avg_n_pred = sum(r["n_pred"] for r in case_results) / n

            result_entry = {
                "scoring": scoring_name,
                "config": config_name,
                "min_prob": min_prob,
                "gtpa_at_1": gtpa_at_1,
                "avg_il": avg_il,
                "ddr": avg_ddr,
                "ddp": avg_ddp,
                "ddf1": ddf1,
                "avg_n_pred": avg_n_pred,
            }
            results.append(result_entry)

            tqdm.write(
                f"{scoring_name:20s} | {config_name:15s} | "
                f"GTPA@1={gtpa_at_1:.1%} IL={avg_il:.1f} DDR={avg_ddr:.1%} DDF1={ddf1:.1%}"
            )

        return results


def main():
    parser = argparse.ArgumentParser(description="Scoring + Stopping Optimization")
    parser.add_argument("--n-cases", type=int, default=500)
    parser.add_argument("--full", action="store_true", help="Run all combinations")
    parser.add_argument("--scoring", type=str, default=None, help="Specific scoring to test")
    parser.add_argument("--config", type=str, default=None, help="Specific config to test")

    args = parser.parse_args()

    exp = ScoringExperiment()

    try:
        if args.full:
            # 전체 조합 테스트
            results = exp.run_experiment(
                n_cases=args.n_cases,
                min_prob_values=[0.01, 0.02, 0.03],  # DDR 테스트
            )
        else:
            # 빠른 테스트: 새 스코어링 + 공격적 종료 조건
            scoring_names = (
                [args.scoring] if args.scoring else
                ["v18_coverage", "v23_mild_ratio", "v24_mild_ratio_02", "v26_hybrid", "v27_coverage_ratio"]
            )
            config_names = (
                [args.config] if args.config else
                ["current", "aggressive_1", "balanced_1"]
            )
            results = exp.run_experiment(
                n_cases=args.n_cases,
                scoring_names=scoring_names,
                config_names=config_names,
            )

        # 결과 정렬 (IL 15~20 범위에서 GTPA@1 최대화)
        print("\n" + "=" * 90)
        print("RESULTS (sorted by GTPA@1, IL 15-20 preferred)")
        print("=" * 90)
        print(f"{'Scoring':<22} {'Config':<15} {'GTPA@1':>8} {'IL':>6} {'DDR':>8} {'DDF1':>8} {'#Pred':>6}")
        print("-" * 90)

        # IL 15-20 범위 우선, 그 다음 GTPA@1 순
        def sort_key(r):
            in_range = 15 <= r["avg_il"] <= 20
            return (-int(in_range), -r["gtpa_at_1"])

        sorted_results = sorted(results, key=sort_key)

        for r in sorted_results:
            marker = "✓" if 15 <= r["avg_il"] <= 20 else ""
            print(
                f"{r['scoring']:<22} {r['config']:<15} "
                f"{r['gtpa_at_1']:>7.1%} {r['avg_il']:>6.1f} "
                f"{r['ddr']:>7.1%} {r['ddf1']:>7.1%} {r['avg_n_pred']:>6.1f} {marker}"
            )

        # Best in target range
        in_range = [r for r in results if 15 <= r["avg_il"] <= 20]
        if in_range:
            best = max(in_range, key=lambda r: r["gtpa_at_1"])
            print("\n" + "-" * 90)
            print(f"BEST (IL 15-20): {best['scoring']} + {best['config']}")
            print(f"  GTPA@1={best['gtpa_at_1']:.1%}, IL={best['avg_il']:.1f}, DDR={best['ddr']:.1%}, DDF1={best['ddf1']:.1%}")

        # 저장
        output_file = Path("results/scoring_v2_experiment.json")
        with open(output_file, "w") as f:
            json.dump({
                "n_cases": args.n_cases,
                "results": results,
            }, f, indent=2)
        print(f"\nSaved: {output_file}")

    finally:
        exp.close()


if __name__ == "__main__":
    main()
