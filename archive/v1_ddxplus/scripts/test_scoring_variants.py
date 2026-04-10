#!/usr/bin/env python3
"""스코어링 공식 변형 테스트: GT가 2위인 케이스에서 어떤 공식이 1위로 올려주는가?"""

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))


# 다양한 스코어링 공식
SCORING_QUERIES = {
    # 현재 v15: confirmed/(confirmed+denied+1) × confirmed
    "v15_ratio": """
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
         toFloat(confirmed) / (toFloat(confirmed) + toFloat(denied) + 1.0) * toFloat(confirmed) AS raw_score
    WHERE raw_score > 0
    RETURN d.cui AS cui, d.name AS name, raw_score, confirmed, denied, total_symptoms
    ORDER BY raw_score DESC
    LIMIT $top_k
    """,

    # v16: confirmed² / (confirmed + denied + 1)
    "v16_squared": """
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
         toFloat(confirmed * confirmed) / (toFloat(confirmed) + toFloat(denied) + 1.0) AS raw_score
    WHERE raw_score > 0
    RETURN d.cui AS cui, d.name AS name, raw_score, confirmed, denied, total_symptoms
    ORDER BY raw_score DESC
    LIMIT $top_k
    """,

    # v17: confirmed - 0.3 × denied (denied 패널티 낮춤)
    "v17_low_penalty": """
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
         toFloat(confirmed) - 0.3 * toFloat(denied) AS raw_score
    WHERE raw_score > 0
    RETURN d.cui AS cui, d.name AS name, raw_score, confirmed, denied, total_symptoms
    ORDER BY raw_score DESC
    LIMIT $top_k
    """,

    # v18: confirmed / (total_symptoms) - coverage 기반
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

    # v19: confirmed × (1 - denied/total_symptoms)
    "v19_denied_ratio": """
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
         toFloat(confirmed) * (1.0 - toFloat(denied) / (toFloat(total_symptoms) + 1.0)) AS raw_score
    WHERE raw_score > 0
    RETURN d.cui AS cui, d.name AS name, raw_score, confirmed, denied, total_symptoms
    ORDER BY raw_score DESC
    LIMIT $top_k
    """,

    # v20: log(confirmed+1) / log(denied+2)
    "v20_log_ratio": """
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
         log(toFloat(confirmed) + 1.0) / log(toFloat(denied) + 2.0) AS raw_score
    WHERE raw_score > 0
    RETURN d.cui AS cui, d.name AS name, raw_score, confirmed, denied, total_symptoms
    ORDER BY raw_score DESC
    LIMIT $top_k
    """,

    # v21: confirmed² / (confirmed + 0.5×denied + 1)
    "v21_balanced": """
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
         toFloat(confirmed * confirmed) / (toFloat(confirmed) + 0.5 * toFloat(denied) + 1.0) AS raw_score
    WHERE raw_score > 0
    RETURN d.cui AS cui, d.name AS name, raw_score, confirmed, denied, total_symptoms
    ORDER BY raw_score DESC
    LIMIT $top_k
    """,

    # v22: confirmed / sqrt(denied + 1)
    "v22_sqrt_penalty": """
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
         toFloat(confirmed) / sqrt(toFloat(denied) + 1.0) AS raw_score
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


@dataclass
class StoppingConfig:
    min_il: int = 2
    confidence_threshold: float = 0.35
    gap_threshold: float = 0.10
    relative_gap_threshold: float = 1.8


class ScoringTester:
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

    def close(self):
        self.driver.close()

    def get_symptom_cui(self, code: str) -> str | None:
        base = code.split("_@_")[0] if "_@_" in code else code
        info = self.symptom_map.get(base, {})
        return info.get("cui")

    def get_disease_cui(self, name_fr: str) -> str | None:
        info = self.fr_to_info.get(name_fr, {})
        return info.get("umls_cui")

    def simulate_with_scoring(
        self,
        case_idx: int,
        scoring_name: str,
        config: StoppingConfig,
        max_il: int = 50,
    ) -> dict:
        """특정 스코어링으로 시뮬레이션."""
        row = self.severe_df.iloc[case_idx]

        gt_pathology = row["PATHOLOGY"]
        gt_cui = self.get_disease_cui(gt_pathology)

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
                top_k=10,
            )

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

            # 다음 질문 (baseline 유지)
            next_candidates = self._run_query(
                QUESTION_QUERY,
                confirmed_cuis=list(confirmed_cuis),
                denied_cuis=list(denied_cuis),
                asked_cuis=list(asked_cuis),
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
            top_k=10,
        )

        if not final_candidates:
            return {"correct": False, "il": il, "gt_rank": -1}

        predicted_cui = final_candidates[0]["cui"]
        correct = (predicted_cui == gt_cui)

        gt_rank = -1
        for i, c in enumerate(final_candidates):
            if c["cui"] == gt_cui:
                gt_rank = i + 1
                break

        return {"correct": correct, "il": il, "gt_rank": gt_rank}

    def _run_query(self, query: str, **params) -> list[dict]:
        with self.driver.session() as session:
            try:
                result = session.run(query, **params)
                return [dict(r) for r in result]
            except Exception as e:
                print(f"Query error: {e}")
                return []


def main(n_cases: int = 200):
    """스코어링 변형 테스트."""
    tester = ScoringTester()
    config = StoppingConfig()

    print(f"Testing {len(SCORING_QUERIES)} scoring variants on {n_cases} cases...\n")

    results = {}

    for scoring_name in SCORING_QUERIES:
        print(f"Testing: {scoring_name}...")

        case_results = []
        for i in range(n_cases):
            result = tester.simulate_with_scoring(i, scoring_name, config)
            case_results.append(result)

        n = len(case_results)
        gtpa_at_1 = sum(1 for r in case_results if r["correct"]) / n
        top_3 = sum(1 for r in case_results if 1 <= r["gt_rank"] <= 3) / n
        top_5 = sum(1 for r in case_results if 1 <= r["gt_rank"] <= 5) / n
        avg_il = sum(r["il"] for r in case_results) / n

        results[scoring_name] = {
            "gtpa_at_1": gtpa_at_1,
            "top_3": top_3,
            "top_5": top_5,
            "avg_il": avg_il,
        }

        print(f"  GTPA@1: {gtpa_at_1:.1%}, Top-3: {top_3:.1%}, IL: {avg_il:.1f}")

    tester.close()

    # 결과 정렬
    print("\n" + "=" * 70)
    print("스코어링 비교 (GTPA@1 순)")
    print("=" * 70)
    print(f"{'Scoring':<20} {'GTPA@1':>10} {'Top-3':>10} {'Top-5':>10} {'Avg IL':>10}")
    print("-" * 60)

    sorted_results = sorted(results.items(), key=lambda x: -x[1]["gtpa_at_1"])
    for name, r in sorted_results:
        marker = " ✓" if r["gtpa_at_1"] > results["v15_ratio"]["gtpa_at_1"] else ""
        print(f"{name:<20} {r['gtpa_at_1']:>9.1%} {r['top_3']:>9.1%} {r['top_5']:>9.1%} {r['avg_il']:>9.1f}{marker}")

    # 저장
    output_file = Path("results/scoring_variants.json")
    with open(output_file, "w") as f:
        json.dump({"n_cases": n_cases, "results": results}, f, indent=2)
    print(f"\n저장: {output_file}")


if __name__ == "__main__":
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(n_cases)
