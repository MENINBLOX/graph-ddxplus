#!/usr/bin/env python3
"""최종 검증 v2: 더 많은 설정 테스트."""

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

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


class StoppingValidator:
    """조기 종료 검증기."""

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

    def simulate_case(
        self,
        case_idx: int,
        config: StoppingConfig,
        max_il: int = 50,
    ) -> tuple[bool, int, int]:
        """단일 케이스 시뮬레이션."""
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

        while il < max_il:
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

            if il >= config.min_il:
                if len(candidates) == 1:
                    break
                if top1_score >= config.confidence_threshold:
                    break
                if top1_score - top2_score >= config.gap_threshold:
                    break
                if top2_score > 0 and top1_score / top2_score >= config.relative_gap_threshold:
                    break

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

    def validate(
        self,
        config: StoppingConfig,
        n_cases: int,
    ) -> dict:
        """설정 검증."""
        results = []

        for case_idx in range(n_cases):
            if (case_idx + 1) % 200 == 0:
                print(f"    {case_idx + 1}/{n_cases}...")

            correct, il, gt_rank = self.simulate_case(case_idx, config)
            results.append({
                "correct": correct,
                "il": il,
                "gt_rank": gt_rank,
            })

        n = len(results)
        gtpa_at_1 = sum(1 for r in results if r["correct"]) / n
        top_3_acc = sum(1 for r in results if 1 <= r["gt_rank"] <= 3) / n
        top_5_acc = sum(1 for r in results if 1 <= r["gt_rank"] <= 5) / n

        ils = [r["il"] for r in results]
        avg_il = sum(ils) / n
        median_il = sorted(ils)[n // 2]

        return {
            "config": str(config),
            "gtpa_at_1": gtpa_at_1,
            "top_3_acc": top_3_acc,
            "top_5_acc": top_5_acc,
            "avg_il": avg_il,
            "median_il": median_il,
        }

    def _run_query(self, query: str, **params) -> list[dict]:
        with self.driver.session() as session:
            try:
                result = session.run(query, **params)
                return [dict(r) for r in result]
            except Exception:
                return []


def main(n_cases: int = 1000):
    """검증 실행."""
    validator = StoppingValidator()

    print(f"Validating with {n_cases} cases...\n")

    # 더 공격적인 설정 테스트 (IL ≈ 15 타겟)
    configs = [
        # confidence 낮춤 (더 빠른 종료)
        StoppingConfig(2, 0.30, 0.08, 1.5),  # 매우 공격적
        StoppingConfig(2, 0.32, 0.08, 1.5),
        StoppingConfig(2, 0.33, 0.10, 1.8),
        StoppingConfig(3, 0.33, 0.10, 1.8),
        # 기존 최적
        StoppingConfig(2, 0.35, 0.10, 1.8),
        StoppingConfig(3, 0.35, 0.10, 1.8),
    ]

    all_results = []

    for config in configs:
        print(f"Testing: {config}")
        result = validator.validate(config, n_cases)
        all_results.append(result)
        print(f"  GTPA@1={result['gtpa_at_1']:.1%} IL={result['avg_il']:.1f} (median={result['median_il']})\n")

    validator.close()

    # 결과 정렬 (IL ≈ 15 + GTPA@1 최대화)
    all_results.sort(key=lambda r: (abs(r["avg_il"] - 15), -r["gtpa_at_1"]))

    print("=" * 70)
    print("결과 (IL ≈ 15 순)")
    print("=" * 70)
    print(f"{'Config':<25} {'GTPA@1':>8} {'Top-5':>8} {'Avg IL':>8} {'Median':>8}")
    print("-" * 65)

    for r in all_results:
        marker = " ✓" if abs(r["avg_il"] - 15) <= 1 else ""
        print(f"{r['config']:<25} {r['gtpa_at_1']:>7.1%} {r['top_5_acc']:>7.1%} {r['avg_il']:>7.1f} {r['median_il']:>7}{marker}")

    # 저장
    output = {"n_cases": n_cases, "results": all_results}
    output_file = Path("results/stopping_validation_v2.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n저장: {output_file}")


if __name__ == "__main__":
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(n_cases)
