#!/usr/bin/env python3
"""KG 증상 제안의 Hit Rate 분석 v2.

추가 분석:
1. Top-N 내 최소 1개 유효 증상 존재 확률
2. 첫 번째 유효 증상이 나타나는 위치 분포
3. 라운드별 Hit Rate 변화
"""

import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))


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


class HitRateAnalyzerV2:
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

    def analyze_case(self, case_idx: int, max_rounds: int = 20) -> list[dict]:
        """단일 케이스 분석."""
        row = self.df[self.df["CATEGORY"] == 2].iloc[case_idx]

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

        results = []

        for round_num in range(1, max_rounds + 1):
            candidates = self._run_query(
                QUESTION_QUERY,
                confirmed_cuis=list(confirmed_cuis),
                denied_cuis=list(denied_cuis),
                asked_cuis=list(asked_cuis),
                limit=10,
            )

            if not candidates:
                break

            kg_suggestions = [c["cui"] for c in candidates]

            # 각 위치에서 유효한지 체크
            valid_mask = [cui in all_valid_cuis for cui in kg_suggestions]

            # 첫 번째 유효 증상의 위치 (1-indexed, 없으면 -1)
            first_valid_pos = -1
            for i, is_valid in enumerate(valid_mask):
                if is_valid:
                    first_valid_pos = i + 1
                    break

            # Top-N별 분석
            analysis = {
                "case_idx": case_idx,
                "round": round_num,
                "n_suggestions": len(kg_suggestions),
                "n_valid": sum(valid_mask),
                "first_valid_pos": first_valid_pos,
                "valid_mask": valid_mask,
            }

            # Top-N 내 최소 1개 존재 여부
            for n in range(1, 11):
                if n <= len(valid_mask):
                    analysis[f"has_valid_in_top{n}"] = any(valid_mask[:n])
                    analysis[f"n_valid_in_top{n}"] = sum(valid_mask[:n])

            results.append(analysis)

            # 다음 라운드: Top-1 선택
            next_cui = kg_suggestions[0]
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
    """분석 실행."""
    analyzer = HitRateAnalyzerV2()

    print("=" * 70)
    print(f"KG Hit Rate 분석 v2 (cases={n_cases})")
    print("=" * 70)

    all_results = []

    for case_idx in range(n_cases):
        if (case_idx + 1) % 20 == 0:
            print(f"  Processing case {case_idx + 1}/{n_cases}...")

        results = analyzer.analyze_case(case_idx)
        all_results.extend(results)

    analyzer.close()

    # 1. Top-N 내 최소 1개 유효 증상 존재 확률
    print("\n" + "=" * 70)
    print("1. Top-N 내 최소 1개 유효 증상 존재 확률")
    print("=" * 70)
    print(f"{'Top-N':<8} {'존재 확률':>12} {'평균 유효 개수':>15}")
    print("-" * 40)

    for n in range(1, 11):
        key = f"has_valid_in_top{n}"
        count_key = f"n_valid_in_top{n}"

        has_valid = [r[key] for r in all_results if key in r]
        n_valid = [r[count_key] for r in all_results if count_key in r]

        if has_valid:
            prob = sum(has_valid) / len(has_valid)
            avg_count = sum(n_valid) / len(n_valid) if n_valid else 0
            print(f"Top-{n:<5} {prob:>11.1%} {avg_count:>14.2f}개")

    # 2. 첫 번째 유효 증상 위치 분포
    print("\n" + "=" * 70)
    print("2. 첫 번째 유효 증상 위치 분포")
    print("=" * 70)

    first_positions = [r["first_valid_pos"] for r in all_results]
    position_counts = defaultdict(int)
    for pos in first_positions:
        position_counts[pos] += 1

    total = len(first_positions)
    cumulative = 0

    print(f"{'위치':<8} {'빈도':>10} {'비율':>10} {'누적':>10}")
    print("-" * 45)

    for pos in sorted(position_counts.keys()):
        if pos == -1:
            label = "없음"
        else:
            label = f"{pos}번째"
        count = position_counts[pos]
        pct = count / total
        cumulative += pct if pos != -1 else 0
        print(f"{label:<8} {count:>10} {pct:>9.1%} {cumulative:>9.1%}")

    # 3. 라운드별 Hit Rate 변화
    print("\n" + "=" * 70)
    print("3. 라운드별 Top-1 Hit Rate 변화")
    print("=" * 70)

    round_hit = defaultdict(list)
    for r in all_results:
        if r["valid_mask"]:
            round_hit[r["round"]].append(r["valid_mask"][0])

    print(f"{'라운드':<8} {'Top-1 Hit Rate':>15} {'샘플 수':>10}")
    print("-" * 40)

    for rnd in sorted(round_hit.keys())[:15]:
        hits = round_hit[rnd]
        hit_rate = sum(hits) / len(hits)
        print(f"Round {rnd:<3} {hit_rate:>14.1%} {len(hits):>10}")

    # 4. 결론
    print("\n" + "=" * 70)
    print("결론")
    print("=" * 70)

    # Top-N 별 "최소 1개 유효" 확률 계산
    probs = {}
    for n in range(1, 11):
        key = f"has_valid_in_top{n}"
        has_valid = [r[key] for r in all_results if key in r]
        if has_valid:
            probs[n] = sum(has_valid) / len(has_valid)

    # 50% 이상 확률되는 최소 N
    n_for_50pct = 10
    for n in range(1, 11):
        if probs.get(n, 0) >= 0.5:
            n_for_50pct = n
            break

    # 80% 이상 확률되는 최소 N
    n_for_80pct = 10
    for n in range(1, 11):
        if probs.get(n, 0) >= 0.8:
            n_for_80pct = n
            break

    print(f"\n최소 1개 유효 증상 존재 확률:")
    print(f"  - 50% 이상 확률: Top-{n_for_50pct} ({probs.get(n_for_50pct, 0):.1%})")
    print(f"  - 80% 이상 확률: Top-{n_for_80pct} ({probs.get(n_for_80pct, 0):.1%})")

    no_valid_pct = position_counts.get(-1, 0) / total
    print(f"\n유효 증상이 아예 없는 라운드: {no_valid_pct:.1%}")

    # 추천
    print(f"\n[추천]")
    print(f"  - LLM에게 Top-{n_for_50pct} 제시: {probs.get(n_for_50pct, 0):.1%} 확률로 유효 증상 포함")
    print(f"  - Hit Rate 향상을 위해 질문 선택 전략 재검토 필요")

    # JSON 저장
    summary = {
        "n_cases": n_cases,
        "n_rounds": len(all_results),
        "has_valid_prob": probs,
        "first_position_dist": dict(position_counts),
        "n_for_50pct": n_for_50pct,
        "n_for_80pct": n_for_80pct,
    }

    output_file = Path("results/kg_hit_rate_v2.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n결과 저장: {output_file}")

    return summary


if __name__ == "__main__":
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_analysis(n_cases=n_cases)
