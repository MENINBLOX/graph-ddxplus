#!/usr/bin/env python3
"""실패 케이스 분석: 왜 82.2%에서 멈추는가?"""

import ast
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class StoppingConfig:
    """조기 종료 설정."""
    min_il: int = 2
    confidence_threshold: float = 0.35
    gap_threshold: float = 0.10
    relative_gap_threshold: float = 1.8


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


class FailureAnalyzer:
    """실패 케이스 분석기."""

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

    def simulate_case_detailed(
        self,
        case_idx: int,
        config: StoppingConfig,
        max_il: int = 50,
    ) -> dict:
        """상세 시뮬레이션 (디버그 정보 포함)."""
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
        stop_reason = ""
        trajectory = []  # 각 스텝의 상태 기록

        while il < max_il:
            candidates = self._run_query(
                DIAGNOSIS_QUERY,
                confirmed_cuis=list(confirmed_cuis),
                denied_cuis=list(denied_cuis),
                top_k=10,
            )

            if not candidates:
                stop_reason = "no_candidates"
                break

            top1_score = candidates[0]["score"]
            top2_score = candidates[1]["score"] if len(candidates) > 1 else 0
            top1_cui = candidates[0]["cui"]

            # 현재 상태 기록
            trajectory.append({
                "il": il,
                "top1_cui": top1_cui,
                "top1_name": candidates[0]["name"],
                "top1_score": top1_score,
                "top2_score": top2_score,
                "gt_rank": next((i+1 for i, c in enumerate(candidates) if c["cui"] == gt_cui), -1),
                "confirmed": len(confirmed_cuis),
                "denied": len(denied_cuis),
            })

            # 조기 종료 체크
            if il >= config.min_il:
                if len(candidates) == 1:
                    stop_reason = "single_disease"
                    break
                if top1_score >= config.confidence_threshold:
                    stop_reason = f"confidence ({top1_score:.3f})"
                    break
                if top1_score - top2_score >= config.gap_threshold:
                    stop_reason = f"gap ({top1_score:.3f}-{top2_score:.3f})"
                    break
                if top2_score > 0 and top1_score / top2_score >= config.relative_gap_threshold:
                    stop_reason = f"ratio ({top1_score/top2_score:.2f})"
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
                stop_reason = "no_next_questions"
                break

            next_cui = next_candidates[0]["cui"]
            asked_cuis.add(next_cui)

            if next_cui in all_valid_cuis:
                confirmed_cuis.add(next_cui)
            else:
                denied_cuis.add(next_cui)

            il += 1

        if not stop_reason:
            stop_reason = f"max_il ({il})"

        # 최종 결과
        final_candidates = self._run_query(
            DIAGNOSIS_QUERY,
            confirmed_cuis=list(confirmed_cuis),
            denied_cuis=list(denied_cuis),
            top_k=10,
        )

        if not final_candidates:
            return {
                "case_idx": case_idx,
                "correct": False,
                "il": il,
                "gt_rank": -1,
                "stop_reason": stop_reason,
                "trajectory": trajectory,
                "gt_pathology": gt_pathology,
                "predicted": None,
                "total_patient_symptoms": len(all_valid_cuis),
            }

        predicted_cui = final_candidates[0]["cui"]
        predicted_name = final_candidates[0]["name"]
        correct = (predicted_cui == gt_cui)

        gt_rank = -1
        for i, c in enumerate(final_candidates):
            if c["cui"] == gt_cui:
                gt_rank = i + 1
                break

        return {
            "case_idx": case_idx,
            "correct": correct,
            "il": il,
            "gt_rank": gt_rank,
            "stop_reason": stop_reason,
            "trajectory": trajectory,
            "gt_pathology": gt_pathology,
            "gt_cui": gt_cui,
            "predicted": predicted_name,
            "predicted_cui": predicted_cui,
            "total_patient_symptoms": len(all_valid_cuis),
            "confirmed_count": len(confirmed_cuis),
            "denied_count": len(denied_cuis),
            "final_top1_score": final_candidates[0]["score"],
            "final_top2_score": final_candidates[1]["score"] if len(final_candidates) > 1 else 0,
        }

    def _run_query(self, query: str, **params) -> list[dict]:
        with self.driver.session() as session:
            try:
                result = session.run(query, **params)
                return [dict(r) for r in result]
            except Exception:
                return []


def main(n_cases: int = 200):
    """실패 케이스 분석."""
    analyzer = FailureAnalyzer()
    config = StoppingConfig()

    print(f"Analyzing {n_cases} cases...\n")

    results = []
    failures = []
    near_misses = []  # gt_rank 2-3인 케이스

    for i in range(n_cases):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_cases}...")

        result = analyzer.simulate_case_detailed(i, config)
        results.append(result)

        if not result["correct"]:
            failures.append(result)

        if result["gt_rank"] in [2, 3]:
            near_misses.append(result)

    analyzer.close()

    # 기본 통계
    n = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    gtpa_at_1 = correct_count / n
    avg_il = sum(r["il"] for r in results) / n

    print("\n" + "=" * 70)
    print("기본 통계")
    print("=" * 70)
    print(f"GTPA@1: {gtpa_at_1:.1%} ({correct_count}/{n})")
    print(f"Avg IL: {avg_il:.1f}")
    print(f"실패 케이스: {len(failures)}")
    print(f"Near Miss (2-3위): {len(near_misses)}")

    # 실패 원인 분석
    print("\n" + "=" * 70)
    print("실패 케이스 분석")
    print("=" * 70)

    # 1. GT rank 분포
    gt_rank_dist = Counter(r["gt_rank"] for r in failures)
    print("\n[GT Rank 분포]")
    for rank, count in sorted(gt_rank_dist.items()):
        pct = count / len(failures) * 100
        print(f"  Rank {rank}: {count} ({pct:.1f}%)")

    # 2. Stop reason 분포
    stop_reason_dist = Counter(r["stop_reason"].split()[0] for r in failures)
    print("\n[Stop Reason 분포]")
    for reason, count in sorted(stop_reason_dist.items(), key=lambda x: -x[1]):
        pct = count / len(failures) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")

    # 3. IL 분포 (실패 케이스)
    fail_ils = [r["il"] for r in failures]
    print(f"\n[실패 케이스 IL]")
    print(f"  평균: {sum(fail_ils)/len(fail_ils):.1f}")
    print(f"  중앙값: {sorted(fail_ils)[len(fail_ils)//2]}")

    # 4. 질환별 실패율
    pathology_failures = Counter(r["gt_pathology"] for r in failures)
    pathology_totals = Counter(r["gt_pathology"] for r in results)
    print("\n[질환별 실패율 (Top 10)]")
    failure_rates = []
    for pathology, fail_count in pathology_failures.items():
        total = pathology_totals[pathology]
        rate = fail_count / total
        failure_rates.append((pathology, fail_count, total, rate))

    for pathology, fail_count, total, rate in sorted(failure_rates, key=lambda x: -x[3])[:10]:
        print(f"  {pathology[:40]:<40} {fail_count}/{total} ({rate:.0%})")

    # 5. Near miss 분석 (개선 가능성 높음)
    print("\n" + "=" * 70)
    print("Near Miss 분석 (2-3위)")
    print("=" * 70)
    print(f"케이스 수: {len(near_misses)}")

    if near_misses:
        avg_score_gap = sum(
            r["final_top1_score"] - r["final_top2_score"]
            for r in near_misses if r["gt_rank"] == 2
        ) / max(1, sum(1 for r in near_misses if r["gt_rank"] == 2))
        print(f"GT=2위인 경우 평균 점수 차: {avg_score_gap:.4f}")

        # Near miss에서 더 질문했으면 개선됐을지 확인
        confirmed_ratio = sum(r["confirmed_count"] for r in near_misses) / sum(r["total_patient_symptoms"] for r in near_misses)
        print(f"Near miss 증상 확인 비율: {confirmed_ratio:.1%}")

    # 6. 개선 포인트 제안
    print("\n" + "=" * 70)
    print("개선 포인트")
    print("=" * 70)

    # 6a. 조기 종료 문제
    early_stop_failures = [r for r in failures if r["il"] < 15 and r["gt_rank"] in [2, 3]]
    print(f"\n[조기 종료로 인한 실패 (IL<15, GT rank 2-3)]")
    print(f"  케이스 수: {len(early_stop_failures)}")
    if early_stop_failures:
        print(f"  → 종료 조건을 늦추면 개선 가능")

    # 6b. 스코어링 문제
    scoring_failures = [r for r in failures if r["gt_rank"] == 2]
    print(f"\n[스코어링 문제 (GT가 2위)]")
    print(f"  케이스 수: {len(scoring_failures)}")
    if scoring_failures:
        print(f"  → 스코어링 공식 개선으로 해결 가능")

    # 6c. 질문 선택 문제 (denied가 너무 많은 경우)
    high_denied = [r for r in failures if r.get("denied_count", 0) > 10]
    print(f"\n[질문 선택 문제 (denied > 10)]")
    print(f"  케이스 수: {len(high_denied)}")
    if high_denied:
        print(f"  → 질문 선택 전략 개선 필요")

    # 결과 저장
    output = {
        "n_cases": n_cases,
        "gtpa_at_1": gtpa_at_1,
        "avg_il": avg_il,
        "failure_count": len(failures),
        "near_miss_count": len(near_misses),
        "gt_rank_distribution": dict(gt_rank_dist),
        "stop_reason_distribution": dict(stop_reason_dist),
        "pathology_failure_rates": [
            {"pathology": p, "failures": f, "total": t, "rate": r}
            for p, f, t, r in sorted(failure_rates, key=lambda x: -x[3])[:20]
        ],
        "improvement_opportunities": {
            "early_stop_failures": len(early_stop_failures),
            "scoring_failures": len(scoring_failures),
            "high_denied_failures": len(high_denied),
        },
    }

    output_file = Path("results/failure_analysis.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n결과 저장: {output_file}")


if __name__ == "__main__":
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(n_cases)
