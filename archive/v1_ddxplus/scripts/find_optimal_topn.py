#!/usr/bin/env python3
"""Top-N 최적값 탐색: LLM에게 몇 개의 증상을 제시해야 하는가?

실험 설계:
- KG-only 시뮬레이션에서 Top-N 중 랜덤 선택
- N=1,2,3,5,7,10으로 변화시키며 성능 측정
- 각 N에 대해 여러 번 반복하여 평균 계산

목표:
- IL이 급격히 증가하지 않으면서 정확도 유지되는 최적 N 찾기
- LLM 프롬프트에서 제시할 증상 수 결정 근거 마련
"""

import ast
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TopNResult:
    """Top-N 실험 결과."""
    n: int
    trial: int
    case_idx: int
    correct: bool
    il: int
    gt_rank: int


# 진단 쿼리 (v15_ratio)
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

# 질문 선택 쿼리 (Information Gain)
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


class TopNExperiment:
    """Top-N 실험."""

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
            disease_data = json.load(f)
            self.disease_map = disease_data["mapping"]

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

    def simulate_case_topn(
        self,
        case_idx: int,
        top_n: int,
        max_il: int = 25,
        min_il: int = 3,
        stop_confidence: float = 0.3,
        stop_gap: float = 0.10,
    ) -> TopNResult:
        """Top-N 중 랜덤 선택으로 케이스 시뮬레이션."""
        row = self.df[self.df["CATEGORY"] == 2].iloc[case_idx]

        gt_pathology = row["PATHOLOGY"]
        gt_cui = self.get_disease_cui(gt_pathology)

        # 환자의 모든 증상
        evidences_raw = ast.literal_eval(row["EVIDENCES"])
        all_symptom_cuis = set()
        for ev in evidences_raw:
            cui = self.get_symptom_cui(ev)
            if cui:
                all_symptom_cuis.add(cui)

        initial_cui = self.get_symptom_cui(row["INITIAL_EVIDENCE"])
        if initial_cui:
            all_symptom_cuis.add(initial_cui)

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

            top_score = candidates[0]["score"]
            top2_score = candidates[1]["score"] if len(candidates) > 1 else 0

            # 조기 종료
            if il >= min_il:
                if top_score >= stop_confidence:
                    break
                if top_score - top2_score >= stop_gap:
                    break

            # 다음 질문 후보
            next_candidates = self._run_query(
                QUESTION_QUERY,
                confirmed_cuis=list(confirmed_cuis),
                denied_cuis=list(denied_cuis),
                asked_cuis=list(asked_cuis),
                limit=max(top_n, 10),  # 최소 10개 가져옴
            )

            if not next_candidates:
                break

            # Top-N 중 랜덤 선택 (핵심!)
            n_available = min(top_n, len(next_candidates))
            selected_idx = random.randint(0, n_available - 1)
            next_cui = next_candidates[selected_idx]["cui"]

            asked_cuis.add(next_cui)

            if next_cui in all_symptom_cuis:
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

        predicted_cui = final_candidates[0]["cui"] if final_candidates else None
        correct = (predicted_cui == gt_cui)

        gt_rank = -1
        for i, c in enumerate(final_candidates):
            if c["cui"] == gt_cui:
                gt_rank = i + 1
                break

        return TopNResult(
            n=top_n,
            trial=0,
            case_idx=case_idx,
            correct=correct,
            il=il,
            gt_rank=gt_rank,
        )

    def _run_query(self, query: str, **params) -> list[dict]:
        with self.driver.session() as session:
            try:
                result = session.run(query, **params)
                return [dict(r) for r in result]
            except Exception as e:
                print(f"Query error: {e}")
                return []


def run_experiment(n_cases: int = 100, n_trials: int = 5):
    """Top-N 실험 실행."""
    exp = TopNExperiment()

    print("=" * 70)
    print(f"Top-N 최적값 탐색 (cases={n_cases}, trials={n_trials})")
    print("=" * 70)

    # 테스트할 N 값들
    n_values = [1, 2, 3, 5, 7, 10]

    results = {n: [] for n in n_values}

    for n in n_values:
        print(f"\nTesting Top-{n}...")

        for trial in range(n_trials):
            random.seed(42 + trial)  # 재현성을 위한 시드

            trial_correct = 0
            trial_il = 0

            for case_idx in range(n_cases):
                result = exp.simulate_case_topn(case_idx, top_n=n)
                results[n].append(result)

                if result.correct:
                    trial_correct += 1
                trial_il += result.il

            trial_acc = trial_correct / n_cases
            trial_avg_il = trial_il / n_cases
            print(f"  Trial {trial + 1}: GTPA@1={trial_acc:.1%}, Avg IL={trial_avg_il:.1f}")

    exp.close()

    # 결과 집계
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"{'Top-N':<8} {'GTPA@1':>10} {'Std':>8} {'Avg IL':>10} {'Std':>8} {'Top-5 Acc':>10}")
    print("-" * 70)

    summary = []
    for n in n_values:
        n_results = results[n]

        # Trial 별로 그룹화
        trials_acc = []
        trials_il = []
        trials_top5 = []

        for trial in range(n_trials):
            trial_results = [r for r in n_results if r.trial == trial or True]  # 모든 결과
            # 실제로는 trial 정보가 없으므로 전체를 n_trials로 나눔
            start_idx = trial * n_cases
            end_idx = start_idx + n_cases
            trial_subset = n_results[start_idx:end_idx]

            if trial_subset:
                acc = sum(1 for r in trial_subset if r.correct) / len(trial_subset)
                avg_il = sum(r.il for r in trial_subset) / len(trial_subset)
                top5 = sum(1 for r in trial_subset if 1 <= r.gt_rank <= 5) / len(trial_subset)

                trials_acc.append(acc)
                trials_il.append(avg_il)
                trials_top5.append(top5)

        if trials_acc:
            mean_acc = sum(trials_acc) / len(trials_acc)
            std_acc = (sum((x - mean_acc) ** 2 for x in trials_acc) / len(trials_acc)) ** 0.5

            mean_il = sum(trials_il) / len(trials_il)
            std_il = (sum((x - mean_il) ** 2 for x in trials_il) / len(trials_il)) ** 0.5

            mean_top5 = sum(trials_top5) / len(trials_top5)

            print(f"Top-{n:<5} {mean_acc:>9.1%} {std_acc:>7.1%} {mean_il:>9.1f} {std_il:>7.1f} {mean_top5:>9.1%}")

            summary.append({
                "n": n,
                "gtpa1_mean": mean_acc,
                "gtpa1_std": std_acc,
                "il_mean": mean_il,
                "il_std": std_il,
                "top5_mean": mean_top5,
            })

    # 최적 N 찾기
    print("\n" + "=" * 70)
    print("분석")
    print("=" * 70)

    if len(summary) >= 2:
        baseline = summary[0]  # Top-1
        print(f"\nBaseline (Top-1): GTPA@1={baseline['gtpa1_mean']:.1%}, IL={baseline['il_mean']:.1f}")

        print("\nTop-1 대비 변화:")
        for s in summary[1:]:
            acc_delta = s["gtpa1_mean"] - baseline["gtpa1_mean"]
            il_delta = s["il_mean"] - baseline["il_mean"]
            il_ratio = s["il_mean"] / baseline["il_mean"]

            status = "✅" if acc_delta >= -0.05 and il_ratio <= 1.5 else "⚠️"
            print(f"  Top-{s['n']}: GTPA@1 {acc_delta:+.1%}, IL {il_delta:+.1f} ({il_ratio:.2f}x) {status}")

    # JSON 저장
    output_file = Path("results/topn_experiment.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n결과 저장: {output_file}")

    return summary


if __name__ == "__main__":
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_experiment(n_cases=n_cases, n_trials=n_trials)
