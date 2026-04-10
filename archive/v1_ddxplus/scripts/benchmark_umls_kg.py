#!/usr/bin/env python3
"""UMLS KG 벤치마크: DDXPlus 49개 질환 스코핑.

UMLS KG의 Disease-Symptom 관계를 사용하되,
탐색/진단 대상을 DDXPlus 49개 질환으로 제한.

- 관계 데이터: UMLS MRREL (external knowledge)
- 탐색 범위: DDXPlus 49개 질환
- 진단 범위: DDXPlus 49개 질환

Usage:
    uv run python scripts/benchmark_umls_kg.py -n 200 --port 7688
    uv run python scripts/benchmark_umls_kg.py -n 1000 --port 7688 --scoring v15_ratio
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from src.data_loader import DDXPlusLoader, Patient


@dataclass
class PatientState:
    patient: Patient
    idx: int
    initial_cui: str | None = None
    confirmed_cuis: set = field(default_factory=set)
    denied_cuis: set = field(default_factory=set)
    asked_cuis: set = field(default_factory=set)
    il: int = 0
    predicted: str | None = None
    done: bool = False


class UMLSKGBenchmark:
    """UMLS KG 벤치마크 (DDXPlus 스코핑)."""

    def __init__(
        self,
        port: int = 7688,
        scoring: str = "v15_ratio",
        max_il: int = 50,
        min_il: int = 13,
        confidence_threshold: float = 0.30,
        gap_threshold: float = 0.005,
        relative_gap_threshold: float = 1.5,
        denied_filter: int = 5,
    ):
        self.scoring = scoring
        self.max_il = max_il
        self.min_il = min_il
        self.confidence_threshold = confidence_threshold
        self.gap_threshold = gap_threshold
        self.relative_gap_threshold = relative_gap_threshold
        self.denied_filter = denied_filter

        self.driver = GraphDatabase.driver(
            f"bolt://localhost:{port}", auth=("neo4j", "password123")
        )
        self.loader = DDXPlusLoader()
        self._cui_to_codes = self.loader.build_cui_to_codes()

        # DDXPlus disease CUI mappings
        self._ddx_disease_cuis: list[str] = []
        self._cui_to_disease: dict[str, str] = {}
        for name_eng, info in self.loader.disease_mapping.items():
            cui = info.get("umls_cui")
            if cui:
                cond = self.loader.conditions.get(name_eng)
                if cond:
                    self._ddx_disease_cuis.append(cui)
                    self._cui_to_disease[cui] = cond.name_fr

        # DDXPlus symptom CUIs (환자가 응답 가능한 증상만)
        self._ddx_symptom_cuis: list[str] = list(self._cui_to_codes.keys())

        print(f"Disease CUIs: {len(self._ddx_disease_cuis)}/49")
        print(f"Symptom CUIs: {len(self._ddx_symptom_cuis)} (DDXPlus mappable)")
        print(f"Scoring: {self.scoring}, max_il: {self.max_il}, min_il: {self.min_il}")
        print(f"gap: {self.gap_threshold}, confidence: {self.confidence_threshold}")

    def get_candidate_symptoms(
        self,
        confirmed_cuis: set[str],
        denied_cuis: set[str],
        asked_cuis: set[str],
        limit: int = 10,
    ) -> list[tuple[str, str, int]]:
        """DDXPlus 질환 스코핑된 2-hop 증상 탐색.

        Returns: [(cui, name, coverage), ...]
        """
        if len(confirmed_cuis) <= 1:
            # Initial: simple 2-hop, DDXPlus 증상만 제안
            query = """
            MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
            WHERE confirmed.cui IN $confirmed_cuis
              AND d.cui IN $ddx_cuis
            WITH DISTINCT d
            MATCH (d)<-[:INDICATES]-(next:Symptom)
            WHERE NOT next.cui IN $asked_cuis
              AND next.cui IN $ddx_sym_cuis
            WITH next, count(DISTINCT d) AS coverage,
                 CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority
            RETURN next.cui AS cui, next.name AS name, coverage
            ORDER BY priority ASC, coverage DESC
            LIMIT $limit
            """
        else:
            # Accumulated: co-occurrence, DDXPlus 증상만 제안
            query = """
            MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
            WHERE confirmed.cui IN $confirmed_cuis
              AND d.cui IN $ddx_cuis
            WITH DISTINCT d

            OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
            WHERE denied.cui IN $denied_cuis
            WITH d, count(DISTINCT denied) AS denied_count
            WHERE denied_count < $denied_filter

            WITH collect(DISTINCT d) AS valid_diseases
            WHERE size(valid_diseases) > 0

            UNWIND valid_diseases AS d
            MATCH (d)<-[:INDICATES]-(next:Symptom)
            WHERE NOT next.cui IN $confirmed_cuis
              AND NOT next.cui IN $denied_cuis
              AND NOT next.cui IN $asked_cuis
              AND next.cui IN $ddx_sym_cuis

            WITH next, d
            MATCH (d)<-[:INDICATES]-(conf:Symptom)
            WHERE conf.cui IN $confirmed_cuis
            WITH next, count(DISTINCT d) AS coverage, count(DISTINCT conf) AS cooccur,
                 CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority

            RETURN next.cui AS cui, next.name AS name, coverage
            ORDER BY priority ASC, toFloat(cooccur) * coverage DESC
            LIMIT $limit
            """

        with self.driver.session() as s:
            result = s.run(
                query,
                confirmed_cuis=list(confirmed_cuis),
                denied_cuis=list(denied_cuis),
                asked_cuis=list(asked_cuis),
                ddx_cuis=self._ddx_disease_cuis,
                ddx_sym_cuis=self._ddx_symptom_cuis,
                denied_filter=self.denied_filter,
                limit=limit,
            )
            return [(r["cui"], r["name"], r["coverage"]) for r in result]

    def get_diagnosis(
        self,
        confirmed_cuis: set[str],
        denied_cuis: set[str],
        top_k: int = 10,
    ) -> list[tuple[str, str, float]]:
        """DDXPlus 질환만 대상으로 진단 스코어링.

        Returns: [(cui, name, score), ...]
        """
        if not confirmed_cuis:
            return []

        if self.scoring == "v15_ratio":
            score_expr = """
                toFloat(confirmed_count) /
                (toFloat(confirmed_count) + toFloat(denied_count) + 1.0) *
                toFloat(confirmed_count)
            """
        elif self.scoring == "v23_mild_denied":
            score_expr = """
                (toFloat(confirmed_count) / (toFloat(total_symptoms) + 1.0) *
                 toFloat(confirmed_count)) *
                (1.0 - 0.1 * toFloat(denied_count) / (toFloat(total_symptoms) + 1.0))
            """
        else:
            score_expr = """
                toFloat(confirmed_count) / (toFloat(total_symptoms) + 1.0) *
                toFloat(confirmed_count)
            """

        query = f"""
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
          AND d.cui IN $ddx_cuis
        WITH DISTINCT d
        OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
        WITH d,
             count(DISTINCT s) AS total_symptoms,
             count(DISTINCT CASE WHEN s.cui IN $confirmed_cuis THEN s END) AS confirmed_count,
             count(DISTINCT CASE WHEN s.cui IN $denied_cuis THEN s END) AS denied_count
        WHERE confirmed_count > 0
        WITH d, confirmed_count, denied_count, total_symptoms,
             {score_expr} AS raw_score
        WHERE raw_score > 0
        WITH collect({{
            cui: d.cui, name: d.name, raw_score: raw_score,
            confirmed_count: confirmed_count, total_symptoms: total_symptoms
        }}) AS all_candidates
        WITH all_candidates,
             reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
        UNWIND all_candidates AS c
        RETURN c.cui AS cui, c.name AS name,
               CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score
        ORDER BY score DESC
        LIMIT $top_k
        """

        with self.driver.session() as s:
            result = s.run(
                query,
                confirmed_cuis=list(confirmed_cuis),
                denied_cuis=list(denied_cuis),
                ddx_cuis=self._ddx_disease_cuis,
                top_k=top_k,
            )
            return [(r["cui"], r["name"], r["score"]) for r in result]

    def should_stop(
        self,
        confirmed_cuis: set[str],
        denied_cuis: set[str],
        asked_count: int,
    ) -> tuple[bool, str]:
        """종료 조건 확인."""
        if asked_count >= self.max_il:
            return True, "max_il"

        diag = self.get_diagnosis(confirmed_cuis, denied_cuis, top_k=2)
        if not diag:
            return True, "no_candidates"

        if asked_count < self.min_il:
            return False, ""

        if len(diag) == 1:
            return True, "single"

        top1_score = diag[0][2]
        top2_score = diag[1][2] if len(diag) > 1 else 0.0

        if top1_score >= self.confidence_threshold:
            return True, "confidence"

        if top1_score - top2_score >= self.gap_threshold:
            return True, "gap"

        if top2_score > 0 and top1_score / top2_score >= self.relative_gap_threshold:
            return True, "ratio"

        return False, ""

    def check_evidence(self, patient: Patient, codes: set) -> bool:
        evidences = set(patient.evidences)
        for code in codes:
            if code in evidences:
                return True
            for ev in evidences:
                if ev.startswith(f"{code}_@_"):
                    return True
        return False

    def get_cui_for_code(self, code: str) -> str | None:
        for cui, codes in self._cui_to_codes.items():
            if code in codes:
                return cui
        return None

    def run(self, n_samples: int, severity: int | None = None) -> dict:
        patients = self.loader.load_patients(
            split="test", n_samples=n_samples, severity=severity
        )
        print(f"Loaded {len(patients):,} patients")

        start = time.time()
        correct = 0
        total = 0
        total_il = 0

        pbar = tqdm(patients, desc="Benchmark", unit="pt")
        for patient in pbar:
            init_cui = self.get_cui_for_code(patient.initial_evidence)
            if not init_cui:
                continue

            confirmed = {init_cui}
            denied: set[str] = set()
            asked = {init_cui}
            il = 1

            for _ in range(self.max_il):
                stop, reason = self.should_stop(confirmed, denied, len(asked))
                if stop:
                    break

                candidates = self.get_candidate_symptoms(
                    confirmed, denied, asked, limit=10
                )
                if not candidates:
                    break

                sel_cui, sel_name, _ = candidates[0]
                codes = self._cui_to_codes.get(sel_cui, [])
                has = self.check_evidence(patient, set(codes))

                asked.add(sel_cui)
                if has:
                    confirmed.add(sel_cui)
                else:
                    denied.add(sel_cui)
                il += 1

            # Diagnosis
            diag = self.get_diagnosis(confirmed, denied, top_k=49)
            predicted = None
            for cui, name, score in diag:
                if cui in self._cui_to_disease:
                    predicted = self._cui_to_disease[cui]
                    break
            if not predicted:
                predicted = list(self._cui_to_disease.values())[0]

            if predicted == patient.pathology:
                correct += 1
            total += 1
            total_il += il

            if total % 50 == 0:
                pbar.set_postfix({
                    "GTPA@1": f"{correct/total:.1%}",
                    "IL": f"{total_il/total:.1f}",
                })

        pbar.close()
        elapsed = time.time() - start

        gtpa = correct / total if total > 0 else 0
        avg_il = total_il / total if total > 0 else 0

        print(f"\n{'='*60}")
        print(f"UMLS KG BENCHMARK (DDXPlus scoped)")
        print(f"{'='*60}")
        print(f"Samples:  {total:,}")
        print(f"GTPA@1:   {gtpa:.2%}")
        print(f"Avg IL:   {avg_il:.1f}")
        print(f"Scoring:  {self.scoring}")
        print(f"Time:     {elapsed:.0f}s ({total/elapsed:.1f} pt/s)")
        print(f"{'='*60}")

        result = {
            "gtpa_at_1": gtpa,
            "avg_il": avg_il,
            "samples": total,
            "scoring": self.scoring,
            "min_il": self.min_il,
            "max_il": self.max_il,
            "gap_threshold": self.gap_threshold,
            "confidence_threshold": self.confidence_threshold,
            "denied_filter": self.denied_filter,
            "elapsed": elapsed,
        }

        output = Path("results/umls_kg_benchmark.json")
        output.parent.mkdir(exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {output}")

        return result

    def close(self):
        self.driver.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=200)
    parser.add_argument("--port", type=int, default=7688)
    parser.add_argument("--scoring", default="v15_ratio")
    parser.add_argument("--max-il", type=int, default=50)
    parser.add_argument("--min-il", type=int, default=13)
    parser.add_argument("--gap", type=float, default=0.005)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--denied-filter", type=int, default=5)
    parser.add_argument("--severity", type=int, default=None)
    args = parser.parse_args()

    bench = UMLSKGBenchmark(
        port=args.port,
        scoring=args.scoring,
        max_il=args.max_il,
        min_il=args.min_il,
        confidence_threshold=args.confidence,
        gap_threshold=args.gap,
        denied_filter=args.denied_filter,
    )
    bench.run(args.n, severity=args.severity)
    bench.close()


if __name__ == "__main__":
    main()
