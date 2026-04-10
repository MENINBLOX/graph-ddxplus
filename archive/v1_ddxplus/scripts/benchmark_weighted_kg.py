#!/usr/bin/env python3
"""가중치 KG 벤치마크: 엣지 weight를 scoring에 반영.

Usage:
    uv run python scripts/benchmark_weighted_kg.py -n 1000 --port 7688
"""
import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from src.data_loader import DDXPlusLoader, Patient


class WeightedKGBenchmark:
    def __init__(self, port=7688, max_il=50, min_il=13,
                 confidence_threshold=0.30, gap_threshold=0.005,
                 relative_gap_threshold=1.5, denied_filter=5):
        self.max_il = max_il
        self.min_il = min_il
        self.confidence_threshold = confidence_threshold
        self.gap_threshold = gap_threshold
        self.relative_gap_threshold = relative_gap_threshold
        self.denied_filter = denied_filter

        self.driver = GraphDatabase.driver(
            f"bolt://localhost:{port}", auth=("neo4j", "password123"))
        self.loader = DDXPlusLoader()
        self._cui_to_codes = self.loader.build_cui_to_codes()

        self._ddx_disease_cuis = []
        self._cui_to_disease = {}
        for name, info in self.loader.disease_mapping.items():
            cui = info.get("umls_cui")
            if cui:
                cond = self.loader.conditions.get(name)
                if cond:
                    self._ddx_disease_cuis.append(cui)
                    self._cui_to_disease[cui] = cond.name_fr

        self._ddx_sym_cuis = list(self._cui_to_codes.keys())

    def get_candidate_symptoms(self, confirmed, denied, asked, limit=10):
        """가중치 기반 증상 탐색: weight 높은 엣지를 따라가는 증상 우선."""
        if len(confirmed) <= 1:
            query = """
            MATCH (c:Symptom)-[r:INDICATES]->(d:Disease)
            WHERE c.cui IN $confirmed AND d.cui IN $ddx_cuis
            WITH DISTINCT d, sum(r.weight) AS d_weight
            MATCH (d)<-[r2:INDICATES]-(next:Symptom)
            WHERE NOT next.cui IN $asked AND next.cui IN $sym_cuis
            WITH next, count(DISTINCT d) AS coverage,
                 sum(r2.weight * d_weight) AS weighted_score,
                 CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority
            RETURN next.cui AS cui, next.name AS name, coverage
            ORDER BY priority ASC, weighted_score DESC
            LIMIT $limit
            """
        else:
            query = """
            MATCH (c:Symptom)-[r:INDICATES]->(d:Disease)
            WHERE c.cui IN $confirmed AND d.cui IN $ddx_cuis
            WITH DISTINCT d, sum(r.weight) AS d_weight

            OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
            WHERE denied.cui IN $denied
            WITH d, d_weight, count(DISTINCT denied) AS denied_count
            WHERE denied_count < $denied_filter

            WITH collect({d: d, w: d_weight}) AS valid
            WHERE size(valid) > 0
            UNWIND valid AS v
            WITH v.d AS d, v.w AS d_weight

            MATCH (d)<-[r2:INDICATES]-(next:Symptom)
            WHERE NOT next.cui IN $confirmed
              AND NOT next.cui IN $denied
              AND NOT next.cui IN $asked
              AND next.cui IN $sym_cuis

            WITH next, d, d_weight, r2.weight AS edge_weight
            MATCH (d)<-[:INDICATES]-(conf:Symptom)
            WHERE conf.cui IN $confirmed
            WITH next, count(DISTINCT d) AS coverage,
                 sum(edge_weight * d_weight) AS weighted_score,
                 count(DISTINCT conf) AS cooccur,
                 CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority
            RETURN next.cui AS cui, next.name AS name, coverage
            ORDER BY priority ASC, weighted_score DESC
            LIMIT $limit
            """
        with self.driver.session() as s:
            result = s.run(query,
                confirmed=list(confirmed), denied=list(denied),
                asked=list(asked), ddx_cuis=self._ddx_disease_cuis,
                sym_cuis=self._ddx_sym_cuis, denied_filter=self.denied_filter,
                limit=limit)
            return [(r["cui"], r["name"], r["coverage"]) for r in result]

    def get_diagnosis(self, confirmed, denied, top_k=10):
        """가중치 기반 진단: edge weight를 scoring에 반영."""
        if not confirmed:
            return []
        query = """
        MATCH (c:Symptom)-[r:INDICATES]->(d:Disease)
        WHERE c.cui IN $confirmed AND d.cui IN $ddx_cuis
        WITH DISTINCT d
        OPTIONAL MATCH (d)<-[r2:INDICATES]-(s:Symptom)
        WITH d,
             count(DISTINCT s) AS total_symptoms,
             sum(CASE WHEN s.cui IN $confirmed THEN r2.weight ELSE 0 END) AS confirmed_weight,
             count(DISTINCT CASE WHEN s.cui IN $confirmed THEN s END) AS confirmed_count,
             sum(CASE WHEN s.cui IN $denied THEN r2.weight ELSE 0 END) AS denied_weight,
             count(DISTINCT CASE WHEN s.cui IN $denied THEN s END) AS denied_count
        WHERE confirmed_count > 0
        WITH d, confirmed_count, denied_count, confirmed_weight, denied_weight, total_symptoms,
             toFloat(confirmed_weight) /
             (toFloat(confirmed_weight) + toFloat(denied_weight) + 1.0) *
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
               CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score
        ORDER BY score DESC
        LIMIT $top_k
        """
        with self.driver.session() as s:
            result = s.run(query,
                confirmed=list(confirmed), denied=list(denied),
                ddx_cuis=self._ddx_disease_cuis, top_k=top_k)
            return [(r["cui"], r["name"], r["score"]) for r in result]

    def should_stop(self, confirmed, denied, asked_count):
        if asked_count >= self.max_il:
            return True
        diag = self.get_diagnosis(confirmed, denied, top_k=2)
        if not diag:
            return True
        if asked_count < self.min_il:
            return False
        if len(diag) == 1:
            return True
        t1, t2 = diag[0][2], diag[1][2] if len(diag) > 1 else 0.0
        if t1 >= self.confidence_threshold:
            return True
        if t1 - t2 >= self.gap_threshold:
            return True
        if t2 > 0 and t1 / t2 >= self.relative_gap_threshold:
            return True
        return False

    def check_evidence(self, patient, codes):
        evidences = set(patient.evidences)
        for code in codes:
            if code in evidences:
                return True
            for ev in evidences:
                if ev.startswith(f"{code}_@_"):
                    return True
        return False

    def run(self, n_samples=1000, severity=None):
        patients = self.loader.load_patients(split="test", n_samples=n_samples, severity=severity)
        correct = total = total_il = 0
        start = time.time()

        for patient in tqdm(patients, desc="Benchmark", unit="pt"):
            init_cui = None
            for cui, codes in self._cui_to_codes.items():
                if patient.initial_evidence in codes:
                    init_cui = cui
                    break
            if not init_cui:
                continue

            confirmed, denied, asked = {init_cui}, set(), {init_cui}
            il = 1

            for _ in range(self.max_il):
                if self.should_stop(confirmed, denied, len(asked)):
                    break
                candidates = self.get_candidate_symptoms(confirmed, denied, asked)
                if not candidates:
                    break
                sel_cui = candidates[0][0]
                codes = self._cui_to_codes.get(sel_cui, [])
                if self.check_evidence(patient, set(codes)):
                    confirmed.add(sel_cui)
                else:
                    denied.add(sel_cui)
                asked.add(sel_cui)
                il += 1

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

        elapsed = time.time() - start
        gtpa = correct / total if total > 0 else 0
        avg_il = total_il / total if total > 0 else 0
        print(f"\nGTPA@1:   {gtpa:.2%}")
        print(f"Avg IL:   {avg_il:.1f}")
        print(f"Time:     {elapsed:.0f}s")
        return {"gtpa": gtpa, "avg_il": avg_il}

    def close(self):
        self.driver.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=1000)
    p.add_argument("--port", type=int, default=7688)
    p.add_argument("--min-il", type=int, default=13)
    p.add_argument("--gap", type=float, default=0.005)
    p.add_argument("--confidence", type=float, default=0.30)
    p.add_argument("--denied-filter", type=int, default=5)
    args = p.parse_args()

    b = WeightedKGBenchmark(port=args.port, min_il=args.min_il,
                            gap_threshold=args.gap,
                            confidence_threshold=args.confidence,
                            denied_filter=args.denied_filter)
    b.run(args.n)
    b.close()


if __name__ == "__main__":
    main()
