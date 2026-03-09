"""조기 종료 케이스 기반 Cypher 쿼리 최적화 v2.

실제 환자 데이터로 직접 테스트합니다.
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase

from src.data_loader import DDXPlusLoader


@dataclass
class KGState:
    """KG 상태."""
    confirmed_cuis: set[str] = field(default_factory=set)
    denied_cuis: set[str] = field(default_factory=set)
    asked_cuis: set[str] = field(default_factory=set)


@dataclass
class CypherConfig:
    """Cypher 쿼리 설정."""
    name: str
    denied_threshold: int | None  # None = no filter
    use_ratio_filter: bool = False
    denied_ratio: float = 0.5


class CypherTester:
    """Cypher 테스터."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password123",
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.loader = DDXPlusLoader()

    def close(self) -> None:
        self.driver.close()

    def get_candidate_symptoms(
        self,
        config: CypherConfig,
        state: KGState,
        limit: int = 10,
    ) -> list[dict]:
        """후보 증상 쿼리."""

        if not state.confirmed_cuis:
            return []

        # Build denied filter based on config
        if config.denied_threshold is None:
            denied_filter = ""
        elif config.use_ratio_filter:
            denied_filter = f"""
            WITH d, candidate_diseases, count(denied) AS denied_count
            OPTIONAL MATCH (d)<-[:INDICATES]-(all_s:Symptom)
            WITH d, candidate_diseases, denied_count, count(DISTINCT all_s) AS total_symptoms
            WHERE total_symptoms = 0 OR toFloat(denied_count) / toFloat(total_symptoms) < {config.denied_ratio}
            """
        else:
            denied_filter = f"""
            WITH d, candidate_diseases, count(denied) AS denied_count
            WITH d, candidate_diseases
            WHERE denied_count < {config.denied_threshold}
            """

        query = f"""
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH collect(DISTINCT d) AS candidate_diseases

        UNWIND candidate_diseases AS d
        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        {denied_filter}

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

        RETURN next.cui AS cui, next.name AS name, coverage, ig_score
        ORDER BY ig_score DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                confirmed_cuis=list(state.confirmed_cuis),
                denied_cuis=list(state.denied_cuis),
                asked_cuis=list(state.asked_cuis),
                limit=limit,
            )
            return [dict(r) for r in result]

    def get_diagnosis(
        self,
        state: KGState,
        top_k: int = 5,
    ) -> list[dict]:
        """진단 후보 쿼리."""

        if not state.confirmed_cuis:
            return []

        query = """
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
             toFloat(confirmed_count) / (toFloat(total_symptoms) + 1.0) * toFloat(confirmed_count) AS raw_score
        WHERE raw_score > 0
        WITH collect({cui: d.cui, name: d.name, raw_score: raw_score, confirmed_count: confirmed_count, total_symptoms: total_symptoms}) AS all_candidates
        WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
        UNWIND all_candidates AS c
        RETURN c.cui AS cui, c.name AS name,
               CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
               c.confirmed_count AS confirmed_count, c.total_symptoms AS total_symptoms
        ORDER BY score DESC
        LIMIT $top_k
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                confirmed_cuis=list(state.confirmed_cuis),
                denied_cuis=list(state.denied_cuis),
                top_k=top_k,
            )
            return [dict(r) for r in result]

    def run_diagnosis(
        self,
        patient,
        config: CypherConfig,
        max_il: int = 25,
        confidence_threshold: float = 0.35,
        gap_threshold: float = 0.10,
    ) -> dict:
        """환자 진단 실행."""

        state = KGState()

        # Get initial symptom CUI
        initial_cui = self.loader.get_symptom_cui(patient.initial_evidence)
        if not initial_cui:
            return {'il': 0, 'correct': False, 'exhausted': True}

        state.confirmed_cuis.add(initial_cui)
        state.asked_cuis.add(initial_cui)

        # Build patient evidence lookup
        patient_evidence = {}
        for ev in patient.evidences:
            if '_@_' in ev:
                code, value = ev.split('_@_')
                patient_evidence[code] = value
            else:
                patient_evidence[ev] = 'True'

        il = 1

        while il < max_il:
            # Check stopping conditions
            candidates = self.get_diagnosis(state, top_k=2)
            if candidates:
                top1_score = candidates[0]['score']
                top2_score = candidates[1]['score'] if len(candidates) > 1 else 0.0

                if top1_score >= confidence_threshold:
                    break
                if top1_score - top2_score >= gap_threshold:
                    break

            # Get next symptom to ask
            symptom_candidates = self.get_candidate_symptoms(config, state)

            if not symptom_candidates:
                # No more candidates - exhausted
                break

            # Pick first candidate and check patient response
            next_cui = symptom_candidates[0]['cui']

            # Find the DDXPlus code for this CUI
            cui_to_codes = self.loader.build_cui_to_codes()
            codes = cui_to_codes.get(next_cui, [])

            # Check patient response
            confirmed = False
            for code in codes:
                if code in patient_evidence:
                    value = patient_evidence[code]
                    if value == 'True' or value not in ['False', 'N']:
                        confirmed = True
                        break

            if confirmed:
                state.confirmed_cuis.add(next_cui)
            else:
                state.denied_cuis.add(next_cui)
            state.asked_cuis.add(next_cui)

            il += 1

        # Final diagnosis
        final_candidates = self.get_diagnosis(state, top_k=1)
        if final_candidates:
            pred_cui = final_candidates[0]['cui']
            gt_cui = self.loader.get_pathology_cui(patient.pathology)
            correct = pred_cui == gt_cui
        else:
            correct = False

        exhausted = len(self.get_candidate_symptoms(config, state)) == 0

        return {
            'il': il,
            'correct': correct,
            'exhausted': exhausted,
            'final_confirmed': len(state.confirmed_cuis),
            'final_denied': len(state.denied_cuis),
        }

    def evaluate_config(
        self,
        config: CypherConfig,
        patients: list,
    ) -> dict:
        """설정 평가."""

        results = []
        correct_count = 0
        exhausted_count = 0

        for patient in patients:
            result = self.run_diagnosis(patient, config)
            results.append(result)

            if result['correct']:
                correct_count += 1
            if result['exhausted']:
                exhausted_count += 1

        avg_il = sum(r['il'] for r in results) / len(results)
        accuracy = correct_count / len(results)
        exhausted_ratio = exhausted_count / len(results)

        return {
            'config': config.name,
            'accuracy': accuracy,
            'avg_il': avg_il,
            'exhausted_ratio': exhausted_ratio,
            'exhausted_count': exhausted_count,
            'correct_count': correct_count,
            'total_cases': len(results),
        }


def main():
    print("=" * 60)
    print("Cypher Query Optimization v2 (Actual Patient Data)")
    print("=" * 60)
    print()

    tester = CypherTester()

    # Load patients (severity=2, first 500)
    patients = tester.loader.load_patients(
        split="validate",
        n_samples=500,
        severity=2,
    )
    print(f"Loaded {len(patients)} patients")
    print()

    # Test configurations
    configs = [
        CypherConfig("denied<3 (original)", denied_threshold=3),
        CypherConfig("denied<5", denied_threshold=5),
        CypherConfig("denied<7", denied_threshold=7),
        CypherConfig("denied<10", denied_threshold=10),
        CypherConfig("denied<15", denied_threshold=15),
        CypherConfig("no_filter", denied_threshold=None),
        CypherConfig("ratio<0.3", denied_threshold=None, use_ratio_filter=True, denied_ratio=0.3),
        CypherConfig("ratio<0.5", denied_threshold=None, use_ratio_filter=True, denied_ratio=0.5),
        CypherConfig("ratio<0.7", denied_threshold=None, use_ratio_filter=True, denied_ratio=0.7),
    ]

    results = []

    try:
        for i, config in enumerate(configs):
            print(f"[{i+1}/{len(configs)}] Testing: {config.name}")
            start = time.time()
            result = tester.evaluate_config(config, patients)
            elapsed = time.time() - start
            result['elapsed_sec'] = elapsed
            results.append(result)

            print(f"  → Accuracy: {result['accuracy']*100:.1f}%, Avg IL: {result['avg_il']:.1f}, Exhausted: {result['exhausted_ratio']*100:.1f}%")
            print()

        # Sort by accuracy, then by lowest exhausted ratio
        results.sort(key=lambda x: (-x['accuracy'], x['exhausted_ratio']))

        print()
        print("=" * 60)
        print("RESULTS (sorted by accuracy)")
        print("=" * 60)
        print()
        print(f"{'Config':<20} {'Accuracy':>10} {'Avg IL':>10} {'Exhausted':>12}")
        print("-" * 55)

        for r in results:
            marker = "★" if r == results[0] else " "
            print(f"{marker}{r['config']:<19} {r['accuracy']*100:>9.1f}% {r['avg_il']:>10.1f} {r['exhausted_ratio']*100:>11.1f}%")

        # Save results
        output_file = Path("/home/max/Graph-DDXPlus/results/cypher_optimization_v2.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print()
        print(f"Results saved to: {output_file}")

    finally:
        tester.close()


if __name__ == "__main__":
    main()
