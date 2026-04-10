"""조기 종료 케이스 기반 Cypher 쿼리 최적화.

조기 종료되는 100개 케이스를 사용하여 denied_count 필터와
기타 Cypher 파라미터를 최적화합니다.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from neo4j import GraphDatabase

# Load early termination cases
CASES_FILE = Path("/home/max/Graph-DDXPlus/results/early_termination_cases.json")


@dataclass
class CypherConfig:
    """Cypher 쿼리 설정."""

    denied_threshold: int | None  # None = no filter
    min_coverage: int  # minimum disease coverage for symptoms
    use_ratio_filter: bool  # denied/total ratio instead of absolute
    denied_ratio: float  # if use_ratio_filter, max denied ratio

    def __str__(self) -> str:
        if self.denied_threshold is None:
            denied_str = "no_filter"
        elif self.use_ratio_filter:
            denied_str = f"ratio<{self.denied_ratio}"
        else:
            denied_str = f"denied<{self.denied_threshold}"
        return f"denied={denied_str}_cov>={self.min_coverage}"


class CypherOptimizer:
    """Cypher 최적화기."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password123",
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.cases = self._load_cases()

    def _load_cases(self) -> list[dict]:
        """조기 종료 케이스 로드."""
        with open(CASES_FILE) as f:
            return json.load(f)

    def close(self) -> None:
        self.driver.close()

    def get_accumulated_candidates(
        self,
        config: CypherConfig,
        confirmed_cuis: list[str],
        denied_cuis: list[str],
        asked_cuis: list[str],
        limit: int = 10,
    ) -> list[dict]:
        """설정에 따른 후보 증상 쿼리."""

        if config.denied_threshold is None:
            # No denied filter
            denied_filter = ""
        elif config.use_ratio_filter:
            # Ratio-based filter
            denied_filter = f"""
            WITH d, candidate_diseases, count(denied) AS denied_count
            OPTIONAL MATCH (d)<-[:INDICATES]-(all_s:Symptom)
            WITH d, candidate_diseases, denied_count, count(DISTINCT all_s) AS total_symptoms
            WHERE total_symptoms = 0 OR toFloat(denied_count) / toFloat(total_symptoms) < {config.denied_ratio}
            """
        else:
            # Absolute count filter
            denied_filter = f"""
            WITH d, candidate_diseases, count(denied) AS denied_count
            WITH d, candidate_diseases
            WHERE denied_count < {config.denied_threshold}
            """

        query = f"""
        // Step 1: confirmed 증상과 연결된 후보 질환 수집
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH collect(DISTINCT d) AS candidate_diseases

        // Step 2: 각 후보 질환에 대해 denied 패널티 계산
        UNWIND candidate_diseases AS d
        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        {denied_filter}

        // Step 3: 유효한 후보 질환들
        WITH collect(d) AS valid_diseases
        WHERE size(valid_diseases) > 0

        // Step 4: 다음 질문 후보 증상 탐색
        UNWIND valid_diseases AS d
        MATCH (d)<-[:INDICATES]-(next:Symptom)
        WHERE NOT next.cui IN $confirmed_cuis
          AND NOT next.cui IN $denied_cuis
          AND NOT next.cui IN $asked_cuis
        WITH DISTINCT next, valid_diseases

        // Step 5: Information Gain 계산
        WITH next, valid_diseases,
             size([vd IN valid_diseases WHERE (next)-[:INDICATES]->(vd)]) AS coverage,
             size(valid_diseases) AS total

        WHERE coverage >= {config.min_coverage}

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

        with self.driver.session() as session:
            result = session.run(
                query,
                confirmed_cuis=confirmed_cuis,
                denied_cuis=denied_cuis,
                asked_cuis=asked_cuis,
                limit=limit,
            )
            return [dict(r) for r in result]

    def simulate_case(
        self,
        case: dict,
        config: CypherConfig,
        max_il: int = 25,
    ) -> dict:
        """단일 케이스 시뮬레이션."""

        # Get initial state from case
        interactions = case.get('interactions', [])
        if not interactions:
            return {'il': 0, 'candidates_exhausted': True}

        # Extract confirmed/denied from interactions
        confirmed_cuis = set()
        denied_cuis = set()
        asked_cuis = set()

        # Get initial symptom
        first_interaction = interactions[0]
        initial_confirmed = first_interaction.get('kg_input_confirmed', [])
        for cui in initial_confirmed:
            confirmed_cuis.add(cui)
            asked_cuis.add(cui)

        il = 1
        candidates_exhausted = False

        # Simulate interactions
        for interaction in interactions[1:]:
            if il >= max_il:
                break

            # Get candidates with this config
            candidates = self.get_accumulated_candidates(
                config=config,
                confirmed_cuis=list(confirmed_cuis),
                denied_cuis=list(denied_cuis),
                asked_cuis=list(asked_cuis),
            )

            if not candidates:
                candidates_exhausted = True
                break

            # Simulate patient response (from original interaction)
            # Find what symptom was asked and what was the response
            new_confirmed = set(interaction.get('kg_input_confirmed', [])) - confirmed_cuis
            new_denied = set(interaction.get('kg_input_denied', [])) - denied_cuis

            if new_confirmed:
                for cui in new_confirmed:
                    confirmed_cuis.add(cui)
                    asked_cuis.add(cui)
            if new_denied:
                for cui in new_denied:
                    denied_cuis.add(cui)
                    asked_cuis.add(cui)

            il += 1

        return {
            'il': il,
            'candidates_exhausted': candidates_exhausted,
            'final_confirmed': len(confirmed_cuis),
            'final_denied': len(denied_cuis),
        }

    def evaluate_config(self, config: CypherConfig) -> dict:
        """설정 평가."""

        results = []
        exhausted_count = 0

        for case in self.cases:
            result = self.simulate_case(case, config, max_il=25)
            results.append(result)
            if result['candidates_exhausted']:
                exhausted_count += 1

        avg_il = sum(r['il'] for r in results) / len(results)

        return {
            'config': str(config),
            'avg_il': avg_il,
            'exhausted_ratio': exhausted_count / len(results),
            'exhausted_count': exhausted_count,
            'total_cases': len(results),
        }

    def optimize(self) -> list[dict]:
        """다양한 설정 테스트."""

        configs = [
            # Original config
            CypherConfig(denied_threshold=3, min_coverage=1, use_ratio_filter=False, denied_ratio=0.5),

            # Relaxed denied threshold
            CypherConfig(denied_threshold=5, min_coverage=1, use_ratio_filter=False, denied_ratio=0.5),
            CypherConfig(denied_threshold=7, min_coverage=1, use_ratio_filter=False, denied_ratio=0.5),
            CypherConfig(denied_threshold=10, min_coverage=1, use_ratio_filter=False, denied_ratio=0.5),

            # No denied filter
            CypherConfig(denied_threshold=None, min_coverage=1, use_ratio_filter=False, denied_ratio=0.5),

            # Ratio-based filter
            CypherConfig(denied_threshold=None, min_coverage=1, use_ratio_filter=True, denied_ratio=0.3),
            CypherConfig(denied_threshold=None, min_coverage=1, use_ratio_filter=True, denied_ratio=0.5),
            CypherConfig(denied_threshold=None, min_coverage=1, use_ratio_filter=True, denied_ratio=0.7),

            # Higher min coverage
            CypherConfig(denied_threshold=5, min_coverage=2, use_ratio_filter=False, denied_ratio=0.5),
            CypherConfig(denied_threshold=None, min_coverage=2, use_ratio_filter=False, denied_ratio=0.5),
        ]

        results = []

        print(f"Testing {len(configs)} configurations on {len(self.cases)} early termination cases...")
        print()

        for i, config in enumerate(configs):
            print(f"[{i+1}/{len(configs)}] Testing: {config}")
            start = time.time()
            result = self.evaluate_config(config)
            elapsed = time.time() - start
            result['elapsed_sec'] = elapsed
            results.append(result)

            print(f"  → Avg IL: {result['avg_il']:.1f}, Exhausted: {result['exhausted_count']}/{result['total_cases']} ({result['exhausted_ratio']*100:.1f}%)")
            print()

        # Sort by lowest exhausted ratio, then by highest avg_il
        results.sort(key=lambda x: (x['exhausted_ratio'], -x['avg_il']))

        return results


def main():
    print("=" * 60)
    print("Cypher Query Optimization for Early Termination Cases")
    print("=" * 60)
    print()

    optimizer = CypherOptimizer()

    try:
        results = optimizer.optimize()

        print()
        print("=" * 60)
        print("RESULTS (sorted by exhausted ratio, then avg IL)")
        print("=" * 60)
        print()

        for i, r in enumerate(results):
            marker = "★" if i == 0 else " "
            print(f"{marker} [{i+1}] {r['config']}")
            print(f"    Avg IL: {r['avg_il']:.1f}")
            print(f"    Exhausted: {r['exhausted_count']}/{r['total_cases']} ({r['exhausted_ratio']*100:.1f}%)")
            print()

        # Save results
        output_file = Path("/home/max/Graph-DDXPlus/results/cypher_early_stop_optimization.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")

    finally:
        optimizer.close()


if __name__ == "__main__":
    main()
