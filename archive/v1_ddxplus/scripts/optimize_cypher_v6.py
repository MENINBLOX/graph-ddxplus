"""Cypher 쿼리 + Stopping 조건 종합 최적화 v6.

목표: IL ~20, Accuracy ~86%
"""

import json
import sys
import time
from dataclasses import dataclass, field
from itertools import product
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
class OptimConfig:
    """최적화 설정."""
    denied_threshold: int
    confidence_threshold: float
    gap_threshold: float
    relative_gap: float
    min_il: int

    def __str__(self) -> str:
        return f"denied<{self.denied_threshold}_conf>={self.confidence_threshold}_gap>={self.gap_threshold}_ratio>={self.relative_gap}_min_il={self.min_il}"


class CypherOptimizer:
    """Cypher + Stopping 최적화기."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password123",
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.loader = DDXPlusLoader()
        self._cui_to_codes = None

    def close(self) -> None:
        self.driver.close()

    @property
    def cui_to_codes(self):
        if self._cui_to_codes is None:
            self._cui_to_codes = self.loader.build_cui_to_codes()
        return self._cui_to_codes

    def get_candidate_symptoms(
        self,
        config: OptimConfig,
        state: KGState,
        limit: int = 10,
    ) -> list[dict]:
        """후보 증상 쿼리."""

        if not state.confirmed_cuis:
            return []

        query = f"""
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH collect(DISTINCT d) AS candidate_diseases

        UNWIND candidate_diseases AS d
        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        WITH d, candidate_diseases, count(denied) AS denied_count
        WITH d, candidate_diseases
        WHERE denied_count < {config.denied_threshold}

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

    def should_stop(
        self,
        config: OptimConfig,
        state: KGState,
        il: int,
        max_il: int = 50,
    ) -> tuple[bool, str]:
        """종료 조건 확인."""

        if il >= max_il:
            return True, "max_il"

        candidates = self.get_diagnosis(state, top_k=2)

        if not candidates:
            return True, "no_candidates"

        if il < config.min_il:
            return False, ""

        if len(candidates) == 1:
            return True, "single_disease"

        top1_score = candidates[0]['score']
        top2_score = candidates[1]['score'] if len(candidates) > 1 else 0.0

        if top1_score >= config.confidence_threshold:
            return True, "confidence"

        if top1_score - top2_score >= config.gap_threshold:
            return True, "gap"

        if top2_score > 0 and top1_score / top2_score >= config.relative_gap:
            return True, "ratio"

        return False, ""

    def run_diagnosis(
        self,
        patient,
        config: OptimConfig,
        max_il: int = 50,
    ) -> dict:
        """환자 진단 실행."""

        state = KGState()

        # Get initial symptom CUI
        initial_cui = self.loader.get_symptom_cui(patient.initial_evidence)
        if not initial_cui:
            return {'il': 0, 'correct': False, 'stop_reason': 'no_initial'}

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
        stop_reason = ""

        while il < max_il:
            # Check stopping conditions
            should_stop, reason = self.should_stop(config, state, il, max_il)
            if should_stop:
                stop_reason = reason
                break

            # Get next symptom to ask
            symptom_candidates = self.get_candidate_symptoms(config, state)

            if not symptom_candidates:
                stop_reason = "exhausted"
                break

            # Pick first candidate and check patient response
            next_cui = symptom_candidates[0]['cui']
            codes = self.cui_to_codes.get(next_cui, [])

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

        return {
            'il': il,
            'correct': correct,
            'stop_reason': stop_reason,
        }

    def evaluate_config(
        self,
        config: OptimConfig,
        patients: list,
    ) -> dict:
        """설정 평가."""

        results = []
        correct_count = 0
        stop_reasons = {}

        for patient in patients:
            result = self.run_diagnosis(patient, config)
            results.append(result)

            if result['correct']:
                correct_count += 1

            reason = result['stop_reason']
            stop_reasons[reason] = stop_reasons.get(reason, 0) + 1

        avg_il = sum(r['il'] for r in results) / len(results)
        accuracy = correct_count / len(results)
        exhausted = stop_reasons.get('exhausted', 0) / len(results)

        return {
            'config': str(config),
            'denied_threshold': config.denied_threshold,
            'confidence_threshold': config.confidence_threshold,
            'gap_threshold': config.gap_threshold,
            'relative_gap': config.relative_gap,
            'min_il': config.min_il,
            'accuracy': accuracy,
            'avg_il': avg_il,
            'exhausted_ratio': exhausted,
            'stop_reasons': stop_reasons,
            'total_cases': len(results),
        }


def main():
    print("=" * 70)
    print("Cypher + Stopping 종합 최적화 v6")
    print("목표: IL ~20, Accuracy ~86%")
    print("=" * 70)
    print()

    optimizer = CypherOptimizer()

    # Load patients (severity=2, offset 1000, count 1000)
    all_patients = optimizer.loader.load_patients(
        split="validate",
        n_samples=2000,
        severity=2,
    )
    patients = all_patients[1000:2000]
    print(f"Using patients 1000-2000: {len(patients)} cases")
    print()

    # Grid search - IL ~20, Acc ~86% 목표
    # v4 결과: conf>=0.28, gap>=0.06 → 88.5%, IL 21.2
    # 더 세밀한 조정 필요
    denied_thresholds = [5, 6]
    confidence_thresholds = [0.25, 0.28, 0.30, 0.32, 0.35]
    gap_thresholds = [0.06, 0.08, 0.10]
    relative_gaps = [1.8, 2.0]
    min_ils = [2, 3]

    configs = []
    for dt, ct, gt, rg, mi in product(
        denied_thresholds, confidence_thresholds, gap_thresholds, relative_gaps, min_ils
    ):
        configs.append(OptimConfig(
            denied_threshold=dt,
            confidence_threshold=ct,
            gap_threshold=gt,
            relative_gap=rg,
            min_il=mi,
        ))

    print(f"Testing {len(configs)} configurations...")
    print()

    results = []

    try:
        for i, config in enumerate(configs):
            print(f"[{i+1}/{len(configs)}] denied<{config.denied_threshold}_conf>={config.confidence_threshold}_gap>={config.gap_threshold}", end=" ", flush=True)
            start = time.time()
            result = optimizer.evaluate_config(config, patients)
            elapsed = time.time() - start
            result['elapsed_sec'] = elapsed
            results.append(result)

            print(f"→ Acc: {result['accuracy']*100:.1f}%, IL: {result['avg_il']:.1f}, Exh: {result['exhausted_ratio']*100:.1f}%")

        # Filter results with IL close to 20 (18-22 range) and sort by accuracy
        target_il_results = [r for r in results if 18 <= r['avg_il'] <= 22]
        target_il_results.sort(key=lambda x: -x['accuracy'])

        print()
        print("=" * 70)
        print("TOP 10 RESULTS (IL 18~22 범위, 정확도 순)")
        print("=" * 70)
        print()
        print(f"{'Denied':<8} {'Conf':<8} {'Gap':<8} {'Ratio':<8} {'MinIL':<6} {'Accuracy':>10} {'Avg IL':>10} {'Exhausted':>10}")
        print("-" * 80)

        for i, r in enumerate(target_il_results[:10]):
            marker = "★" if i == 0 else " "
            print(f"{marker} <{r['denied_threshold']:<5} >={r['confidence_threshold']:<5} >={r['gap_threshold']:<5} >={r['relative_gap']:<5} {r['min_il']:<6} {r['accuracy']*100:>9.1f}% {r['avg_il']:>10.1f} {r['exhausted_ratio']*100:>9.1f}%")

        if not target_il_results:
            print("No results in IL 18-22 range")
            # Show closest to IL 20
            by_il_diff = sorted(results, key=lambda x: abs(x['avg_il'] - 20))
            print()
            print("Closest to IL=20:")
            for r in by_il_diff[:5]:
                print(f"  denied<{r['denied_threshold']}, conf>={r['confidence_threshold']}, gap>={r['gap_threshold']} → Acc: {r['accuracy']*100:.1f}%, IL: {r['avg_il']:.1f}")

        # Filter for ~86% accuracy
        target_acc_results = [r for r in results if 0.84 <= r['accuracy'] <= 0.88]
        target_acc_results.sort(key=lambda x: abs(x['avg_il'] - 20))

        print()
        print("=" * 70)
        print("ACCURACY 84-88% 범위 (IL 20 근접 순)")
        print("=" * 70)
        print()
        print(f"{'Denied':<8} {'Conf':<8} {'Gap':<8} {'Ratio':<8} {'MinIL':<6} {'Accuracy':>10} {'Avg IL':>10} {'Exhausted':>10}")
        print("-" * 80)

        for i, r in enumerate(target_acc_results[:10]):
            marker = "★" if i == 0 else " "
            print(f"{marker} <{r['denied_threshold']:<5} >={r['confidence_threshold']:<5} >={r['gap_threshold']:<5} >={r['relative_gap']:<5} {r['min_il']:<6} {r['accuracy']*100:>9.1f}% {r['avg_il']:>10.1f} {r['exhausted_ratio']*100:>9.1f}%")

        # Best overall
        all_sorted = sorted(results, key=lambda x: -x['accuracy'])

        print()
        print("=" * 70)
        print("BEST OVERALL (정확도 순, TOP 10)")
        print("=" * 70)
        print()
        print(f"{'Denied':<8} {'Conf':<8} {'Gap':<8} {'Ratio':<8} {'MinIL':<6} {'Accuracy':>10} {'Avg IL':>10} {'Exhausted':>10}")
        print("-" * 80)

        for i, r in enumerate(all_sorted[:10]):
            marker = "★" if i == 0 else " "
            print(f"{marker} <{r['denied_threshold']:<5} >={r['confidence_threshold']:<5} >={r['gap_threshold']:<5} >={r['relative_gap']:<5} {r['min_il']:<6} {r['accuracy']*100:>9.1f}% {r['avg_il']:>10.1f} {r['exhausted_ratio']*100:>9.1f}%")

        # Save all results
        output_file = Path("/home/max/Graph-DDXPlus/results/cypher_optimization_v6.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print()
        print(f"All results saved to: {output_file}")

        # Best recommendation
        best_86 = [r for r in results if 0.84 <= r['accuracy'] <= 0.88 and 18 <= r['avg_il'] <= 22]
        if best_86:
            best_86.sort(key=lambda x: (-x['accuracy'], abs(x['avg_il'] - 20)))
            best = best_86[0]
            print()
            print("=" * 70)
            print("추천 설정 (Acc ~86%, IL ~20 목표)")
            print("=" * 70)
            print(f"  denied_threshold: {best['denied_threshold']}")
            print(f"  confidence_threshold: {best['confidence_threshold']}")
            print(f"  gap_threshold: {best['gap_threshold']}")
            print(f"  relative_gap: {best['relative_gap']}")
            print(f"  min_il: {best['min_il']}")
            print(f"  → Accuracy: {best['accuracy']*100:.1f}%, IL: {best['avg_il']:.1f}")

    finally:
        optimizer.close()


if __name__ == "__main__":
    main()
