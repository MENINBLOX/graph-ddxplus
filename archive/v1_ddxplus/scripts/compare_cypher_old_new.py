#!/usr/bin/env python3
"""기존/최적화 Cypher 비교.

동일한 KG-only 진단 루프에서 기존 쿼리와 현재 쿼리를 같은 샘플에 적용해
점수와 실행 시간을 비교한다.
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DDXPlusLoader, Patient
from src.umls_kg import DiagnosisCandidate, KGState, SymptomCandidate, UMLSKG


class OldQueryUMLSKG(UMLSKG):
    """최적화 전 Cypher를 유지한 비교용 KG."""

    def _get_accumulated_candidates(
        self,
        limit: int,
        *,
        confirmed_cuis: set[str] | None = None,
        denied_cuis: set[str] | None = None,
        asked_cuis: set[str] | None = None,
    ) -> list[SymptomCandidate]:
        _confirmed = confirmed_cuis if confirmed_cuis is not None else self.state.confirmed_cuis
        _denied = denied_cuis if denied_cuis is not None else self.state.denied_cuis
        _asked = asked_cuis if asked_cuis is not None else self.state.asked_cuis

        query = """
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH collect(DISTINCT d) AS candidate_diseases

        UNWIND candidate_diseases AS d
        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        WITH d, candidate_diseases, count(denied) AS denied_count
        WITH d, candidate_diseases
        WHERE denied_count < 5

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
        with self.driver.session() as session:
            result = session.run(
                query,
                confirmed_cuis=list(_confirmed),
                denied_cuis=list(_denied),
                asked_cuis=list(_asked),
                limit=limit,
            )
            return [
                SymptomCandidate(
                    cui=r["cui"],
                    name=r["name"],
                    disease_coverage=r["disease_coverage"],
                )
                for r in result
            ]

    def get_diagnosis_candidates(
        self,
        top_k: int = 5,
        scoring: str = "v23_mild_denied",
        *,
        confirmed_cuis: set[str] | None = None,
        denied_cuis: set[str] | None = None,
    ) -> list[DiagnosisCandidate]:
        _confirmed = confirmed_cuis if confirmed_cuis is not None else self.state.confirmed_cuis
        _denied = denied_cuis if denied_cuis is not None else self.state.denied_cuis

        if not _confirmed:
            return []

        if scoring == "v23_mild_denied":
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
            WITH d, confirmed_count, denied_count, total_symptoms,
                 (toFloat(confirmed_count) / (toFloat(total_symptoms) + 1.0) * toFloat(confirmed_count))
                 * (1.0 - 0.1 * toFloat(denied_count) / (toFloat(total_symptoms) + 1.0)) AS raw_score
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
        elif scoring == "v18_coverage":
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
        elif scoring == "v15_ratio":
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
        else:
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
                 toFloat(confirmed_count) - 0.5 * toFloat(denied_count) AS raw_score
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

        with self.driver.session() as session:
            result = session.run(
                query,
                confirmed_cuis=list(_confirmed),
                denied_cuis=list(_denied),
                top_k=top_k,
            )
            return [
                DiagnosisCandidate(
                    cui=r["cui"],
                    name=r["name"],
                    score=r["score"],
                    confirmed_count=r["confirmed_count"],
                    total_symptoms=r["total_symptoms"],
                )
                for r in result
            ]


@dataclass
class PatientState:
    patient: Patient
    idx: int
    initial_cui: str | None = None
    confirmed_cuis: set[str] = field(default_factory=set)
    denied_cuis: set[str] = field(default_factory=set)
    asked_cuis: set[str] = field(default_factory=set)
    il: int = 0
    predicted: str | None = None
    done: bool = False


def build_cui_to_disease(loader: DDXPlusLoader) -> tuple[dict[str, str], dict[str, str]]:
    cui_to_disease: dict[str, str] = {}
    disease_to_cui: dict[str, str] = {}
    for name_eng, info in loader.disease_mapping.items():
        cui = info.get("umls_cui")
        if cui:
            cond = loader.conditions.get(name_eng)
            if cond:
                cui_to_disease[cui] = cond.name_fr
                disease_to_cui[cond.name_fr] = cui
    return cui_to_disease, disease_to_cui


def get_cui_for_code(cui_to_codes: dict[str, list[str]], code: str) -> str | None:
    for cui, codes in cui_to_codes.items():
        if code in codes:
            return cui
    return None


def check_evidence(patient: Patient, codes: list[str]) -> bool:
    evidences = set(patient.evidences)
    for code in codes:
        if code in evidences:
            return True
        for ev in evidences:
            if ev.startswith(f"{code}_@_"):
                return True
    return False


def run_benchmark(
    kg_cls: type[UMLSKG],
    n_samples: int,
    severity: int | None,
    scoring: str,
) -> dict:
    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test", n_samples=n_samples, severity=severity)
    cui_to_codes = loader.build_cui_to_codes()
    cui_to_disease, _ = build_cui_to_disease(loader)
    kg = kg_cls()
    start = time.time()

    states: list[PatientState] = []
    for idx, patient in enumerate(patients):
        state = PatientState(patient=patient, idx=idx)
        init_cui = get_cui_for_code(cui_to_codes, patient.initial_evidence)
        if init_cui:
            state.initial_cui = init_cui
            state.confirmed_cuis.add(init_cui)
            state.asked_cuis.add(init_cui)
        state.il = 1
        states.append(state)

    max_il = 50
    for _round in range(1, max_il + 1):
        active = [s for s in states if not s.done]
        if not active:
            break
        for state in active:
            kg.state = KGState(
                confirmed_cuis=state.confirmed_cuis.copy(),
                denied_cuis=state.denied_cuis.copy(),
                asked_cuis=state.asked_cuis.copy(),
            )
            should_stop, _ = kg.should_stop(
                max_il=max_il,
                min_il=3,
                confidence_threshold=0.25,
                gap_threshold=0.06,
                relative_gap_threshold=1.5,
            )
            if should_stop or not state.initial_cui:
                dx = kg.get_diagnosis_candidates(
                    top_k=100,
                    scoring=scoring,
                    confirmed_cuis=state.confirmed_cuis.copy(),
                    denied_cuis=state.denied_cuis.copy(),
                )
                for c in dx:
                    if c.cui in cui_to_disease:
                        state.predicted = cui_to_disease[c.cui]
                        break
                if not state.predicted:
                    state.predicted = next(iter(cui_to_disease.values()))
                state.done = True
                continue

            cands = kg.get_candidate_symptoms(
                state.initial_cui,
                limit=10,
                confirmed_cuis=state.confirmed_cuis.copy(),
                denied_cuis=state.denied_cuis.copy(),
                asked_cuis=state.asked_cuis.copy(),
            )
            if not cands:
                dx = kg.get_diagnosis_candidates(
                    top_k=100,
                    scoring=scoring,
                    confirmed_cuis=state.confirmed_cuis.copy(),
                    denied_cuis=state.denied_cuis.copy(),
                )
                for c in dx:
                    if c.cui in cui_to_disease:
                        state.predicted = cui_to_disease[c.cui]
                        break
                if not state.predicted:
                    state.predicted = next(iter(cui_to_disease.values()))
                state.done = True
                continue

            sel = cands[0]
            state.asked_cuis.add(sel.cui)
            if check_evidence(state.patient, cui_to_codes.get(sel.cui, [])):
                state.confirmed_cuis.add(sel.cui)
            else:
                state.denied_cuis.add(sel.cui)
            state.il += 1

    for state in states:
        if not state.done:
            dx = kg.get_diagnosis_candidates(
                top_k=100,
                scoring=scoring,
                confirmed_cuis=state.confirmed_cuis.copy(),
                denied_cuis=state.denied_cuis.copy(),
            )
            for c in dx:
                if c.cui in cui_to_disease:
                    state.predicted = cui_to_disease[c.cui]
                    break
            if not state.predicted:
                state.predicted = next(iter(cui_to_disease.values()))
            state.done = True

    elapsed = time.time() - start
    correct = sum(1 for s in states if s.predicted == s.patient.pathology)
    avg_il = sum(s.il for s in states) / len(states)
    kg.close()
    return {
        "n_samples": len(states),
        "gtpa_at_1": correct / len(states),
        "avg_il": avg_il,
        "elapsed_sec": elapsed,
        "samples_per_min": len(states) / elapsed * 60.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old/new Cypher queries")
    parser.add_argument("-n", "--n-samples", type=int, default=200)
    parser.add_argument("--severity", type=int, default=2)
    parser.add_argument("--scoring", default="v18_coverage")
    args = parser.parse_args()

    print(f"Comparing old vs new Cypher on n={args.n_samples}, severity={args.severity}, scoring={args.scoring}")
    old_res = run_benchmark(OldQueryUMLSKG, args.n_samples, args.severity, args.scoring)
    new_res = run_benchmark(UMLSKG, args.n_samples, args.severity, args.scoring)

    print("\nOLD")
    print(old_res)
    print("\nNEW")
    print(new_res)
    print("\nDELTA")
    print({
        "gtpa_at_1_diff": new_res["gtpa_at_1"] - old_res["gtpa_at_1"],
        "avg_il_diff": new_res["avg_il"] - old_res["avg_il"],
        "elapsed_sec_diff": new_res["elapsed_sec"] - old_res["elapsed_sec"],
        "samples_per_min_diff": new_res["samples_per_min"] - old_res["samples_per_min"],
    })


if __name__ == "__main__":
    main()
