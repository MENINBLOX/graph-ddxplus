#!/usr/bin/env python3
"""증상 탐색 전략별 Hit Rate 곡선 분석 v2.

과거 + 현재 모든 방법 포함.
종료 조건 없이 후보 증상이 소진될 때까지 매 IL마다 기록.

=== 방법 분류 ===

A. KG 후보 생성 방식 (Cypher 변형)
   - cooccur: co-occurrence 기반 (현재 기본)
   - coverage_only: disease_coverage만 사용 (초기 방식)
   - cooccur_no_deny_filter: denied<5 필터 제거
   - coverage_no_antecedent: is_antecedent 우선순위 제거

B. 선택 전략 (후보 중 어떤 것을 선택할지)
   - greedy: top-1 선택 (기본)
   - ig_expected: 기대 정보이득 최대화
   - ig_max: 최대 정보이득 최대화
   - ig_binary_split: 균등 이분할
   - minimax_score: 최악 진단 점수 최대화
   - minimax_entropy: 최악 엔트로피 최소화

C. 조합
   method = "{cypher_variant}+{selection_strategy}"
   예: "cooccur+greedy", "coverage_only+ig_expected"
"""

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

MAX_IL = 100


def calc_entropy(scores):
    if not scores:
        return 0.0
    total = sum(scores)
    if total <= 0:
        return 0.0
    probs = [s / total for s in scores if s > 0]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def get_candidates_by_variant(kg, initial_cui, variant, confirmed_cuis, denied_cuis, asked_cuis):
    """Cypher 변형별 후보 생성."""
    _confirmed = confirmed_cuis
    _denied = denied_cuis
    _asked = asked_cuis

    if not _confirmed - {initial_cui}:
        # 초기 상태: 모든 변형이 동일한 초기 쿼리 사용
        query = """
        MATCH (s:Symptom {cui: $initial_cui})-[:INDICATES]->(d:Disease)
        MATCH (d)<-[:INDICATES]-(related:Symptom)
        WHERE related.cui <> $initial_cui
          AND NOT related.cui IN $asked_cuis
        WITH related, count(DISTINCT d) AS disease_coverage
        RETURN related.cui AS cui, related.name AS name, disease_coverage,
               CASE WHEN related.is_antecedent = false THEN 0 ELSE 1 END AS priority
        ORDER BY priority ASC, disease_coverage DESC
        LIMIT 10
        """
        with kg.driver.session() as session:
            result = session.run(query, initial_cui=initial_cui, asked_cuis=list(_asked))
            from src.umls_kg import SymptomCandidate
            return [SymptomCandidate(cui=r["cui"], name=r["name"], disease_coverage=r["disease_coverage"]) for r in result]

    if variant == "cooccur":
        # 현재 기본: co-occurrence + denied<5 필터 + antecedent 우선순위
        query = """
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d
        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        WITH d, count(DISTINCT denied) AS denied_count
        WHERE denied_count < 5
        WITH collect(DISTINCT d) AS valid_diseases
        WHERE size(valid_diseases) > 0
        UNWIND valid_diseases AS d
        MATCH (d)<-[:INDICATES]-(next:Symptom)
        WHERE NOT next.cui IN $confirmed_cuis
          AND NOT next.cui IN $denied_cuis
          AND NOT next.cui IN $asked_cuis
        WITH next, d
        MATCH (d)<-[:INDICATES]-(conf:Symptom)
        WHERE conf.cui IN $confirmed_cuis
        WITH next, count(DISTINCT d) AS coverage, count(DISTINCT conf) AS cooccur_count,
             CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority
        RETURN next.cui AS cui, next.name AS name, coverage AS disease_coverage, priority
        ORDER BY priority ASC, toFloat(cooccur_count) * coverage DESC
        LIMIT 10
        """
    elif variant == "coverage_only":
        # 초기 방식: disease_coverage만
        query = """
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d
        MATCH (d)<-[:INDICATES]-(next:Symptom)
        WHERE NOT next.cui IN $confirmed_cuis
          AND NOT next.cui IN $denied_cuis
          AND NOT next.cui IN $asked_cuis
        WITH next, count(DISTINCT d) AS disease_coverage,
             CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority
        RETURN next.cui AS cui, next.name AS name, disease_coverage, priority
        ORDER BY priority ASC, disease_coverage DESC
        LIMIT 10
        """
    elif variant == "cooccur_no_deny_filter":
        # co-occurrence but denied<5 필터 제거
        query = """
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d
        MATCH (d)<-[:INDICATES]-(next:Symptom)
        WHERE NOT next.cui IN $confirmed_cuis
          AND NOT next.cui IN $denied_cuis
          AND NOT next.cui IN $asked_cuis
        WITH next, d
        MATCH (d)<-[:INDICATES]-(conf:Symptom)
        WHERE conf.cui IN $confirmed_cuis
        WITH next, count(DISTINCT d) AS coverage, count(DISTINCT conf) AS cooccur_count,
             CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority
        RETURN next.cui AS cui, next.name AS name, coverage AS disease_coverage, priority
        ORDER BY priority ASC, toFloat(cooccur_count) * coverage DESC
        LIMIT 10
        """
    elif variant == "coverage_no_antecedent":
        # coverage만, antecedent 우선순위 없이
        query = """
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d
        MATCH (d)<-[:INDICATES]-(next:Symptom)
        WHERE NOT next.cui IN $confirmed_cuis
          AND NOT next.cui IN $denied_cuis
          AND NOT next.cui IN $asked_cuis
        WITH next, count(DISTINCT d) AS disease_coverage
        RETURN next.cui AS cui, next.name AS name, disease_coverage, 0 AS priority
        ORDER BY disease_coverage DESC
        LIMIT 10
        """
    else:
        raise ValueError(f"Unknown variant: {variant}")

    with kg.driver.session() as session:
        result = session.run(query,
                             confirmed_cuis=list(_confirmed),
                             denied_cuis=list(_denied),
                             asked_cuis=list(_asked))
        from src.umls_kg import SymptomCandidate
        return [SymptomCandidate(cui=r["cui"], name=r["name"], disease_coverage=r["disease_coverage"]) for r in result]


def get_disease_distribution(kg, top_k=10):
    candidates = kg.get_diagnosis_candidates(top_k=top_k)
    return [(c.cui, c.score) for c in candidates]


def select_from_candidates(kg, candidates, selection, patient_positive_cuis):
    """선택 전략."""
    if selection == "greedy" or not candidates:
        return candidates[0].cui if candidates else None

    if selection in ("ig_expected", "ig_max", "ig_binary_split"):
        current_dist = get_disease_distribution(kg, top_k=10)
        current_entropy = calc_entropy([s for _, s in current_dist])
        best_cui, best_score = candidates[0].cui, -float('inf')

        for c in candidates[:7]:
            sc, sd, sa = kg.state.confirmed_cuis.copy(), kg.state.denied_cuis.copy(), kg.state.asked_cuis.copy()
            kg.state.confirmed_cuis = sc | {c.cui}; kg.state.asked_cuis = sa | {c.cui}
            ey = calc_entropy([s for _, s in get_disease_distribution(kg, top_k=10)])
            kg.state.confirmed_cuis = sc.copy(); kg.state.denied_cuis = sd | {c.cui}; kg.state.asked_cuis = sa | {c.cui}
            en = calc_entropy([s for _, s in get_disease_distribution(kg, top_k=10)])
            kg.state.confirmed_cuis, kg.state.denied_cuis, kg.state.asked_cuis = sc, sd, sa

            n_d = max(len(current_dist), 1)
            py = min(max(c.disease_coverage / n_d, 0.1), 0.9)
            if selection == "ig_expected": score = current_entropy - (py * ey + (1 - py) * en)
            elif selection == "ig_max": score = max(current_entropy - ey, current_entropy - en)
            else: score = -abs(py - 0.5)
            if score > best_score: best_score, best_cui = score, c.cui
        return best_cui

    if selection in ("minimax_score", "minimax_entropy"):
        best_cui, best_score = candidates[0].cui, -float('inf')
        for c in candidates[:7]:
            sc, sd, sa = kg.state.confirmed_cuis.copy(), kg.state.denied_cuis.copy(), kg.state.asked_cuis.copy()
            kg.state.confirmed_cuis = sc | {c.cui}; kg.state.asked_cuis = sa | {c.cui}
            dy = get_disease_distribution(kg, top_k=10)
            sy, ey = (dy[0][1] if dy else 0), calc_entropy([s for _, s in dy])
            kg.state.confirmed_cuis = sc.copy(); kg.state.denied_cuis = sd | {c.cui}; kg.state.asked_cuis = sa | {c.cui}
            dn = get_disease_distribution(kg, top_k=10)
            sn, en2 = (dn[0][1] if dn else 0), calc_entropy([s for _, s in dn])
            kg.state.confirmed_cuis, kg.state.denied_cuis, kg.state.asked_cuis = sc, sd, sa
            if selection == "minimax_score": score = min(sy, sn)
            else: score = -max(ey, en2)
            if score > best_score: best_score, best_cui = score, c.cui
        return best_cui

    return candidates[0].cui


def run_single_patient(args: tuple) -> dict:
    patient_data, loader_data, cypher_variant, selection, neo4j_port = args

    from src.data_loader import DDXPlusLoader, Patient
    from src.umls_kg import UMLSKG

    loader = DDXPlusLoader()
    loader._symptom_mapping = loader_data["symptom_mapping"]
    loader._disease_mapping = loader_data["disease_mapping"]
    loader._fr_to_eng = loader_data["fr_to_eng"]
    loader._conditions = loader_data["conditions"]

    try:
        kg = UMLSKG(uri=f"bolt://localhost:{neo4j_port}")
    except Exception:
        return {"error": True}

    try:
        patient = Patient(
            age=patient_data["age"], sex=patient_data["sex"],
            initial_evidence=patient_data["initial_evidence"],
            evidences=patient_data["evidences"],
            pathology=patient_data["pathology"],
            differential_diagnosis=patient_data["differential_diagnosis"],
        )

        patient_positive_cuis = set()
        for ev_str in patient.evidences:
            code = ev_str.split("_@_")[0] if "_@_" in ev_str else ev_str
            cui = loader.get_symptom_cui(code)
            if cui:
                patient_positive_cuis.add(cui)

        initial_cui = loader.get_symptom_cui(patient.initial_evidence)
        if not initial_cui:
            kg.close()
            return {"error": True}

        kg.reset_state()
        kg.state.add_confirmed(initial_cui)

        cumulative_confirmed = [1]
        cumulative_denied = [0]
        step_hit = []

        for _ in range(MAX_IL):
            candidates = get_candidates_by_variant(
                kg, initial_cui, cypher_variant,
                kg.state.confirmed_cuis, kg.state.denied_cuis, kg.state.asked_cuis,
            )
            if not candidates:
                break

            selected_cui = select_from_candidates(kg, candidates, selection, patient_positive_cuis)
            if not selected_cui:
                break

            if selected_cui in patient_positive_cuis:
                kg.state.add_confirmed(selected_cui)
                step_hit.append(1)
            else:
                kg.state.add_denied(selected_cui)
                step_hit.append(0)

            cumulative_confirmed.append(cumulative_confirmed[-1] + step_hit[-1])
            cumulative_denied.append(cumulative_denied[-1] + (1 - step_hit[-1]))

        kg.close()
        return {
            "error": False,
            "il": len(step_hit),
            "total_confirmed": cumulative_confirmed[-1],
            "total_denied": cumulative_denied[-1],
            "total_patient_symptoms": len(patient_positive_cuis),
            "cumulative_confirmed": cumulative_confirmed,
            "step_hit": step_hit,
        }
    except Exception:
        kg.close()
        return {"error": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cypher", required=True,
                        choices=["cooccur", "coverage_only", "cooccur_no_deny_filter", "coverage_no_antecedent"])
    parser.add_argument("--selection", required=True,
                        choices=["greedy", "ig_expected", "ig_max", "ig_binary_split", "minimax_score", "minimax_entropy"])
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ports", type=str, default="7687,7688")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    method_name = f"{args.cypher}+{args.selection}"
    ports = [int(p.strip()) for p in args.ports.split(",")]
    print(f"=== Hit Rate 곡선: {method_name} ({args.n_samples}건) ===")

    from src.data_loader import DDXPlusLoader
    loader = DDXPlusLoader()
    all_patients = loader.load_patients(split="test")
    random.seed(args.seed)
    indices = random.sample(range(len(all_patients)), min(args.n_samples, len(all_patients)))
    patients = [all_patients[i] for i in indices]

    loader_data = {
        "symptom_mapping": loader.symptom_mapping,
        "disease_mapping": loader.disease_mapping,
        "fr_to_eng": loader.fr_to_eng,
        "conditions": {
            k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
            for k, v in loader.conditions.items()
        },
    }

    patients_data = [
        {"age": p.age, "sex": p.sex, "initial_evidence": p.initial_evidence,
         "evidences": p.evidences, "pathology": p.pathology,
         "differential_diagnosis": p.differential_diagnosis}
        for p in patients
    ]

    tasks = [(pd, loader_data, args.cypher, args.selection, ports[i % len(ports)]) for i, pd in enumerate(patients_data)]

    start_time = time.time()
    results = []
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_patient, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc=method_name) as pbar:
            for future in as_completed(futures):
                r = future.result()
                if r and not r.get("error"):
                    results.append(r)
                else:
                    errors += 1
                pbar.update(1)

    elapsed = time.time() - start_time
    count = len(results)

    # 곡선 계산
    max_observed_il = max(r["il"] for r in results) if results else 0
    curve_data = {}
    for il in range(1, min(max_observed_il + 1, MAX_IL + 1)):
        vals_hr = []
        vals_confirmed = []
        marginal_hits = []
        for r in results:
            if il < len(r["cumulative_confirmed"]):
                c = r["cumulative_confirmed"][il]
                total = c + (r["cumulative_confirmed"][0] - 1 + il - (c - 1))  # confirmed + denied at step il
                d = il - (c - 1)  # denied = total_steps - (confirmed - initial)
                hr = c / (c + d) if (c + d) > 0 else 0
                vals_hr.append(hr)
                vals_confirmed.append(c)
            if il - 1 < len(r["step_hit"]):
                marginal_hits.append(r["step_hit"][il - 1])
        if len(vals_hr) < 10:
            break
        curve_data[str(il)] = {
            "n": len(vals_hr),
            "hit_rate": float(np.mean(vals_hr)),
            "confirmed": float(np.mean(vals_confirmed)),
            "marginal": float(np.mean(marginal_hits)) if marginal_hits else 0,
        }

    avg_il = np.mean([r["il"] for r in results]) if results else 0
    avg_confirmed = np.mean([r["total_confirmed"] for r in results]) if results else 0
    avg_hr = np.mean([r["total_confirmed"] / max(r["total_confirmed"] + r["total_denied"], 1) for r in results]) if results else 0

    output = {
        "method": method_name, "cypher": args.cypher, "selection": args.selection,
        "n_samples": args.n_samples, "seed": args.seed,
        "count": count, "errors": errors,
        "avg_il": float(avg_il), "avg_confirmed": float(avg_confirmed),
        "avg_hit_rate": float(avg_hr),
        "curve": curve_data, "elapsed": elapsed,
    }

    print(f"\n{'IL':>4} | {'Hit Rate':>8} | {'Marginal':>8} | {'Confirmed':>9} | {'Active':>6}")
    print("-" * 55)
    for il in [1, 3, 5, 10, 15, 20, 30, 50]:
        if str(il) in curve_data:
            d = curve_data[str(il)]
            print(f"{il:>4} | {d['hit_rate']:>7.1%} | {d['marginal']:>7.1%} | {d['confirmed']:>8.1f} | {d['n']:>6}")

    print(f"\n최종: IL={avg_il:.1f}, Confirmed={avg_confirmed:.1f}, Hit Rate={avg_hr:.1%}, Time={elapsed:.0f}s")

    output_path = Path("results") / f"hitcurve2_{args.cypher}_{args.selection}_{args.n_samples}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
