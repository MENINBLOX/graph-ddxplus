#!/usr/bin/env python3
"""증상 탐색 전략 Hit Rate 곡선 - 완전 요인 설계 (2×4×2 × 6 선택 = 96개 조합).

2(co-occurrence) × 4(denied threshold: 0,3,5,7) × 2(antecedent) × 6(선택 전략) = 96개.
denied_threshold=0은 필터 없음을 의미.

사용법: python analyze_hit_rate_curve_v2_full.py \
          --cooccur 1 --deny-threshold 5 --antecedent 1 --selection greedy \
          --n-samples 1000 --workers 1 --ports "7687"
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

MAX_IL = 223


def calc_entropy(scores):
    if not scores:
        return 0.0
    total = sum(scores)
    if total <= 0:
        return 0.0
    probs = [s / total for s in scores if s > 0]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def get_candidates_full(kg, initial_cui, use_cooccur, deny_threshold, use_antecedent,
                        confirmed_cuis, denied_cuis, asked_cuis):
    """완전 요인 설계 후보 생성."""
    _confirmed = confirmed_cuis
    _denied = denied_cuis
    _asked = asked_cuis

    # 초기 상태
    if not _confirmed - {initial_cui}:
        antecedent_clause = """
               CASE WHEN related.is_antecedent = false THEN 0 ELSE 1 END AS priority
        ORDER BY priority ASC, disease_coverage DESC
        """ if use_antecedent else """
               0 AS priority
        ORDER BY disease_coverage DESC
        """
        query = f"""
        MATCH (s:Symptom {{cui: $initial_cui}})-[:INDICATES]->(d:Disease)
        MATCH (d)<-[:INDICATES]-(related:Symptom)
        WHERE related.cui <> $initial_cui
          AND NOT related.cui IN $asked_cuis
        WITH related, count(DISTINCT d) AS disease_coverage
        RETURN related.cui AS cui, related.name AS name, disease_coverage,
               {antecedent_clause}
        LIMIT 10
        """
        with kg.driver.session() as session:
            result = session.run(query, initial_cui=initial_cui, asked_cuis=list(_asked))
            from src.umls_kg import SymptomCandidate
            return [SymptomCandidate(cui=r["cui"], name=r["name"], disease_coverage=r["disease_coverage"]) for r in result]

    # 누적 상태
    # denied 필터 조건
    deny_filter = f"WHERE denied_count < {deny_threshold}" if deny_threshold > 0 else ""

    # antecedent 우선순위
    antecedent_order = "priority ASC," if use_antecedent else ""
    antecedent_field = "CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority" if use_antecedent else "0 AS priority"

    if use_cooccur:
        query = f"""
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d
        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        WITH d, count(DISTINCT denied) AS denied_count
        {deny_filter}
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
             {antecedent_field}
        RETURN next.cui AS cui, next.name AS name, coverage AS disease_coverage, priority
        ORDER BY {antecedent_order} toFloat(cooccur_count) * coverage DESC
        LIMIT 10
        """
    else:
        query = f"""
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d
        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        WITH d, count(DISTINCT denied) AS denied_count
        {deny_filter}
        WITH collect(DISTINCT d) AS valid_diseases
        WHERE size(valid_diseases) > 0
        UNWIND valid_diseases AS d
        MATCH (d)<-[:INDICATES]-(next:Symptom)
        WHERE NOT next.cui IN $confirmed_cuis
          AND NOT next.cui IN $denied_cuis
          AND NOT next.cui IN $asked_cuis
        WITH next, count(DISTINCT d) AS disease_coverage,
             {antecedent_field}
        RETURN next.cui AS cui, next.name AS name, disease_coverage, priority
        ORDER BY {antecedent_order} disease_coverage DESC
        LIMIT 10
        """

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


def select_from_candidates(kg, candidates, selection):
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
            if selection == "ig_expected":
                score = current_entropy - (py * ey + (1 - py) * en)
            elif selection == "ig_max":
                score = max(current_entropy - ey, current_entropy - en)
            else:
                score = -abs(py - 0.5)
            if score > best_score:
                best_score, best_cui = score, c.cui
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
            if selection == "minimax_score":
                score = min(sy, sn)
            else:
                score = -max(ey, en2)
            if score > best_score:
                best_score, best_cui = score, c.cui
        return best_cui

    return candidates[0].cui


def run_single_patient(args: tuple) -> dict:
    patient_data, loader_data, use_cooccur, deny_threshold, use_antecedent, selection, neo4j_port = args

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
        step_hit = []

        for _ in range(MAX_IL):
            candidates = get_candidates_full(
                kg, initial_cui, use_cooccur, deny_threshold, use_antecedent,
                kg.state.confirmed_cuis, kg.state.denied_cuis, kg.state.asked_cuis,
            )
            if not candidates:
                break

            selected_cui = select_from_candidates(kg, candidates, selection)
            if not selected_cui:
                break

            if selected_cui in patient_positive_cuis:
                kg.state.add_confirmed(selected_cui)
                step_hit.append(1)
            else:
                kg.state.add_denied(selected_cui)
                step_hit.append(0)

            cumulative_confirmed.append(cumulative_confirmed[-1] + step_hit[-1])

        kg.close()
        total_confirmed = cumulative_confirmed[-1]
        total_asked = len(step_hit)
        return {
            "error": False,
            "il": total_asked,
            "total_confirmed": total_confirmed,
            "total_denied": total_asked - (total_confirmed - 1),
            "cumulative_confirmed": cumulative_confirmed,
            "step_hit": step_hit,
        }
    except Exception:
        kg.close()
        return {"error": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cooccur", type=int, required=True, choices=[0, 1])
    parser.add_argument("--deny-threshold", type=int, required=True)  # 0=no filter
    parser.add_argument("--antecedent", type=int, required=True, choices=[0, 1])
    parser.add_argument("--selection", required=True,
                        choices=["greedy", "ig_expected", "ig_max", "ig_binary_split", "minimax_score", "minimax_entropy"])
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ports", type=str, default="7687")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cooccur_str = "cooccur" if args.cooccur else "coverage"
    deny_str = f"deny{args.deny_threshold}" if args.deny_threshold > 0 else "nodeny"
    ante_str = "ante" if args.antecedent else "noante"
    method_name = f"{cooccur_str}_{deny_str}_{ante_str}+{args.selection}"

    ports = [int(p.strip()) for p in args.ports.split(",")]
    print(f"=== {method_name} ({args.n_samples}건) ===")

    from src.data_loader import DDXPlusLoader
    from collections import defaultdict
    loader = DDXPlusLoader()
    all_patients = loader.load_patients(split="validate")  # validation set 사용

    # 층화 추출 (stratified sampling): 49개 질환 분포 반영
    random.seed(args.seed)
    by_disease = defaultdict(list)
    for i, p in enumerate(all_patients):
        by_disease[p.pathology].append(i)

    n_per_disease = max(1, args.n_samples // len(by_disease))
    indices = []
    for disease, idxs in by_disease.items():
        sampled = random.sample(idxs, min(n_per_disease, len(idxs)))
        indices.extend(sampled)
    # 부족분 랜덤 채우기
    if len(indices) < args.n_samples:
        remaining = list(set(range(len(all_patients))) - set(indices))
        indices.extend(random.sample(remaining, args.n_samples - len(indices)))
    indices = indices[:args.n_samples]
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

    tasks = [(pd, loader_data, bool(args.cooccur), args.deny_threshold,
              bool(args.antecedent), args.selection, ports[i % len(ports)])
             for i, pd in enumerate(patients_data)]

    start_time = time.time()
    results = []
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_patient, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc=method_name[:35]) as pbar:
            for future in as_completed(futures):
                r = future.result()
                if r and not r.get("error"):
                    results.append(r)
                else:
                    errors += 1
                pbar.update(1)

    elapsed = time.time() - start_time
    count = len(results)

    avg_il = np.mean([r["il"] for r in results]) if results else 0
    avg_confirmed = np.mean([r["total_confirmed"] for r in results]) if results else 0
    avg_hr = np.mean([r["total_confirmed"] / max(r["total_confirmed"] + r["total_denied"], 1) for r in results]) if results else 0

    # 곡선 데이터 (주요 포인트만)
    curve = {}
    max_il_obs = max(r["il"] for r in results) if results else 0
    for il in range(1, min(max_il_obs + 1, MAX_IL + 1)):
        hrs = [r["cumulative_confirmed"][il] / (r["cumulative_confirmed"][il] + il - (r["cumulative_confirmed"][il] - 1))
               for r in results if il < len(r["cumulative_confirmed"])]
        marginals = [r["step_hit"][il - 1] for r in results if il - 1 < len(r["step_hit"])]
        if len(hrs) < 10:
            break
        curve[str(il)] = {"n": len(hrs), "hit_rate": float(np.mean(hrs)), "marginal": float(np.mean(marginals)) if marginals else 0}

    output = {
        "method": method_name, "cooccur": args.cooccur, "deny_threshold": args.deny_threshold,
        "antecedent": args.antecedent, "selection": args.selection,
        "n_samples": args.n_samples, "seed": args.seed,
        "count": count, "errors": errors,
        "avg_il": float(avg_il), "avg_confirmed": float(avg_confirmed), "avg_hit_rate": float(avg_hr),
        "curve": curve, "elapsed": elapsed,
    }

    print(f"\nHit Rate: {avg_hr:.1%}, Confirmed: {avg_confirmed:.1f}, IL: {avg_il:.1f}, Time: {elapsed:.0f}s")

    output_path = Path("results") / f"hitcurve_val_{cooccur_str}_{deny_str}_{ante_str}_{args.selection}_{args.n_samples}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
