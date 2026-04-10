#!/usr/bin/env python3
"""최종 실험: 후보 생성 3 × 종료 5 × 스코어링 5 = 75개 조합.

후보 생성에 deny_threshold와 antecedent를 파라미터로 받음.
선택 전략은 greedy 고정.
134,529건 전체 평가.

사용법: python experiment_final_75.py \
          --deny-threshold 5 --antecedent 0 \
          --stopping top3_stable_5 --scoring v15_ratio \
          --workers 4 --ports "7687,7688,7689,7690"
"""

import argparse
import json
import math
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

MAX_IL = 223


# ============================================================
# 스코어링
# ============================================================

def score_v15_ratio(c, d, t):
    return (float(c) / (float(c) + float(d) + 1.0)) * float(c)

def score_v18_coverage(c, d, t):
    return (float(c) / (float(t) + 1.0)) * float(c)

def score_jaccard(c, d, t):
    u = max(float(t) - float(c) - float(d), 0)
    denom = float(c) + float(d) + u
    return (float(c) / denom) * float(c) if denom > 0 else 0.0

def score_tfidf(c, d, t):
    if c == 0 or t == 0: return 0.0
    tf = float(c) / (float(t) + 1.0)
    prevalence = float(t) / 223.0
    idf = math.log(1.0 + 1.0 / (prevalence + 0.01))
    return tf * idf * float(c)

def score_cosine(c, d, t):
    if c == 0 or t == 0: return 0.0
    qn = math.sqrt(float(c) + float(d)) if (c + d) > 0 else 1.0
    dn = math.sqrt(float(t)) if t > 0 else 1.0
    return (float(c) / (qn * dn)) * float(c)

SCORING_FUNCTIONS = {
    "v15_ratio": score_v15_ratio,
    "v18_coverage": score_v18_coverage,
    "jaccard": score_jaccard,
    "tfidf": score_tfidf,
    "cosine": score_cosine,
}


# ============================================================
# 후보 생성 (deny_threshold, antecedent 파라미터화)
# ============================================================

def get_candidates(kg, initial_cui, deny_threshold, use_antecedent,
                   confirmed_cuis, denied_cuis, asked_cuis):
    _confirmed = confirmed_cuis
    _denied = denied_cuis
    _asked = asked_cuis

    if not _confirmed - {initial_cui}:
        ante_clause = """
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
               {ante_clause}
        LIMIT 10
        """
        with kg.driver.session() as session:
            result = session.run(query, initial_cui=initial_cui, asked_cuis=list(_asked))
            from src.umls_kg import SymptomCandidate
            return [SymptomCandidate(cui=r["cui"], name=r["name"], disease_coverage=r["disease_coverage"]) for r in result]

    deny_filter = f"WHERE denied_count < {deny_threshold}" if deny_threshold > 0 else ""
    ante_order = "priority ASC," if use_antecedent else ""
    ante_field = "CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority" if use_antecedent else "0 AS priority"

    # co-occurrence 사용 (ablation에서 효과 미미하므로 기본 포함)
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
         {ante_field}
    RETURN next.cui AS cui, next.name AS name, coverage AS disease_coverage, priority
    ORDER BY {ante_order} toFloat(cooccur_count) * coverage DESC
    LIMIT 10
    """
    with kg.driver.session() as session:
        result = session.run(query,
                             confirmed_cuis=list(_confirmed),
                             denied_cuis=list(_denied),
                             asked_cuis=list(_asked))
        from src.umls_kg import SymptomCandidate
        return [SymptomCandidate(cui=r["cui"], name=r["name"], disease_coverage=r["disease_coverage"]) for r in result]


# ============================================================
# 종료 조건
# ============================================================

def calc_entropy(scores):
    if not scores:
        return 0.0
    total = sum(scores)
    if total <= 0:
        return 0.0
    probs = [s / total for s in scores if s > 0]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def check_stopping(stopping, il, step_hits, confirmed_count, rank_history, kg_dist):
    if stopping == "top3_stable_5":
        if len(rank_history) >= 5:
            recent = list(rank_history)[-5:]
            return all(r == recent[0] for r in recent)
        return False

    if stopping == "top1_stable_5":
        if len(rank_history) >= 5:
            recent = [r[0] for r in list(rank_history)[-5:]]
            return all(r == recent[0] for r in recent)
        return False

    if stopping == "conf_gap_005":
        if kg_dist and len(kg_dist) >= 2:
            return (kg_dist[0][1] - kg_dist[1][1]) >= 0.05
        return kg_dist and len(kg_dist) == 1

    if stopping == "cumulative_confirmed_5":
        return confirmed_count >= 5

    if stopping == "hr_plateau":
        if len(step_hits) >= 10:
            return sum(step_hits[-10:]) == 0 or (sum(step_hits[-5:]) == 0 and sum(step_hits[-10:-5]) <= 1)
        return False

    return False


# ============================================================
# 커스텀 진단
# ============================================================

def get_custom_diagnosis(kg, scoring_name, confirmed_cuis, denied_cuis, top_k=10):
    scoring_fn = SCORING_FUNCTIONS[scoring_name]
    query = """
    MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
    WHERE confirmed.cui IN $confirmed_cuis
    WITH DISTINCT d
    OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
    WITH d,
         count(DISTINCT s) AS total_symptoms,
         count(DISTINCT CASE WHEN s.cui IN $confirmed_cuis THEN s END) AS confirmed_count,
         count(DISTINCT CASE WHEN s.cui IN $denied_cuis THEN s END) AS denied_count
    WHERE confirmed_count > 0
    RETURN d.cui AS cui, d.name AS name,
           confirmed_count, denied_count, total_symptoms
    """
    with kg.driver.session() as session:
        result = session.run(query,
                             confirmed_cuis=list(confirmed_cuis),
                             denied_cuis=list(denied_cuis))
        candidates = []
        for r in result:
            raw = scoring_fn(r["confirmed_count"], r["denied_count"], r["total_symptoms"])
            candidates.append({"cui": r["cui"], "name": r["name"], "score": raw,
                               "confirmed_count": r["confirmed_count"], "total_symptoms": r["total_symptoms"]})

    total = sum(c["score"] for c in candidates)
    for c in candidates:
        c["score"] = c["score"] / total if total > 0 else 0
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


# ============================================================
# 메인
# ============================================================

def run_single_patient(args):
    patient_data, loader_data, deny_threshold, use_antecedent, stopping, scoring, neo4j_port = args

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

        gt_disease_eng = loader.fr_to_eng.get(patient.pathology, patient.pathology)
        gt_cui = loader.get_disease_cui(gt_disease_eng)

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

        il = 0
        confirmed_count = 1
        step_hits = []
        rank_history = deque(maxlen=10)

        for _ in range(MAX_IL):
            candidates = get_candidates(
                kg, initial_cui, deny_threshold, use_antecedent,
                kg.state.confirmed_cuis, kg.state.denied_cuis, kg.state.asked_cuis,
            )
            if not candidates:
                break

            selected_cui = candidates[0].cui  # greedy

            hit = 1 if selected_cui in patient_positive_cuis else 0
            if hit:
                kg.state.add_confirmed(selected_cui)
                confirmed_count += 1
            else:
                kg.state.add_denied(selected_cui)
            il += 1
            step_hits.append(hit)

            # 종료 판단: 기존 kg.get_diagnosis_candidates() (v15_ratio)
            kg_diag = kg.get_diagnosis_candidates(top_k=10)
            kg_dist = [(c.cui, c.score) for c in kg_diag] if kg_diag else []
            current_ranks = tuple(cui for cui, _ in kg_dist[:3])
            rank_history.append(current_ranks)

            if check_stopping(stopping, il, step_hits, confirmed_count, rank_history, kg_dist):
                break

        # 최종 진단: 커스텀 스코어링
        final = get_custom_diagnosis(kg, scoring, kg.state.confirmed_cuis, kg.state.denied_cuis, top_k=10)
        correct_at_1 = final[0]["cui"] == gt_cui if final else False
        correct_at_10 = any(c["cui"] == gt_cui for c in final[:10]) if final else False

        kg.close()
        return {
            "error": False,
            "correct_at_1": int(correct_at_1),
            "correct_at_10": int(correct_at_10),
            "il": il,
            "confirmed": confirmed_count,
        }
    except Exception:
        kg.close()
        return {"error": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deny-threshold", type=int, required=True)
    parser.add_argument("--antecedent", type=int, required=True, choices=[0, 1])
    parser.add_argument("--stopping", required=True,
                        choices=["top3_stable_5", "top1_stable_5", "conf_gap_005",
                                 "cumulative_confirmed_5", "hr_plateau"])
    parser.add_argument("--scoring", required=True, choices=list(SCORING_FUNCTIONS.keys()))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ports", type=str, default="7687,7688,7689,7690")
    args = parser.parse_args()

    ante_str = "ante" if args.antecedent else "noante"
    method_name = f"deny{args.deny_threshold}_{ante_str}+{args.stopping}+{args.scoring}"
    ports = [int(p.strip()) for p in args.ports.split(",")]

    from src.data_loader import DDXPlusLoader
    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test")
    print(f"=== {method_name} ({len(patients):,}건) ===")

    loader_data = {
        "symptom_mapping": loader.symptom_mapping,
        "disease_mapping": loader.disease_mapping,
        "fr_to_eng": loader.fr_to_eng,
        "conditions": {k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
                       for k, v in loader.conditions.items()},
    }
    patients_data = [
        {"age": p.age, "sex": p.sex, "initial_evidence": p.initial_evidence,
         "evidences": p.evidences, "pathology": p.pathology,
         "differential_diagnosis": p.differential_diagnosis}
        for p in patients
    ]

    tasks = [(pd, loader_data, args.deny_threshold, bool(args.antecedent),
              args.stopping, args.scoring, ports[i % len(ports)])
             for i, pd in enumerate(patients_data)]

    start = time.time()
    results, errors = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_single_patient, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc=method_name[:40]) as pbar:
            for f in as_completed(futures):
                r = f.result()
                if r and not r.get("error"):
                    results.append(r)
                else:
                    errors += 1
                pbar.update(1)

    elapsed = time.time() - start
    count = len(results)
    ils = [r["il"] for r in results]

    output = {
        "deny_threshold": args.deny_threshold, "antecedent": args.antecedent,
        "stopping": args.stopping, "scoring": args.scoring,
        "method": method_name,
        "count": count, "errors": errors,
        "gtpa_1": sum(r["correct_at_1"] for r in results) / count if count else 0,
        "gtpa_10": sum(r["correct_at_10"] for r in results) / count if count else 0,
        "avg_il": float(np.mean(ils)), "il_std": float(np.std(ils)),
        "avg_confirmed": float(np.mean([r["confirmed"] for r in results])),
        "elapsed": elapsed,
    }

    print(f"\nGTPA@1: {output['gtpa_1']:.2%}, GTPA@10: {output['gtpa_10']:.2%}")
    print(f"Avg IL: {output['avg_il']:.1f}, Confirmed: {output['avg_confirmed']:.1f}")

    safe = method_name.replace("+", "_")
    path = Path("results") / f"final75_{safe}_{count}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
