#!/usr/bin/env python3
"""MEDDxAgent와 동일 조건 비교 실험.

MEDDxAgent와 동일한 데이터소스(HuggingFace StreamBench DDXPlus test, 1,764건)에서
seed=42 셔플 후 100건을 추출하고, fixed_il=5,10,15 및 adaptive로 테스트.

사용법:
    # fixed IL=5
    uv run python scripts/experiment_fixed_il_comparison.py \
        --fixed-il 5 --workers 4 --ports "7687,7688,7689,7690"

    # adaptive (stopping criteria 사용)
    uv run python scripts/experiment_fixed_il_comparison.py \
        --adaptive --workers 4 --ports "7687,7688,7689,7690"
"""

import argparse
import json
import random
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# 최적 설정 고정 (75개 조합에서 선정)
DENY_THRESHOLD = 5
USE_ANTECEDENT = False
SCORING = "v15_ratio"
STOPPING = "top3_stable_5"
MAX_IL = 223


# ============================================================
# MEDDxAgent 동일 100건 추출 (HuggingFace 직접 로드)
# ============================================================

def load_meddxagent_100() -> list[dict]:
    """MEDDxAgent와 동일한 100건을 HuggingFace에서 직접 추출.

    Returns:
        list of dicts with keys: age, sex, initial_evidence, evidences,
        pathology_eng, differential_diagnosis
    """
    from ast import literal_eval

    from datasets import load_dataset as hf_load

    hf_dataset = hf_load("appier-ai-research/StreamBench", "ddxplus")["test"]
    print(f"  HuggingFace StreamBench DDXPlus test: {len(hf_dataset):,}건")

    # MEDDxAgent 방식: seed=42로 셔플 후 앞 100건
    indices = list(range(len(hf_dataset)))
    random.seed(42)
    random.shuffle(indices)

    patients = []
    for i in indices[:100]:
        row = hf_dataset[i]
        # EVIDENCES는 문자열 표현의 리스트 (e.g., "['dyspn', 'toux']")
        evidences = row["EVIDENCES"]
        if isinstance(evidences, str):
            evidences = literal_eval(evidences)
        patients.append({
            "age": int(row["AGE"]),
            "sex": row["SEX"],
            "initial_evidence": row["INITIAL_EVIDENCE"],
            "evidences": evidences,
            "pathology_eng": row["PATHOLOGY"],  # 영어 질환명
            "differential_diagnosis": row["DIFFERENTIAL_DIAGNOSIS"],
        })

    print(f"  ✓ 100건 추출 완료")
    return patients


# ============================================================
# 스코어링 (최적 설정: v15_ratio)
# ============================================================

def score_v15_ratio(c, d, t):
    return (float(c) / (float(c) + float(d) + 1.0)) * float(c)


# ============================================================
# 후보 생성 (최적 설정: deny5, noante, cooccur)
# ============================================================

def get_candidates(kg, initial_cui, confirmed_cuis, denied_cuis, asked_cuis):
    _confirmed = confirmed_cuis
    _denied = denied_cuis
    _asked = asked_cuis

    if not _confirmed - {initial_cui}:
        query = """
        MATCH (s:Symptom {cui: $initial_cui})-[:INDICATES]->(d:Disease)
        MATCH (d)<-[:INDICATES]-(related:Symptom)
        WHERE related.cui <> $initial_cui
          AND NOT related.cui IN $asked_cuis
        WITH related, count(DISTINCT d) AS disease_coverage
        RETURN related.cui AS cui, related.name AS name, disease_coverage,
               0 AS priority
        ORDER BY disease_coverage DESC
        LIMIT 10
        """
        with kg.driver.session() as session:
            result = session.run(query, initial_cui=initial_cui, asked_cuis=list(_asked))
            from src.umls_kg import SymptomCandidate
            return [SymptomCandidate(cui=r["cui"], name=r["name"],
                                     disease_coverage=r["disease_coverage"]) for r in result]

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
         0 AS priority
    RETURN next.cui AS cui, next.name AS name, coverage AS disease_coverage, priority
    ORDER BY toFloat(cooccur_count) * coverage DESC
    LIMIT 10
    """
    with kg.driver.session() as session:
        result = session.run(query,
                             confirmed_cuis=list(_confirmed),
                             denied_cuis=list(_denied),
                             asked_cuis=list(_asked))
        from src.umls_kg import SymptomCandidate
        return [SymptomCandidate(cui=r["cui"], name=r["name"],
                                 disease_coverage=r["disease_coverage"]) for r in result]


# ============================================================
# 종료 조건 (adaptive 모드용: top3_stable_5)
# ============================================================

def check_stopping(rank_history):
    if len(rank_history) >= 5:
        recent = list(rank_history)[-5:]
        return all(r == recent[0] for r in recent)
    return False


# ============================================================
# 진단
# ============================================================

def get_diagnosis(kg, confirmed_cuis, denied_cuis, top_k=10):
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
            raw = score_v15_ratio(r["confirmed_count"], r["denied_count"], r["total_symptoms"])
            candidates.append({"cui": r["cui"], "name": r["name"], "score": raw,
                               "confirmed_count": r["confirmed_count"],
                               "total_symptoms": r["total_symptoms"]})

    total = sum(c["score"] for c in candidates)
    for c in candidates:
        c["score"] = c["score"] / total if total > 0 else 0
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


# ============================================================
# 단일 환자 실행
# ============================================================

def run_single_patient(args):
    patient_data, loader_data, fixed_il, use_adaptive, neo4j_port = args

    from src.data_loader import DDXPlusLoader
    from src.umls_kg import UMLSKG

    loader = DDXPlusLoader()
    loader._symptom_mapping = loader_data["symptom_mapping"]
    loader._disease_mapping = loader_data["disease_mapping"]
    loader._fr_to_eng = loader_data["fr_to_eng"]
    loader._conditions = loader_data["conditions"]

    try:
        kg = UMLSKG(uri=f"bolt://localhost:{neo4j_port}")
    except Exception:
        return {"error": True, "reason": "neo4j_connect"}

    try:
        # 영어 질환명 → CUI (StreamBench은 영어 pathology)
        gt_disease_eng = patient_data["pathology_eng"]
        gt_cui = loader.get_disease_cui(gt_disease_eng)

        # 환자의 양성 증상 CUI 추출
        patient_positive_cuis = set()
        for ev_str in patient_data["evidences"]:
            code = ev_str.split("_@_")[0] if "_@_" in ev_str else ev_str
            cui = loader.get_symptom_cui(code)
            if cui:
                patient_positive_cuis.add(cui)

        initial_cui = loader.get_symptom_cui(patient_data["initial_evidence"])
        if not initial_cui:
            kg.close()
            return {"error": True, "reason": "no_initial_cui"}

        kg.reset_state()
        kg.state.add_confirmed(initial_cui)

        il = 0
        confirmed_count = 1
        rank_history = deque(maxlen=10)

        il_limit = fixed_il if fixed_il else MAX_IL

        for _ in range(il_limit):
            candidates = get_candidates(
                kg, initial_cui,
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

            # adaptive 모드에서만 종료 조건 체크
            if use_adaptive:
                kg_diag = kg.get_diagnosis_candidates(top_k=10)
                kg_dist = [(c.cui, c.score) for c in kg_diag] if kg_diag else []
                current_ranks = tuple(cui for cui, _ in kg_dist[:3])
                rank_history.append(current_ranks)

                if check_stopping(rank_history):
                    break

        # 최종 진단
        final = get_diagnosis(kg, kg.state.confirmed_cuis, kg.state.denied_cuis, top_k=10)
        correct_at_1 = final[0]["cui"] == gt_cui if final else False
        correct_at_10 = any(c["cui"] == gt_cui for c in final[:10]) if final else False

        kg.close()
        return {
            "error": False,
            "correct_at_1": int(correct_at_1),
            "correct_at_10": int(correct_at_10),
            "il": il,
            "confirmed": confirmed_count,
            "pathology": gt_disease_eng,
        }
    except Exception as e:
        kg.close()
        return {"error": True, "reason": str(e)}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fixed-il", type=int, help="고정 IL (5, 10, 15)")
    group.add_argument("--adaptive", action="store_true", help="adaptive 종료 (top3_stable_5)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ports", type=str, default="7687,7688,7689,7690")
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",")]

    # MEDDxAgent 동일 100건 추출 (HuggingFace에서 직접)
    patients_100 = load_meddxagent_100()

    mode = f"fixed_il{args.fixed_il}" if args.fixed_il else "adaptive"
    print(f"\n=== MEDDxAgent 비교: {mode}, {len(patients_100)}건 (seed=42) ===")

    from src.data_loader import DDXPlusLoader
    loader = DDXPlusLoader()

    loader_data = {
        "symptom_mapping": loader.symptom_mapping,
        "disease_mapping": loader.disease_mapping,
        "fr_to_eng": loader.fr_to_eng,
        "conditions": {k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
                       for k, v in loader.conditions.items()},
    }

    use_adaptive = args.adaptive
    fixed_il = args.fixed_il if args.fixed_il else 0

    tasks = [(pd, loader_data, fixed_il, use_adaptive, ports[i % len(ports)])
             for i, pd in enumerate(patients_100)]

    start = time.time()
    results, errors = [], 0
    error_reasons = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_single_patient, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc=mode) as pbar:
            for f in as_completed(futures):
                r = f.result()
                if r and not r.get("error"):
                    results.append(r)
                else:
                    errors += 1
                    if r:
                        error_reasons.append(r.get("reason", "unknown"))
                pbar.update(1)

    elapsed = time.time() - start
    count = len(results)
    ils = [r["il"] for r in results]

    gtpa_1 = sum(r["correct_at_1"] for r in results) / count if count else 0
    gtpa_10 = sum(r["correct_at_10"] for r in results) / count if count else 0

    output = {
        "mode": mode,
        "fixed_il": fixed_il if fixed_il else None,
        "adaptive": use_adaptive,
        "config": {
            "deny_threshold": DENY_THRESHOLD,
            "antecedent": USE_ANTECEDENT,
            "scoring": SCORING,
            "stopping": STOPPING if use_adaptive else "none (fixed)",
        },
        "sample": {
            "method": "MEDDxAgent identical: HuggingFace StreamBench DDXPlus test, seed=42 shuffle, first 100",
            "source": "appier-ai-research/StreamBench ddxplus test",
            "total_in_source": 1764,
            "count": count,
            "errors": errors,
            "error_reasons": error_reasons,
            "seed": 42,
        },
        "results": {
            "gtpa_1": gtpa_1,
            "gtpa_10": gtpa_10,
            "avg_il": float(np.mean(ils)) if ils else 0,
            "il_std": float(np.std(ils)) if ils else 0,
            "avg_confirmed": float(np.mean([r["confirmed"] for r in results])) if results else 0,
        },
        "elapsed": elapsed,
    }

    print(f"\nGTPA@1: {gtpa_1:.2%}, GTPA@10: {gtpa_10:.2%}")
    print(f"Avg IL: {np.mean(ils):.1f} (±{np.std(ils):.1f}), Confirmed: {np.mean([r['confirmed'] for r in results]):.1f}")
    if errors:
        print(f"Errors: {errors} ({error_reasons[:5]})")

    path = Path("results") / f"comparison_{mode}_100.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
