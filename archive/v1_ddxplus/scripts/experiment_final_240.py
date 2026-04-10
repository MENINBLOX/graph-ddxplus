#!/usr/bin/env python3
"""최종 실험: 탐색 3 × 종료 10 × 스코어링 8 = 240개 조합.

134,529건 전체 평가.

사용법: python experiment_final_240.py \
          --exploration greedy_cooccur \
          --stopping consecutive_miss_5 \
          --scoring v15_ratio \
          --workers 4 --ports "7687,7688,7689,7690"
"""

import argparse
import json
import math
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

MAX_IL = 223


# ============================================================
# 진단 스코어링 함수들
# ============================================================

def score_v15_ratio(confirmed, denied, total):
    """Evidence Ratio: c/(c+d+1) × c"""
    c, d = float(confirmed), float(denied)
    return (c / (c + d + 1.0)) * c


def score_v18_coverage(confirmed, denied, total):
    """Coverage: c/(total+1) × c"""
    c, t = float(confirmed), float(total)
    return (c / (t + 1.0)) * c


def score_naive_bayes(confirmed, denied, total):
    """Naive Bayes approximation: log P(D|symptoms) ∝ c×log(c/total) - d×log(d/total)
    Simplified: c × log(c/(total+1)) - d × log(d/(total+1))"""
    c, d, t = float(confirmed), float(denied), float(total)
    if c == 0:
        return 0.0
    # P(confirmed|D) ≈ c/total, P(denied|D) ≈ 1 - d/total
    p_pos = c / (t + 1.0)
    p_neg = 1.0 - d / (t + 1.0) if t > 0 else 1.0
    score = c * math.log(p_pos + 1e-10) + (t - d) * math.log(p_neg + 1e-10)
    return max(score, 0.0)


def score_log_likelihood(confirmed, denied, total):
    """Log-Likelihood Ratio: Σ log(P(s|D)/P(s|¬D))
    Simplified: c × log(prevalence) - d × log(1-prevalence)"""
    c, d, t = float(confirmed), float(denied), float(total)
    if c == 0 or t == 0:
        return 0.0
    prevalence = c / (t + 1.0)
    anti_prevalence = d / (t + 1.0)
    lr = c * math.log(prevalence + 1e-10) - d * math.log(anti_prevalence + 1e-10)
    return max(lr, 0.0)


def score_jaccard(confirmed, denied, total):
    """Jaccard: c / (c + d + unasked) where unasked = total - c - d"""
    c, d, t = float(confirmed), float(denied), float(total)
    unasked = max(t - c - d, 0)
    denom = c + d + unasked
    return (c / denom) * c if denom > 0 else 0.0


def score_tfidf(confirmed, denied, total):
    """TF-IDF inspired: tf(c) × idf(c)
    tf = c/total (term frequency), idf = log(1 + 1/(prevalence+0.01))"""
    c, d, t = float(confirmed), float(denied), float(total)
    if c == 0 or t == 0:
        return 0.0
    tf = c / (t + 1.0)
    # idf: 희귀한 증상 매칭일수록 높은 가치
    prevalence = t / 223.0  # 전체 증상 대비 이 질환의 증상 비율
    idf = math.log(1.0 + 1.0 / (prevalence + 0.01))
    return tf * idf * c


def score_bm25(confirmed, denied, total, k1=1.5, b=0.75, avgdl=15.6):
    """BM25: Okapi BM25 adaptation.
    tf = confirmed, dl = total, avgdl = 평균 증상 수"""
    c, d, t = float(confirmed), float(denied), float(total)
    if c == 0:
        return 0.0
    dl = t
    tf_norm = (c * (k1 + 1)) / (c + k1 * (1 - b + b * dl / avgdl))
    # idf component: log((N - df + 0.5) / (df + 0.5)) where N=49, df=diseases with this symptom
    # simplified: use confirmed as relevance signal
    idf = math.log(1.0 + (49.0 - c) / (c + 0.5))
    return max(tf_norm * idf, 0.0)


def score_cosine(confirmed, denied, total):
    """Cosine Similarity: dot(confirmed_vec, disease_vec) / norms
    Simplified: c / sqrt(c+d) / sqrt(total)"""
    c, d, t = float(confirmed), float(denied), float(total)
    if c == 0 or t == 0:
        return 0.0
    query_norm = math.sqrt(c + d) if (c + d) > 0 else 1.0
    doc_norm = math.sqrt(t) if t > 0 else 1.0
    return (c / (query_norm * doc_norm)) * c


SCORING_FUNCTIONS = {
    "v15_ratio": score_v15_ratio,
    "v18_coverage": score_v18_coverage,
    "naive_bayes": score_naive_bayes,
    "log_likelihood": score_log_likelihood,
    "jaccard": score_jaccard,
    "tfidf": score_tfidf,
    "bm25": score_bm25,
    "cosine": score_cosine,
}


# ============================================================
# 증상 탐색
# ============================================================

def calc_entropy(scores):
    if not scores:
        return 0.0
    total = sum(scores)
    if total <= 0:
        return 0.0
    probs = [s / total for s in scores if s > 0]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def get_disease_distribution(kg, top_k=10):
    candidates = kg.get_diagnosis_candidates(top_k=top_k)
    return [(c.cui, c.score) for c in candidates]


def select_symptom(kg, candidates, patient_positive_cuis, initial_cui, exploration):
    if exploration == "greedy_cooccur":
        return candidates[0].cui
    elif exploration == "ig_expected":
        return select_ig_expected(kg, candidates)
    elif exploration == "minimax_score":
        return select_minimax_score(kg, candidates)
    return candidates[0].cui


def select_ig_expected(kg, candidates):
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
        score = current_entropy - (py * ey + (1 - py) * en)
        if score > best_score:
            best_score, best_cui = score, c.cui
    return best_cui


def select_minimax_score(kg, candidates):
    best_cui, best_score = candidates[0].cui, -float('inf')
    for c in candidates[:7]:
        sc, sd, sa = kg.state.confirmed_cuis.copy(), kg.state.denied_cuis.copy(), kg.state.asked_cuis.copy()
        kg.state.confirmed_cuis = sc | {c.cui}; kg.state.asked_cuis = sa | {c.cui}
        dy = get_disease_distribution(kg, top_k=10)
        sy = dy[0][1] if dy else 0
        kg.state.confirmed_cuis = sc.copy(); kg.state.denied_cuis = sd | {c.cui}; kg.state.asked_cuis = sa | {c.cui}
        dn = get_disease_distribution(kg, top_k=10)
        sn = dn[0][1] if dn else 0
        kg.state.confirmed_cuis, kg.state.denied_cuis, kg.state.asked_cuis = sc, sd, sa
        score = min(sy, sn)
        if score > best_score:
            best_score, best_cui = score, c.cui
    return best_cui


# ============================================================
# 종료 조건
# ============================================================

def check_stopping(stopping, il, step_hits, confirmed_count, kg, rank_history, diagnosis_candidates):
    """종료 조건 확인."""

    # --- 스코어링-독립적 ---
    if stopping == "consecutive_miss_3":
        return len(step_hits) >= 3 and all(h == 0 for h in step_hits[-3:])
    if stopping == "consecutive_miss_5":
        return len(step_hits) >= 5 and all(h == 0 for h in step_hits[-5:])
    if stopping == "consecutive_miss_7":
        return len(step_hits) >= 7 and all(h == 0 for h in step_hits[-7:])

    if stopping == "marginal_hr_5_10":
        return len(step_hits) >= 5 and sum(step_hits[-5:]) / 5 < 0.1
    if stopping == "marginal_hr_5_20":
        return len(step_hits) >= 5 and sum(step_hits[-5:]) / 5 < 0.2
    if stopping == "marginal_hr_10_10":
        return len(step_hits) >= 10 and sum(step_hits[-10:]) / 10 < 0.1

    if stopping == "cumulative_confirmed_3":
        return confirmed_count >= 3
    if stopping == "cumulative_confirmed_5":
        return confirmed_count >= 5
    if stopping == "cumulative_confirmed_7":
        return confirmed_count >= 7

    # hit rate plateau: 최근 10개의 기울기가 0에 가까움
    if stopping == "hr_plateau":
        if len(step_hits) >= 10:
            recent = step_hits[-10:]
            first_half = sum(recent[:5])
            second_half = sum(recent[5:])
            return second_half <= first_half and second_half == 0
        return False

    # --- 스코어링-의존적 ---
    if stopping == "top1_stable_5":
        if len(rank_history) >= 5:
            recent = [r[0] for r in list(rank_history)[-5:]]
            return all(r == recent[0] for r in recent)
        return False

    if stopping == "top3_stable_5":
        if len(rank_history) >= 5:
            recent = list(rank_history)[-5:]
            return all(r == recent[0] for r in recent)
        return False

    if stopping == "confidence_03":
        return diagnosis_candidates and diagnosis_candidates[0][1] >= 0.3

    if stopping == "confidence_05":
        return diagnosis_candidates and diagnosis_candidates[0][1] >= 0.5

    if stopping == "conf_gap_005":
        if diagnosis_candidates and len(diagnosis_candidates) >= 2:
            return (diagnosis_candidates[0][1] - diagnosis_candidates[1][1]) >= 0.05
        return diagnosis_candidates and len(diagnosis_candidates) == 1

    if stopping == "conf_gap_01":
        if diagnosis_candidates and len(diagnosis_candidates) >= 2:
            return (diagnosis_candidates[0][1] - diagnosis_candidates[1][1]) >= 0.1
        return diagnosis_candidates and len(diagnosis_candidates) == 1

    if stopping == "entropy_10":
        if diagnosis_candidates:
            h = calc_entropy([s for _, s in diagnosis_candidates])
            return h < 1.0
        return False

    if stopping == "entropy_20":
        if diagnosis_candidates:
            h = calc_entropy([s for _, s in diagnosis_candidates])
            return h < 2.0
        return False

    if stopping == "ig_001_2":
        # 정보이득 < 0.01이 2회 연속이면 종료
        # rank_history에 entropy를 저장해야 하지만, 간소화
        return False  # 별도 구현 필요

    return False


# ============================================================
# 커스텀 진단 수행
# ============================================================

def get_custom_diagnosis(kg, scoring_name, confirmed_cuis, denied_cuis, top_k=10):
    """커스텀 스코어링 함수로 진단 수행."""
    scoring_fn = SCORING_FUNCTIONS[scoring_name]

    # KG에서 후보 질환과 증상 매칭 정보 가져오기
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
        result = session.run(
            query,
            confirmed_cuis=list(confirmed_cuis),
            denied_cuis=list(denied_cuis),
        )
        candidates = []
        for r in result:
            raw_score = scoring_fn(r["confirmed_count"], r["denied_count"], r["total_symptoms"])
            candidates.append({
                "cui": r["cui"],
                "name": r["name"],
                "score": raw_score,
                "confirmed_count": r["confirmed_count"],
                "total_symptoms": r["total_symptoms"],
            })

    # 정규화
    total_score = sum(c["score"] for c in candidates)
    for c in candidates:
        c["score"] = c["score"] / total_score if total_score > 0 else 0

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


# ============================================================
# 메인 실행
# ============================================================

def run_single_patient(args: tuple) -> dict:
    patient_data, loader_data, exploration, stopping, scoring, neo4j_port = args

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
            candidates = kg.get_candidate_symptoms(
                initial_cui=initial_cui, limit=10,
                confirmed_cuis=kg.state.confirmed_cuis,
                denied_cuis=kg.state.denied_cuis,
            )
            if not candidates:
                break

            selected_cui = select_symptom(kg, candidates, patient_positive_cuis, initial_cui, exploration)

            hit = 1 if selected_cui in patient_positive_cuis else 0
            if hit:
                kg.state.add_confirmed(selected_cui)
                confirmed_count += 1
            else:
                kg.state.add_denied(selected_cui)

            il += 1
            step_hits.append(hit)

            # 종료 조건 판단: 기존 kg.get_diagnosis_candidates() 사용 (v15_ratio 기본)
            # 종료 시점은 스코어링 방법과 독립적으로 동일해야 함
            kg_diag = kg.get_diagnosis_candidates(top_k=10)
            kg_dist = [(c.cui, c.score) for c in kg_diag] if kg_diag else []

            # rank history (기존 스코어링 기준)
            current_ranks = tuple(cui for cui, _ in kg_dist[:3])
            rank_history.append(current_ranks)

            if check_stopping(stopping, il, step_hits, confirmed_count, kg, rank_history, kg_dist):
                break

        # 최종 진단: 커스텀 스코어링으로 평가
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
    explorations = ["greedy_cooccur", "ig_expected", "minimax_score"]
    stoppings = [
        "consecutive_miss_3", "consecutive_miss_5", "consecutive_miss_7",
        "marginal_hr_5_10", "marginal_hr_5_20", "marginal_hr_10_10",
        "cumulative_confirmed_3", "cumulative_confirmed_5", "cumulative_confirmed_7",
        "hr_plateau",
        "top1_stable_5", "top3_stable_5",
        "confidence_03", "confidence_05",
        "conf_gap_005", "conf_gap_01",
        "entropy_10", "entropy_20",
    ]
    scorings = list(SCORING_FUNCTIONS.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration", required=True, choices=explorations)
    parser.add_argument("--stopping", required=True, choices=stoppings)
    parser.add_argument("--scoring", required=True, choices=scorings)
    parser.add_argument("--n-samples", type=int, default=None, help="None=전체")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ports", type=str, default="7687,7688,7689,7690")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    method_name = f"{args.exploration}+{args.stopping}+{args.scoring}"
    ports = [int(p.strip()) for p in args.ports.split(",")]

    from src.data_loader import DDXPlusLoader
    loader = DDXPlusLoader()
    all_patients = loader.load_patients(split="test")

    if args.n_samples:
        random.seed(args.seed)
        indices = random.sample(range(len(all_patients)), min(args.n_samples, len(all_patients)))
        patients = [all_patients[i] for i in indices]
    else:
        patients = all_patients

    print(f"=== {method_name} ({len(patients):,}건) ===")

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

    tasks = [(pd, loader_data, args.exploration, args.stopping, args.scoring, ports[i % len(ports)])
             for i, pd in enumerate(patients_data)]

    start_time = time.time()
    results = []
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_patient, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc=method_name[:40]) as pbar:
            for future in as_completed(futures):
                r = future.result()
                if r and not r.get("error"):
                    results.append(r)
                else:
                    errors += 1
                pbar.update(1)

    elapsed = time.time() - start_time
    count = len(results)
    ils = [r["il"] for r in results]

    output = {
        "exploration": args.exploration,
        "stopping": args.stopping,
        "scoring": args.scoring,
        "method": method_name,
        "n_samples": len(patients),
        "seed": args.seed if args.n_samples else None,
        "count": count, "errors": errors,
        "gtpa_1": sum(r["correct_at_1"] for r in results) / count if count else 0,
        "gtpa_10": sum(r["correct_at_10"] for r in results) / count if count else 0,
        "avg_il": float(np.mean(ils)),
        "il_std": float(np.std(ils)),
        "avg_confirmed": float(np.mean([r["confirmed"] for r in results])),
        "elapsed": elapsed,
    }

    print(f"\nGTPA@1:    {output['gtpa_1']:.2%}")
    print(f"GTPA@10:   {output['gtpa_10']:.2%}")
    print(f"Avg IL:    {output['avg_il']:.1f}")
    print(f"Confirmed: {output['avg_confirmed']:.1f}")
    print(f"Time:      {elapsed:.0f}s")

    safe_name = method_name.replace("+", "_")
    output_path = Path("results") / f"final_{safe_name}_{len(patients)}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
