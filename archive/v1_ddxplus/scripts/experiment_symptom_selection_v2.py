#!/usr/bin/env python3
"""증상 선택 전략 비교 실험 v2.

2-hop으로 후보를 생성한 뒤, 어떤 기준으로 증상을 선택하는지 비교.
1,000건 샘플로 빠르게 비교.

=== 방법 분류 ===

1. Greedy (현재 방식)
   - greedy_cooccur: co-occurrence score (현재 기본)
   - greedy_coverage: disease coverage 기준

2. Information Gain (엔트로피 감소 최대화)
   - ig_expected: E[H_before - H_after] (기대 정보이득)
   - ig_max: max(IG_yes, IG_no) (최대 정보이득)
   - ig_binary_split: 후보 질환을 가장 균등하게 이분할

3. Minimax (최악의 경우 최적화)
   - minimax_score: min(score_yes, score_no) 최대화
   - minimax_entropy: max(entropy_yes, entropy_no) 최소화

4. Lookahead (N단계 앞 시뮬레이션)
   - lookahead_ig_1: 1단계 IG 기반 lookahead
   - lookahead_ig_3: 3단계 IG 기반 lookahead
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
STABILITY_K = 3
STABILITY_N = 5


def calc_entropy(scores):
    """정규화된 점수 리스트에서 Shannon 엔트로피 계산."""
    if not scores:
        return 0.0
    total = sum(scores)
    if total <= 0:
        return 0.0
    probs = [s / total for s in scores if s > 0]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def get_disease_distribution(kg, top_k=10):
    """현재 상태에서 질환 분포 반환."""
    candidates = kg.get_diagnosis_candidates(top_k=top_k)
    return [(c.cui, c.score) for c in candidates]


def select_symptom(kg, candidates, patient_positive_cuis, initial_cui, method):
    """방법에 따라 최적 증상 선택."""

    if method == "greedy_cooccur":
        return candidates[0].cui

    elif method == "greedy_coverage":
        return candidates[0].cui

    elif method.startswith("ig_"):
        return select_by_information_gain(kg, candidates, patient_positive_cuis, initial_cui, method)

    elif method.startswith("minimax_"):
        return select_by_minimax(kg, candidates, patient_positive_cuis, initial_cui, method)

    elif method.startswith("lookahead_ig_"):
        depth = int(method.split("_")[-1])
        return select_by_lookahead_ig(kg, candidates, patient_positive_cuis, initial_cui, depth)

    return candidates[0].cui


def select_by_information_gain(kg, candidates, patient_positive_cuis, initial_cui, method):
    """Information Gain 기반 증상 선택."""
    current_dist = get_disease_distribution(kg, top_k=10)
    current_entropy = calc_entropy([s for _, s in current_dist])

    best_cui = candidates[0].cui
    best_score = -float('inf')

    for candidate in candidates[:7]:
        saved_c = kg.state.confirmed_cuis.copy()
        saved_d = kg.state.denied_cuis.copy()
        saved_a = kg.state.asked_cuis.copy()

        # YES 시뮬레이션
        kg.state.confirmed_cuis = saved_c | {candidate.cui}
        kg.state.denied_cuis = saved_d.copy()
        kg.state.asked_cuis = saved_a | {candidate.cui}
        dist_yes = get_disease_distribution(kg, top_k=10)
        entropy_yes = calc_entropy([s for _, s in dist_yes])

        # NO 시뮬레이션
        kg.state.confirmed_cuis = saved_c.copy()
        kg.state.denied_cuis = saved_d | {candidate.cui}
        kg.state.asked_cuis = saved_a | {candidate.cui}
        dist_no = get_disease_distribution(kg, top_k=10)
        entropy_no = calc_entropy([s for _, s in dist_no])

        # 상태 복원
        kg.state.confirmed_cuis = saved_c
        kg.state.denied_cuis = saved_d
        kg.state.asked_cuis = saved_a

        n_diseases = len(current_dist) if current_dist else 1
        p_yes = candidate.disease_coverage / max(n_diseases, 1)
        p_yes = min(max(p_yes, 0.1), 0.9)
        p_no = 1 - p_yes

        if method == "ig_expected":
            expected_entropy = p_yes * entropy_yes + p_no * entropy_no
            score = current_entropy - expected_entropy
        elif method == "ig_max":
            score = max(current_entropy - entropy_yes, current_entropy - entropy_no)
        elif method == "ig_binary_split":
            score = -abs(p_yes - 0.5)
        else:
            score = 0

        if score > best_score:
            best_score = score
            best_cui = candidate.cui

    return best_cui


def select_by_minimax(kg, candidates, patient_positive_cuis, initial_cui, method):
    """Minimax 기반 증상 선택."""
    best_cui = candidates[0].cui
    best_score = -float('inf')

    for candidate in candidates[:7]:
        saved_c = kg.state.confirmed_cuis.copy()
        saved_d = kg.state.denied_cuis.copy()
        saved_a = kg.state.asked_cuis.copy()

        # YES
        kg.state.confirmed_cuis = saved_c | {candidate.cui}
        kg.state.denied_cuis = saved_d.copy()
        kg.state.asked_cuis = saved_a | {candidate.cui}
        dist_yes = get_disease_distribution(kg, top_k=10)
        score_yes = dist_yes[0][1] if dist_yes else 0
        entropy_yes = calc_entropy([s for _, s in dist_yes])

        # NO
        kg.state.confirmed_cuis = saved_c.copy()
        kg.state.denied_cuis = saved_d | {candidate.cui}
        kg.state.asked_cuis = saved_a | {candidate.cui}
        dist_no = get_disease_distribution(kg, top_k=10)
        score_no = dist_no[0][1] if dist_no else 0
        entropy_no = calc_entropy([s for _, s in dist_no])

        # 복원
        kg.state.confirmed_cuis = saved_c
        kg.state.denied_cuis = saved_d
        kg.state.asked_cuis = saved_a

        if method == "minimax_score":
            score = min(score_yes, score_no)
        elif method == "minimax_entropy":
            score = -max(entropy_yes, entropy_no)
        else:
            score = 0

        if score > best_score:
            best_score = score
            best_cui = candidate.cui

    return best_cui


def select_by_lookahead_ig(kg, candidates, patient_positive_cuis, initial_cui, depth):
    """IG 기반 lookahead."""
    best_cui = candidates[0].cui
    best_score = -float('inf')

    for candidate in candidates[:5]:
        saved_c = kg.state.confirmed_cuis.copy()
        saved_d = kg.state.denied_cuis.copy()
        saved_a = kg.state.asked_cuis.copy()

        score = lookahead_ig_simulate(
            kg, candidate.cui, patient_positive_cuis, initial_cui, depth,
            saved_c, saved_d, saved_a,
        )

        kg.state.confirmed_cuis = saved_c
        kg.state.denied_cuis = saved_d
        kg.state.asked_cuis = saved_a

        if score > best_score:
            best_score = score
            best_cui = candidate.cui

    return best_cui


def lookahead_ig_simulate(kg, symptom_cui, patient_positive_cuis, initial_cui, depth,
                          confirmed, denied, asked):
    """IG 기반 lookahead 시뮬레이션."""
    if symptom_cui in patient_positive_cuis:
        new_confirmed = confirmed | {symptom_cui}
        new_denied = denied
    else:
        new_confirmed = confirmed
        new_denied = denied | {symptom_cui}
    new_asked = asked | {symptom_cui}

    kg.state.confirmed_cuis = new_confirmed
    kg.state.denied_cuis = new_denied
    kg.state.asked_cuis = new_asked

    if depth <= 1:
        dist = get_disease_distribution(kg, top_k=10)
        return dist[0][1] if dist else 0.0

    next_candidates = kg.get_candidate_symptoms(
        initial_cui=initial_cui, limit=5,
        confirmed_cuis=new_confirmed, denied_cuis=new_denied,
    )
    if not next_candidates:
        dist = get_disease_distribution(kg, top_k=10)
        return dist[0][1] if dist else 0.0

    # IG 기준 상위 1개 선택
    current_dist = get_disease_distribution(kg, top_k=10)
    current_entropy = calc_entropy([s for _, s in current_dist])

    best_next = next_candidates[0]
    best_ig = -1

    for nc in next_candidates[:3]:
        saved_c2 = kg.state.confirmed_cuis.copy()
        saved_d2 = kg.state.denied_cuis.copy()
        saved_a2 = kg.state.asked_cuis.copy()

        kg.state.confirmed_cuis = saved_c2 | {nc.cui}
        kg.state.asked_cuis = saved_a2 | {nc.cui}
        dist_y = get_disease_distribution(kg, top_k=10)
        h_y = calc_entropy([s for _, s in dist_y])

        kg.state.confirmed_cuis = saved_c2.copy()
        kg.state.denied_cuis = saved_d2 | {nc.cui}
        kg.state.asked_cuis = saved_a2 | {nc.cui}
        dist_n = get_disease_distribution(kg, top_k=10)
        h_n = calc_entropy([s for _, s in dist_n])

        kg.state.confirmed_cuis = saved_c2
        kg.state.denied_cuis = saved_d2
        kg.state.asked_cuis = saved_a2

        ig = current_entropy - 0.5 * (h_y + h_n)
        if ig > best_ig:
            best_ig = ig
            best_next = nc

    return lookahead_ig_simulate(
        kg, best_next.cui, patient_positive_cuis, initial_cui, depth - 1,
        new_confirmed, new_denied, new_asked,
    )


def run_single_patient(args: tuple) -> dict:
    """단일 환자 진단."""
    patient_data, loader_data, method, neo4j_port = args

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
        rank_history = deque(maxlen=10)

        for _ in range(MAX_IL):
            candidates = kg.get_candidate_symptoms(
                initial_cui=initial_cui, limit=10,
                confirmed_cuis=kg.state.confirmed_cuis,
                denied_cuis=kg.state.denied_cuis,
            )
            if not candidates:
                break

            selected_cui = select_symptom(
                kg, candidates, patient_positive_cuis, initial_cui, method,
            )

            if selected_cui in patient_positive_cuis:
                kg.state.add_confirmed(selected_cui)
            else:
                kg.state.add_denied(selected_cui)

            il += 1

            diagnosis_candidates = kg.get_diagnosis_candidates(top_k=10)
            if not diagnosis_candidates:
                break

            current_ranks = tuple(c.cui for c in diagnosis_candidates[:STABILITY_K])
            rank_history.append(current_ranks)
            if len(rank_history) >= STABILITY_N:
                recent = list(rank_history)[-STABILITY_N:]
                if all(r == recent[0] for r in recent):
                    break

        final = kg.get_diagnosis_candidates(top_k=10)
        correct_at_1 = 0
        correct_at_10 = 0
        if final:
            if final[0].cui == gt_cui:
                correct_at_1 = 1
            for c in final[:10]:
                if c.cui == gt_cui:
                    correct_at_10 = 1
                    break

        kg.close()
        return {"error": False, "correct_at_1": correct_at_1, "correct_at_10": correct_at_10, "il": il}

    except Exception:
        kg.close()
        return {"error": True}


def main():
    all_methods = [
        "greedy_cooccur", "greedy_coverage",
        "ig_expected", "ig_max", "ig_binary_split",
        "minimax_score", "minimax_entropy",
        "lookahead_ig_1", "lookahead_ig_3",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=all_methods)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ports", type=str, default="7687,7688")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",")]
    print(f"=== 증상 선택: {args.method} ({args.n_samples}건) ===")

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

    tasks = [(pd, loader_data, args.method, ports[i % len(ports)]) for i, pd in enumerate(patients_data)]

    start_time = time.time()
    results = []
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_patient, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc=args.method) as pbar:
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
        "method": args.method, "n_samples": args.n_samples, "seed": args.seed,
        "count": count, "errors": errors,
        "gtpa_1": sum(r["correct_at_1"] for r in results) / count if count else 0,
        "gtpa_10": sum(r["correct_at_10"] for r in results) / count if count else 0,
        "avg_il": float(np.mean(ils)), "il_std": float(np.std(ils)),
        "elapsed": elapsed, "per_patient_sec": elapsed / count if count else 0,
    }

    print(f"\nGTPA@1:  {output['gtpa_1']:.2%}")
    print(f"GTPA@10: {output['gtpa_10']:.2%}")
    print(f"Avg IL:  {output['avg_il']:.1f}")
    print(f"Time:    {elapsed:.1f}s ({output['per_patient_sec']:.2f}s/pt)")

    output_path = Path("results") / f"symsel_{args.method}_{args.n_samples}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
