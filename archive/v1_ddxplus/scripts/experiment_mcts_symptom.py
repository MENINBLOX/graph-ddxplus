#!/usr/bin/env python3
"""MCTS 기반 증상 선택 실험.

현재 greedy (top-1) 대비 lookahead 기반 증상 선택의 효과를 검증.
1,000건 샘플로 빠르게 비교.

방법:
1. greedy: 현재 방식 (top-1 discrimination score)
2. lookahead_1: 1단계 앞 시뮬레이션 (yes/no 양방향)
3. lookahead_3: 3단계 앞 시뮬레이션
4. lookahead_5: 5단계 앞 시뮬레이션
5. mcts_random: MCTS 랜덤 롤아웃 (5단계, 10회 시뮬레이션)
"""

import argparse
import copy
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
            age=patient_data["age"],
            sex=patient_data["sex"],
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
                initial_cui=initial_cui,
                limit=10,
                confirmed_cuis=kg.state.confirmed_cuis,
                denied_cuis=kg.state.denied_cuis,
            )
            if not candidates:
                break

            # === 증상 선택 방법 ===
            if method == "greedy":
                selected_cui = candidates[0].cui
            else:
                selected_cui = select_with_lookahead(
                    kg, candidates, patient_positive_cuis,
                    initial_cui, method,
                )

            # 환자 응답
            if selected_cui in patient_positive_cuis:
                kg.state.add_confirmed(selected_cui)
            else:
                kg.state.add_denied(selected_cui)

            il += 1

            # Top3_stable_5 종료 조건
            diagnosis_candidates = kg.get_diagnosis_candidates(top_k=10)
            if not diagnosis_candidates:
                break

            current_ranks = tuple(c.cui for c in diagnosis_candidates[:STABILITY_K])
            rank_history.append(current_ranks)
            if len(rank_history) >= STABILITY_N:
                recent = list(rank_history)[-STABILITY_N:]
                if all(r == recent[0] for r in recent):
                    break

        # 최종 진단
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
        return {
            "error": False,
            "correct_at_1": correct_at_1,
            "correct_at_10": correct_at_10,
            "il": il,
        }

    except Exception as e:
        kg.close()
        return {"error": True, "msg": str(e)}


def select_with_lookahead(kg, candidates, patient_positive_cuis, initial_cui, method):
    """Lookahead 기반 증상 선택."""
    if method.startswith("lookahead_"):
        depth = int(method.split("_")[1])
        return lookahead_select(kg, candidates, patient_positive_cuis, initial_cui, depth)
    elif method.startswith("mcts_"):
        n_simulations = int(method.split("_")[1]) if len(method.split("_")) > 1 and method.split("_")[1].isdigit() else 10
        return mcts_select(kg, candidates, patient_positive_cuis, initial_cui, depth=5, n_simulations=n_simulations)
    else:
        return candidates[0].cui


def lookahead_select(kg, candidates, patient_positive_cuis, initial_cui, depth):
    """양방향 lookahead: 각 후보에 대해 yes/no 시뮬레이션 후 최적 선택.

    평가 기준: 시뮬레이션 후 top-1 진단의 score (높을수록 좋음)
    """
    best_cui = candidates[0].cui
    best_score = -1.0

    # 상위 5개 후보만 시뮬레이션 (시간 절약)
    for candidate in candidates[:5]:
        # 현재 상태 저장
        saved_confirmed = kg.state.confirmed_cuis.copy()
        saved_denied = kg.state.denied_cuis.copy()
        saved_asked = kg.state.asked_cuis.copy()

        score = simulate_both_branches(
            kg, candidate.cui, patient_positive_cuis,
            initial_cui, depth, saved_confirmed, saved_denied, saved_asked,
        )

        # 상태 복원
        kg.state.confirmed_cuis = saved_confirmed
        kg.state.denied_cuis = saved_denied
        kg.state.asked_cuis = saved_asked

        if score > best_score:
            best_score = score
            best_cui = candidate.cui

    return best_cui


def simulate_both_branches(kg, symptom_cui, patient_positive_cuis, initial_cui, depth,
                           confirmed, denied, asked):
    """yes/no 양방향 시뮬레이션. 실제 환자 응답 사용."""
    if symptom_cui in patient_positive_cuis:
        # 환자가 이 증상을 가지고 있음
        new_confirmed = confirmed | {symptom_cui}
        new_denied = denied
    else:
        new_confirmed = confirmed
        new_denied = denied | {symptom_cui}

    new_asked = asked | {symptom_cui}

    if depth <= 1:
        # 말단: 진단 점수 평가
        kg.state.confirmed_cuis = new_confirmed
        kg.state.denied_cuis = new_denied
        kg.state.asked_cuis = new_asked
        diag = kg.get_diagnosis_candidates(top_k=3)
        if diag:
            return diag[0].score
        return 0.0

    # 재귀: 다음 단계
    kg.state.confirmed_cuis = new_confirmed
    kg.state.denied_cuis = new_denied
    kg.state.asked_cuis = new_asked

    next_candidates = kg.get_candidate_symptoms(
        initial_cui=initial_cui,
        limit=5,
        confirmed_cuis=new_confirmed,
        denied_cuis=new_denied,
    )

    if not next_candidates:
        diag = kg.get_diagnosis_candidates(top_k=3)
        return diag[0].score if diag else 0.0

    # greedy로 다음 증상 선택하여 재귀
    best_next = next_candidates[0]
    return simulate_both_branches(
        kg, best_next.cui, patient_positive_cuis, initial_cui, depth - 1,
        new_confirmed, new_denied, new_asked,
    )


def mcts_select(kg, candidates, patient_positive_cuis, initial_cui, depth=5, n_simulations=10):
    """MCTS 랜덤 롤아웃 기반 증상 선택.

    각 후보에 대해 n_simulations 횟수만큼 랜덤 롤아웃 후 평균 점수 비교.
    """
    best_cui = candidates[0].cui
    best_score = -1.0

    for candidate in candidates[:5]:
        total_score = 0.0

        for _ in range(n_simulations):
            saved_confirmed = kg.state.confirmed_cuis.copy()
            saved_denied = kg.state.denied_cuis.copy()
            saved_asked = kg.state.asked_cuis.copy()

            score = mcts_rollout(
                kg, candidate.cui, patient_positive_cuis,
                initial_cui, depth, saved_confirmed, saved_denied, saved_asked,
            )

            kg.state.confirmed_cuis = saved_confirmed
            kg.state.denied_cuis = saved_denied
            kg.state.asked_cuis = saved_asked

            total_score += score

        avg_score = total_score / n_simulations
        if avg_score > best_score:
            best_score = avg_score
            best_cui = candidate.cui

    return best_cui


def mcts_rollout(kg, symptom_cui, patient_positive_cuis, initial_cui, depth,
                 confirmed, denied, asked):
    """MCTS 롤아웃: 실제 환자 응답 사용, 이후 랜덤 선택."""
    # 첫 번째 증상은 지정된 것 사용
    if symptom_cui in patient_positive_cuis:
        new_confirmed = confirmed | {symptom_cui}
        new_denied = denied
    else:
        new_confirmed = confirmed
        new_denied = denied | {symptom_cui}
    new_asked = asked | {symptom_cui}

    # 나머지 depth-1 단계는 랜덤 선택
    for _ in range(depth - 1):
        kg.state.confirmed_cuis = new_confirmed
        kg.state.denied_cuis = new_denied
        kg.state.asked_cuis = new_asked

        next_candidates = kg.get_candidate_symptoms(
            initial_cui=initial_cui,
            limit=5,
            confirmed_cuis=new_confirmed,
            denied_cuis=new_denied,
        )
        if not next_candidates:
            break

        # 랜덤 선택 (상위 5개 중)
        chosen = random.choice(next_candidates)
        if chosen.cui in patient_positive_cuis:
            new_confirmed = new_confirmed | {chosen.cui}
        else:
            new_denied = new_denied | {chosen.cui}
        new_asked = new_asked | {chosen.cui}

    # 최종 진단 평가
    kg.state.confirmed_cuis = new_confirmed
    kg.state.denied_cuis = new_denied
    kg.state.asked_cuis = new_asked
    diag = kg.get_diagnosis_candidates(top_k=3)
    return diag[0].score if diag else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["greedy", "lookahead_1", "lookahead_3", "lookahead_5",
                                 "mcts_5", "mcts_10", "mcts_20"])
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ports", type=str, default="7687,7688")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",")]
    print(f"=== MCTS 실험: {args.method} ({args.n_samples}건) ===")

    from src.data_loader import DDXPlusLoader

    loader = DDXPlusLoader()
    all_patients = loader.load_patients(split="test")

    # 고정 시드로 샘플링
    random.seed(args.seed)
    indices = random.sample(range(len(all_patients)), min(args.n_samples, len(all_patients)))
    patients = [all_patients[i] for i in indices]
    print(f"샘플: {len(patients)}건 (seed={args.seed})")

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
        {
            "age": p.age, "sex": p.sex,
            "initial_evidence": p.initial_evidence,
            "evidences": p.evidences,
            "pathology": p.pathology,
            "differential_diagnosis": p.differential_diagnosis,
        }
        for p in patients
    ]

    tasks = [
        (pd, loader_data, args.method, ports[i % len(ports)])
        for i, pd in enumerate(patients_data)
    ]

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
        "method": args.method,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "count": count,
        "errors": errors,
        "gtpa_1": sum(r["correct_at_1"] for r in results) / count if count else 0,
        "gtpa_10": sum(r["correct_at_10"] for r in results) / count if count else 0,
        "avg_il": float(np.mean(ils)),
        "il_std": float(np.std(ils)),
        "elapsed": elapsed,
        "per_patient_sec": elapsed / count if count else 0,
    }

    print(f"\nGTPA@1:  {output['gtpa_1']:.2%}")
    print(f"GTPA@10: {output['gtpa_10']:.2%}")
    print(f"Avg IL:  {output['avg_il']:.1f}")
    print(f"Time:    {elapsed:.1f}s ({output['per_patient_sec']:.2f}s/patient)")

    output_path = Path("results") / f"mcts_{args.method}_{args.n_samples}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
