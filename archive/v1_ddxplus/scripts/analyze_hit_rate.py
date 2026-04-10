#!/usr/bin/env python3
"""증상 탐색 전략별 Hit Rate 분석.

Hit Rate = confirmed / (confirmed + denied) = 시스템이 질문한 증상 중 환자가 "예"한 비율.
Top3_stable_5 자연 종료 시점까지 측정.
1,000건 샘플 (seed=42).

사용법: python analyze_hit_rate.py --method greedy_cooccur --workers 2 --ports "7687,7688"
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


def select_symptom(kg, candidates, patient_positive_cuis, initial_cui, method):
    if method in ("greedy_cooccur", "greedy_coverage"):
        return candidates[0].cui
    elif method == "ig_expected":
        return select_ig(kg, candidates, "ig_expected")
    elif method == "ig_max":
        return select_ig(kg, candidates, "ig_max")
    elif method == "minimax_score":
        return select_minimax(kg, candidates, "minimax_score")
    elif method == "minimax_entropy":
        return select_minimax(kg, candidates, "minimax_entropy")
    return candidates[0].cui


def select_ig(kg, candidates, method):
    current_dist = get_disease_distribution(kg, top_k=10)
    current_entropy = calc_entropy([s for _, s in current_dist])
    best_cui = candidates[0].cui
    best_score = -float('inf')

    for candidate in candidates[:7]:
        saved_c = kg.state.confirmed_cuis.copy()
        saved_d = kg.state.denied_cuis.copy()
        saved_a = kg.state.asked_cuis.copy()

        kg.state.confirmed_cuis = saved_c | {candidate.cui}
        kg.state.denied_cuis = saved_d.copy()
        kg.state.asked_cuis = saved_a | {candidate.cui}
        entropy_yes = calc_entropy([s for _, s in get_disease_distribution(kg, top_k=10)])

        kg.state.confirmed_cuis = saved_c.copy()
        kg.state.denied_cuis = saved_d | {candidate.cui}
        kg.state.asked_cuis = saved_a | {candidate.cui}
        entropy_no = calc_entropy([s for _, s in get_disease_distribution(kg, top_k=10)])

        kg.state.confirmed_cuis = saved_c
        kg.state.denied_cuis = saved_d
        kg.state.asked_cuis = saved_a

        n_diseases = len(current_dist) if current_dist else 1
        p_yes = min(max(candidate.disease_coverage / max(n_diseases, 1), 0.1), 0.9)

        if method == "ig_expected":
            score = current_entropy - (p_yes * entropy_yes + (1 - p_yes) * entropy_no)
        else:  # ig_max
            score = max(current_entropy - entropy_yes, current_entropy - entropy_no)

        if score > best_score:
            best_score = score
            best_cui = candidate.cui
    return best_cui


def select_minimax(kg, candidates, method):
    best_cui = candidates[0].cui
    best_score = -float('inf')

    for candidate in candidates[:7]:
        saved_c = kg.state.confirmed_cuis.copy()
        saved_d = kg.state.denied_cuis.copy()
        saved_a = kg.state.asked_cuis.copy()

        kg.state.confirmed_cuis = saved_c | {candidate.cui}
        kg.state.denied_cuis = saved_d.copy()
        kg.state.asked_cuis = saved_a | {candidate.cui}
        dist_yes = get_disease_distribution(kg, top_k=10)
        score_yes = dist_yes[0][1] if dist_yes else 0
        entropy_yes = calc_entropy([s for _, s in dist_yes])

        kg.state.confirmed_cuis = saved_c.copy()
        kg.state.denied_cuis = saved_d | {candidate.cui}
        kg.state.asked_cuis = saved_a | {candidate.cui}
        dist_no = get_disease_distribution(kg, top_k=10)
        score_no = dist_no[0][1] if dist_no else 0
        entropy_no = calc_entropy([s for _, s in dist_no])

        kg.state.confirmed_cuis = saved_c
        kg.state.denied_cuis = saved_d
        kg.state.asked_cuis = saved_a

        if method == "minimax_score":
            score = min(score_yes, score_no)
        else:
            score = -max(entropy_yes, entropy_no)

        if score > best_score:
            best_score = score
            best_cui = candidate.cui
    return best_cui


def run_single_patient(args: tuple) -> dict:
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
        confirmed_count = 1  # 초기 증상 포함
        denied_count = 0
        rank_history = deque(maxlen=10)

        # IL별 confirmed 추적 (5, 10, 15, 20 시점)
        confirmed_at = {}

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
                confirmed_count += 1
            else:
                kg.state.add_denied(selected_cui)
                denied_count += 1

            il += 1

            # 체크포인트 기록
            if il in (5, 10, 15, 20):
                confirmed_at[il] = confirmed_count

            # Top3_stable_5 종료
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
        correct_at_1 = final[0].cui == gt_cui if final else False

        total_patient_symptoms = len(patient_positive_cuis)

        kg.close()
        return {
            "error": False,
            "correct_at_1": int(correct_at_1),
            "il": il,
            "confirmed": confirmed_count,
            "denied": denied_count,
            "hit_rate": confirmed_count / (confirmed_count + denied_count) if (confirmed_count + denied_count) > 0 else 0,
            "recall": confirmed_count / total_patient_symptoms if total_patient_symptoms > 0 else 0,
            "total_patient_symptoms": total_patient_symptoms,
            "confirmed_at": confirmed_at,
        }

    except Exception:
        kg.close()
        return {"error": True}


def main():
    all_methods = [
        "greedy_cooccur", "greedy_coverage",
        "ig_expected", "ig_max",
        "minimax_score", "minimax_entropy",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=all_methods)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ports", type=str, default="7687,7688")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",")]
    print(f"=== Hit Rate 분석: {args.method} ({args.n_samples}건) ===")

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

    # 집계
    avg_hit_rate = np.mean([r["hit_rate"] for r in results])
    avg_recall = np.mean([r["recall"] for r in results])
    avg_confirmed = np.mean([r["confirmed"] for r in results])
    avg_denied = np.mean([r["denied"] for r in results])
    avg_il = np.mean([r["il"] for r in results])
    gtpa_1 = sum(r["correct_at_1"] for r in results) / count

    # IL별 confirmed 평균
    confirmed_at_avg = {}
    for checkpoint in [5, 10, 15, 20]:
        vals = [r["confirmed_at"].get(str(checkpoint) if isinstance(list(r["confirmed_at"].keys())[0] if r["confirmed_at"] else "5", str) else checkpoint, None)
                for r in results if r["confirmed_at"]]
        # simpler approach
        vals = []
        for r in results:
            ca = r.get("confirmed_at", {})
            v = ca.get(checkpoint) or ca.get(str(checkpoint))
            if v is not None:
                vals.append(v)
        if vals:
            confirmed_at_avg[checkpoint] = float(np.mean(vals))

    output = {
        "method": args.method,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "count": count,
        "errors": errors,
        "gtpa_1": gtpa_1,
        "avg_il": float(avg_il),
        "avg_confirmed": float(avg_confirmed),
        "avg_denied": float(avg_denied),
        "avg_hit_rate": float(avg_hit_rate),
        "avg_recall": float(avg_recall),
        "confirmed_at_il": confirmed_at_avg,
        "elapsed": elapsed,
    }

    print(f"\nGTPA@1:       {gtpa_1:.2%}")
    print(f"Avg IL:       {avg_il:.1f}")
    print(f"Avg Confirmed:{avg_confirmed:.1f}")
    print(f"Avg Denied:   {avg_denied:.1f}")
    print(f"Hit Rate:     {avg_hit_rate:.2%}")
    print(f"Recall:       {avg_recall:.2%}")
    for k, v in confirmed_at_avg.items():
        print(f"  Confirmed@IL={k}: {v:.1f}")

    output_path = Path("results") / f"hitrate_{args.method}_{args.n_samples}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
