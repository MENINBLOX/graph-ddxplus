#!/usr/bin/env python3
"""증상 탐색 전략별 Hit Rate 곡선 분석.

종료 조건 없이 후보 증상이 소진될 때까지 매 IL마다 기록.
1,000건 샘플 (seed=42).

사용법: python analyze_hit_rate_curve.py --method greedy_cooccur --workers 2 --ports "7687,7688"
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

MAX_IL = 100  # 충분히 큰 값 (후보 소진 시 자연 종료)


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
    elif method == "ig_binary_split":
        return select_ig(kg, candidates, "ig_binary_split")
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

        # 매 IL마다 누적 confirmed/denied 기록
        cumulative_confirmed = [1]  # IL=0: 초기 증상
        cumulative_denied = [0]
        step_hit = []  # 각 step에서 hit(1) or miss(0)

        for step in range(MAX_IL):
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
                step_hit.append(1)
            else:
                kg.state.add_denied(selected_cui)
                step_hit.append(0)

            cumulative_confirmed.append(cumulative_confirmed[-1] + step_hit[-1])
            cumulative_denied.append(cumulative_denied[-1] + (1 - step_hit[-1]))

        kg.close()

        total_il = len(step_hit)
        total_confirmed = cumulative_confirmed[-1]
        total_denied = cumulative_denied[-1]

        return {
            "error": False,
            "il": total_il,
            "total_confirmed": total_confirmed,
            "total_denied": total_denied,
            "total_patient_symptoms": len(patient_positive_cuis),
            "cumulative_confirmed": cumulative_confirmed,
            "cumulative_denied": cumulative_denied,
            "step_hit": step_hit,
        }

    except Exception:
        kg.close()
        return {"error": True}


def main():
    all_methods = [
        "greedy_cooccur", "greedy_coverage",
        "ig_expected", "ig_max", "ig_binary_split",
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
    print(f"=== Hit Rate 곡선: {args.method} ({args.n_samples}건) ===")

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

    # IL별 평균 hit rate 곡선 계산
    max_observed_il = max(r["il"] for r in results)
    curve_data = {}

    for il in range(1, min(max_observed_il + 1, MAX_IL + 1)):
        # il 시점까지 도달한 케이스만
        confirmed_at_il = []
        denied_at_il = []
        hit_rate_at_il = []
        n_active = 0

        for r in results:
            if il < len(r["cumulative_confirmed"]):
                c = r["cumulative_confirmed"][il]
                d = r["cumulative_denied"][il]
                confirmed_at_il.append(c)
                denied_at_il.append(d)
                hit_rate_at_il.append(c / (c + d) if (c + d) > 0 else 0)
                n_active += 1

        if n_active < 10:  # 너무 적은 케이스면 중단
            break

        curve_data[il] = {
            "n_active": n_active,
            "avg_confirmed": float(np.mean(confirmed_at_il)),
            "avg_denied": float(np.mean(denied_at_il)),
            "avg_hit_rate": float(np.mean(hit_rate_at_il)),
            "std_hit_rate": float(np.std(hit_rate_at_il)),
            # 해당 step의 marginal hit rate (이번 질문에서 hit한 비율)
            "marginal_hit_rate": float(np.mean([
                r["step_hit"][il - 1] for r in results
                if il - 1 < len(r["step_hit"])
            ])) if any(il - 1 < len(r["step_hit"]) for r in results) else 0,
        }

    # 요약 통계
    avg_il = np.mean([r["il"] for r in results])
    avg_confirmed = np.mean([r["total_confirmed"] for r in results])
    avg_denied = np.mean([r["total_denied"] for r in results])
    avg_final_hit_rate = np.mean([
        r["total_confirmed"] / (r["total_confirmed"] + r["total_denied"])
        if (r["total_confirmed"] + r["total_denied"]) > 0 else 0
        for r in results
    ])

    output = {
        "method": args.method,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "count": count,
        "errors": errors,
        "avg_il": float(avg_il),
        "avg_confirmed": float(avg_confirmed),
        "avg_denied": float(avg_denied),
        "avg_final_hit_rate": float(avg_final_hit_rate),
        "curve": curve_data,
        "elapsed": elapsed,
    }

    # 주요 지점 출력
    print(f"\n{'IL':>4} | {'Active':>6} | {'Hit Rate':>8} | {'Marginal':>8} | {'Confirmed':>9}")
    print("-" * 50)
    for il in [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]:
        if il in curve_data:
            d = curve_data[il]
            print(f"{il:>4} | {d['n_active']:>6} | {d['avg_hit_rate']:>7.1%} | {d['marginal_hit_rate']:>7.1%} | {d['avg_confirmed']:>8.1f}")

    print(f"\n최종: IL={avg_il:.1f}, Confirmed={avg_confirmed:.1f}, Hit Rate={avg_final_hit_rate:.1%}")
    print(f"시간: {elapsed:.1f}s")

    output_path = Path("results") / f"hitcurve_{args.method}_{args.n_samples}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
