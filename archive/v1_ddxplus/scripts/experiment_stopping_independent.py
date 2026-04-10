#!/usr/bin/env python3
"""스코어링-독립적 종료 조건 실험.

증상 탐색 3개 × 종료 조건 후보 = 전체 조합 테스트.
1,000건 샘플 (seed=42).

종료 조건 후보 (A: 스코어링-독립적):
  - consecutive_miss_N: 연속 N회 미적중이면 종료 (N=3,5,7)
  - marginal_hr_k_t: 최근 k개 질문의 적중률이 t 미만이면 종료
  - cumulative_confirmed_N: 확인된 증상이 N개 이상이면 종료 (N=3,5,7,10)

종료 조건 후보 (B: 스코어링-의존적, 비교용):
  - top3_stable_5: 기존 Rank Stability

사용법: python experiment_stopping_independent.py \
          --exploration greedy_cooccur \
          --stopping consecutive_miss_5 \
          --workers 2 --ports "7687,7688"
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
    """증상 탐색 전략."""
    if exploration == "greedy_cooccur":
        return candidates[0].cui

    elif exploration == "ig_expected":
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

    elif exploration == "minimax_score":
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

    return candidates[0].cui


def check_stopping(stopping, il, step_hits, confirmed_count, kg, rank_history):
    """종료 조건 확인. True면 종료."""
    parts = stopping.split("_")

    # consecutive_miss_N
    if stopping.startswith("consecutive_miss_"):
        n = int(parts[-1])
        if len(step_hits) >= n:
            if all(h == 0 for h in step_hits[-n:]):
                return True
        return False

    # marginal_hr_K_T (최근 K개 질문의 적중률 < T)
    # 예: marginal_hr_5_10 → 최근 5개 중 적중률 < 10% (0.1)
    if stopping.startswith("marginal_hr_"):
        k = int(parts[2])
        t = int(parts[3]) / 100.0  # 10 → 0.1
        if len(step_hits) >= k:
            recent_hr = sum(step_hits[-k:]) / k
            if recent_hr < t:
                return True
        return False

    # cumulative_confirmed_N
    if stopping.startswith("cumulative_confirmed_"):
        n = int(parts[-1])
        if confirmed_count >= n:
            return True
        return False

    # top3_stable_5 (기존 비교용)
    if stopping == "top3_stable_5":
        if len(rank_history) >= 5:
            recent = list(rank_history)[-5:]
            if all(r == recent[0] for r in recent):
                return True
        return False

    # top3_stable_7
    if stopping == "top3_stable_7":
        if len(rank_history) >= 7:
            recent = list(rank_history)[-7:]
            if all(r == recent[0] for r in recent):
                return True
        return False

    return False


def run_single_patient(args: tuple) -> dict:
    patient_data, loader_data, exploration, stopping, neo4j_port = args

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

            # rank history 업데이트 (top3_stable용)
            diagnosis_candidates = kg.get_diagnosis_candidates(top_k=10)
            if diagnosis_candidates:
                current_ranks = tuple(c.cui for c in diagnosis_candidates[:3])
                rank_history.append(current_ranks)

            # 종료 조건 확인
            if check_stopping(stopping, il, step_hits, confirmed_count, kg, rank_history):
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
            "confirmed": confirmed_count,
            "denied": il - (confirmed_count - 1),
            "hit_rate": confirmed_count / (confirmed_count + il - (confirmed_count - 1)) if il > 0 else 0,
        }

    except Exception:
        kg.close()
        return {"error": True}


def main():
    explorations = ["greedy_cooccur", "ig_expected", "minimax_score"]
    stoppings = [
        "consecutive_miss_3", "consecutive_miss_5", "consecutive_miss_7",
        "marginal_hr_5_10", "marginal_hr_5_20", "marginal_hr_10_10", "marginal_hr_10_20",
        "cumulative_confirmed_3", "cumulative_confirmed_5", "cumulative_confirmed_7", "cumulative_confirmed_10",
        "top3_stable_5", "top3_stable_7",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration", required=True, choices=explorations)
    parser.add_argument("--stopping", required=True, choices=stoppings)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ports", type=str, default="7687,7688")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    method_name = f"{args.exploration}+{args.stopping}"
    ports = [int(p.strip()) for p in args.ports.split(",")]
    print(f"=== {method_name} ({args.n_samples}건) ===")

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

    tasks = [(pd, loader_data, args.exploration, args.stopping, ports[i % len(ports)])
             for i, pd in enumerate(patients_data)]

    start_time = time.time()
    results = []
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_patient, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc=method_name[:30]) as pbar:
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
        "method": method_name,
        "n_samples": args.n_samples, "seed": args.seed,
        "count": count, "errors": errors,
        "gtpa_1": sum(r["correct_at_1"] for r in results) / count if count else 0,
        "gtpa_10": sum(r["correct_at_10"] for r in results) / count if count else 0,
        "avg_il": float(np.mean(ils)),
        "il_std": float(np.std(ils)),
        "il_median": float(np.median(ils)),
        "avg_confirmed": float(np.mean([r["confirmed"] for r in results])),
        "avg_hit_rate": float(np.mean([r["hit_rate"] for r in results])),
        "elapsed": elapsed,
    }

    print(f"\nGTPA@1:    {output['gtpa_1']:.2%}")
    print(f"GTPA@10:   {output['gtpa_10']:.2%}")
    print(f"Avg IL:    {output['avg_il']:.1f} (median {output['il_median']:.0f})")
    print(f"Confirmed: {output['avg_confirmed']:.1f}")
    print(f"Hit Rate:  {output['avg_hit_rate']:.1%}")

    output_path = Path("results") / f"stopind_{args.exploration}_{args.stopping}_{args.n_samples}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
