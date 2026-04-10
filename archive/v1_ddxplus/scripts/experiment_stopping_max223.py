#!/usr/bin/env python3
"""종료 조건 비교 실험 (max_il=223, DDXPlus 전체 evidence 수).

experiment_group.py의 로직을 정확히 재현하되 max_il=223으로 설정.

사용법: python experiment_stopping_max223.py --method rank_stability --param1 3 --param2 5 --desc Top3_stable_5
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

MAX_IL = 223  # DDXPlus 전체 evidence 수


def run_single_patient(args: tuple) -> dict:
    """단일 환자 진단 - experiment_group.py와 동일한 로직."""
    patient_data, method, param1, param2, loader_data, neo4j_port = args

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
        max_il_reached = False

        # 상태 추적 변수
        prev_top1 = None
        stability_count = 0
        prev_entropy = None
        consecutive_low_ig = 0
        rank_history = deque(maxlen=20)

        def calculate_entropy(scores):
            if not scores or sum(scores) == 0:
                return 0.0
            total = sum(scores)
            probs = [s / total for s in scores if s > 0]
            return -sum(p * math.log2(p) for p in probs if p > 0)

        for _ in range(MAX_IL):
            candidates = kg.get_candidate_symptoms(
                initial_cui=initial_cui,
                limit=10,
                confirmed_cuis=kg.state.confirmed_cuis,
                denied_cuis=kg.state.denied_cuis,
            )
            if not candidates:
                break

            next_cui = candidates[0].cui
            if next_cui in patient_positive_cuis:
                kg.state.add_confirmed(next_cui)
            else:
                kg.state.add_denied(next_cui)

            il += 1

            diagnosis_candidates = kg.get_diagnosis_candidates(top_k=10)
            if not diagnosis_candidates:
                break

            should_stop = False
            scores = [c.score for c in diagnosis_candidates if c.score > 0]
            current_entropy = calculate_entropy(scores) if scores else 0

            # === Stopping Logic ===
            if method == "rank_stability":
                k, n = int(param1), int(param2)
                current_ranks = tuple(c.cui for c in diagnosis_candidates[:k])
                rank_history.append(current_ranks)
                if len(rank_history) >= n:
                    recent = list(rank_history)[-n:]
                    if all(r == recent[0] for r in recent):
                        should_stop = True

            elif method == "confidence_only":
                if diagnosis_candidates[0].score >= param1:
                    should_stop = True

            elif method == "confidence_gap":
                top1_score = diagnosis_candidates[0].score
                top2_score = diagnosis_candidates[1].score if len(diagnosis_candidates) > 1 else 0
                if top1_score >= param1 and (top1_score - top2_score) >= param2:
                    should_stop = True

            elif method == "confidence_stability":
                top1 = diagnosis_candidates[0]
                if top1.score >= param1:
                    if prev_top1 == top1.cui:
                        stability_count += 1
                    else:
                        stability_count = 1
                    prev_top1 = top1.cui
                    if stability_count >= int(param2):
                        should_stop = True
                else:
                    stability_count = 0
                    prev_top1 = None

            elif method == "entropy":
                if current_entropy < param1:
                    should_stop = True

            elif method == "info_gain":
                if prev_entropy is not None:
                    ig = prev_entropy - current_entropy
                    if ig < param1:
                        consecutive_low_ig += 1
                    else:
                        consecutive_low_ig = 0
                    if consecutive_low_ig >= int(param2):
                        should_stop = True
                prev_entropy = current_entropy

            if should_stop:
                break

            if il >= MAX_IL:
                max_il_reached = True
                break

        # 최종 진단
        final_candidates = kg.get_diagnosis_candidates(top_k=10)
        correct_at_1 = 0
        correct_at_10 = 0

        if final_candidates:
            if final_candidates[0].cui == gt_cui:
                correct_at_1 = 1
            for c in final_candidates[:10]:
                if c.cui == gt_cui:
                    correct_at_10 = 1
                    break

        kg.close()
        return {
            "error": False,
            "correct_at_1": correct_at_1,
            "correct_at_10": correct_at_10,
            "il": il,
            "max_il_reached": max_il_reached,
        }

    except Exception:
        kg.close()
        return {"error": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--param1", type=float, required=True)
    parser.add_argument("--param2", type=float, default=0.0)
    parser.add_argument("--desc", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ports", type=str, default="7687,7688")
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",")]
    print(f"=== {args.desc} (max_il={MAX_IL}) ===")
    print(f"Method: {args.method}, param1={args.param1}, param2={args.param2}")
    print(f"Workers: {args.workers}, Ports: {ports}")

    from src.data_loader import DDXPlusLoader

    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test")
    print(f"환자 수: {len(patients):,}")

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
        (pd, args.method, args.param1, args.param2, loader_data, ports[i % len(ports)])
        for i, pd in enumerate(patients_data)
    ]

    start_time = time.time()
    results = []
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_patient, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc=args.desc) as pbar:
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
        "param1": args.param1,
        "param2": args.param2,
        "description": args.desc,
        "max_il": MAX_IL,
        "count": count,
        "errors": errors,
        "gtpa_1": sum(r["correct_at_1"] for r in results) / count if count else 0,
        "gtpa_10": sum(r["correct_at_10"] for r in results) / count if count else 0,
        "avg_il": float(np.mean(ils)),
        "il_std": float(np.std(ils)),
        "il_median": float(np.median(ils)),
        "il_min": int(min(ils)),
        "il_max": int(max(ils)),
        "il_p95": float(np.percentile(ils, 95)),
        "il_p99": float(np.percentile(ils, 99)),
        "max_il_reached_pct": sum(1 for r in results if r.get("max_il_reached")) / count if count else 0,
        "elapsed": elapsed,
    }

    print(f"\nGTPA@1:  {output['gtpa_1']:.2%}")
    print(f"GTPA@10: {output['gtpa_10']:.2%}")
    print(f"Avg IL:  {output['avg_il']:.2f} (SD {output['il_std']:.2f})")
    print(f"IL range: {output['il_min']}-{output['il_max']}, p95={output['il_p95']:.0f}, p99={output['il_p99']:.0f}")

    safe_desc = args.desc.replace(">=", "ge").replace("<=", "le").replace("<", "lt").replace(">", "gt").replace(",", "_").replace(" ", "_")
    output_path = Path("results") / f"stop223_{safe_desc}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
