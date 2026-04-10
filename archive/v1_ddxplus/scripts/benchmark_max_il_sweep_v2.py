#!/usr/bin/env python3
"""max_il sweep v2 - experiment_group.py의 Top3_stable_5 로직을 정확히 재현.

단일 max_il 값으로 실행.
사용법: python benchmark_max_il_sweep_v2.py --max-il 30 --workers 2
        python benchmark_max_il_sweep_v2.py --unlimited --workers 2
"""

import argparse
import json
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_single_patient(args: tuple) -> dict:
    """단일 환자 진단 - experiment_group.py와 동일한 로직."""
    patient_data, loader_data, max_il, neo4j_port = args

    from src.data_loader import DDXPlusLoader, Patient
    from src.umls_kg import UMLSKG

    loader = DDXPlusLoader()
    loader._symptom_mapping = loader_data["symptom_mapping"]
    loader._disease_mapping = loader_data["disease_mapping"]
    loader._fr_to_eng = loader_data["fr_to_eng"]
    loader._conditions = loader_data["conditions"]

    try:
        uri = f"bolt://localhost:{neo4j_port}"
        kg = UMLSKG(uri=uri)
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

        hard_limit = max_il if max_il else 500
        il = 0
        max_il_reached = False

        # Top3_stable_5 파라미터
        stability_k = 3
        stability_n = 5
        rank_history = deque(maxlen=10)

        for _ in range(hard_limit):
            # experiment_group.py 동일: asked_cuis 명시적 전달 안 함
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

            # experiment_group.py 동일: top_k=10으로 진단 후보 조회
            diagnosis_candidates = kg.get_diagnosis_candidates(top_k=10)
            if not diagnosis_candidates:
                break

            # rank_stability: Top3_stable_5
            current_ranks = tuple(c.cui for c in diagnosis_candidates[:stability_k])
            rank_history.append(current_ranks)
            if len(rank_history) >= stability_n:
                recent = list(rank_history)[-stability_n:]
                if all(r == recent[0] for r in recent):
                    break

            # max_il 체크
            if max_il and il >= max_il:
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
    parser.add_argument("--max-il", type=int, default=None)
    parser.add_argument("--unlimited", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ports", type=str, default="7687,7688",
                        help="쉼표로 구분된 Neo4j 포트 목록")
    args = parser.parse_args()

    max_il_value = None if args.unlimited else args.max_il
    max_label = "unlimited" if max_il_value is None else max_il_value
    ports = [int(p.strip()) for p in args.ports.split(",")]

    print(f"=== Top3_stable_5 + max_il={max_label} ===")
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
            "age": p.age,
            "sex": p.sex,
            "initial_evidence": p.initial_evidence,
            "evidences": p.evidences,
            "pathology": p.pathology,
            "differential_diagnosis": p.differential_diagnosis,
        }
        for p in patients
    ]

    tasks = [
        (pd, loader_data, max_il_value, ports[i % len(ports)])
        for i, pd in enumerate(patients_data)
    ]

    start_time = time.time()
    results = []
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_patient, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc=f"max_il={max_label}") as pbar:
            for future in as_completed(futures):
                r = future.result()
                if r and not r.get("error"):
                    results.append(r)
                else:
                    errors += 1
                pbar.update(1)

    elapsed = time.time() - start_time

    count = len(results)
    correct_1 = sum(r["correct_at_1"] for r in results)
    correct_10 = sum(r["correct_at_10"] for r in results)
    ils = [r["il"] for r in results]
    max_il_reached_count = sum(1 for r in results if r.get("max_il_reached"))

    gtpa_1 = correct_1 / count if count else 0
    gtpa_10 = correct_10 / count if count else 0
    avg_il = np.mean(ils) if ils else 0

    print(f"\n완료: {elapsed:.1f}초 (errors: {errors})")
    print(f"GTPA@1:  {gtpa_1:.2%}")
    print(f"GTPA@10: {gtpa_10:.2%}")
    print(f"Avg IL:  {avg_il:.2f} (SD {np.std(ils):.2f})")
    print(f"max_il 도달: {max_il_reached_count}건 ({max_il_reached_count/count:.2%})")

    output = {
        "method": "Top3_stable_5",
        "max_il": max_label,
        "count": count,
        "errors": errors,
        "gtpa_1": gtpa_1,
        "gtpa_10": gtpa_10,
        "avg_il": float(avg_il),
        "il_std": float(np.std(ils)),
        "il_median": float(np.median(ils)),
        "max_il_reached_count": max_il_reached_count,
        "max_il_reached_pct": max_il_reached_count / count if count else 0,
        "elapsed": elapsed,
    }

    output_path = Path("results") / f"sweep_v2_max_il_{max_label}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
