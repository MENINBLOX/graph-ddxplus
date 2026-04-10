#!/usr/bin/env python3
"""그룹별 실험 실행.

4개 그룹으로 분할하여 동시 실행.
각 그룹은 전용 Neo4j 포트 2개를 사용.
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TestConfig:
    method: str
    param1: float
    param2: float = 0.0
    description: str = ""


# 그룹별 Neo4j 포트 할당
GROUP_PORTS = {
    1: [7687, 7688],  # min_il (26개)
    2: [7689, 7690],  # confidence, entropy, info_gain (31개)
    3: [7691, 7692],  # rank_stability, evidence_coverage (22개)
    4: [7693, 7694],  # disease_narrowing, confidence_stability, next_question_quality (48개)
}


def get_group_configs(group: int) -> list[TestConfig]:
    """그룹별 테스트 설정 반환."""
    configs = []

    if group == 1:
        # min_il (26개)
        for v in range(0, 26):
            configs.append(TestConfig("min_il", float(v), description=f"min_il={v}"))

    elif group == 2:
        # confidence_only (11개)
        for v in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
            configs.append(TestConfig("confidence_only", v, description=f"conf>={v}"))

        # entropy (10개)
        for v in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            configs.append(TestConfig("entropy", v, description=f"entropy<{v}"))

        # info_gain (10개)
        for v in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
            configs.append(TestConfig("info_gain", v, description=f"IG<{v}"))

    elif group == 3:
        # rank_stability (12개)
        for k in [1, 3, 5]:
            for n in [2, 3, 4, 5]:
                configs.append(TestConfig("rank_stability", float(k), float(n), f"Top{k}_stable_{n}"))

        # evidence_coverage (10개)
        for v in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
            configs.append(TestConfig("evidence_coverage", v, description=f"EvidCov>={v}"))

    elif group == 4:
        # disease_narrowing (18개)
        for num in [1, 2, 3, 5, 7, 10]:
            for ms in [0.01, 0.05, 0.1]:
                configs.append(TestConfig("disease_narrowing", float(num), ms, f"Diseases<={num},s>={ms}"))

        # confidence_stability (20개)
        for conf in [0.2, 0.3, 0.4, 0.5, 0.6]:
            for stab in [2, 3, 4, 5]:
                configs.append(TestConfig("confidence_stability", conf, float(stab), f"Conf>={conf},Stab>={stab}"))

        # next_question_quality (10개)
        for v in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]:
            configs.append(TestConfig("next_question_quality", v, description=f"NormIG<{v}"))

    return configs


def run_single_patient(args: tuple) -> dict:
    """단일 환자에 대한 진단 실행."""
    patient_data, config, loader_data, neo4j_port = args

    from collections import deque

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

        max_il = 50
        il = 0

        # 상태 추적 변수들
        prev_top1 = None
        stability_count = 0
        prev_entropy = None
        consecutive_low_ig = 0
        rank_history = deque(maxlen=10)

        def calculate_entropy(scores):
            if not scores or sum(scores) == 0:
                return 0.0
            total = sum(scores)
            probs = [s / total for s in scores if s > 0]
            return -sum(p * math.log2(p) for p in probs if p > 0)

        for _ in range(max_il):
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

            # Stopping logic
            if config.method == "min_il":
                min_il_val = int(config.param1)
                if il >= min_il_val:
                    top1_score = diagnosis_candidates[0].score
                    top2_score = diagnosis_candidates[1].score if len(diagnosis_candidates) > 1 else 0
                    if top1_score >= 0.30 or (top1_score - top2_score) >= 0.005:
                        should_stop = True

            elif config.method == "confidence_only":
                if diagnosis_candidates[0].score >= config.param1:
                    should_stop = True

            elif config.method == "entropy":
                if current_entropy < config.param1:
                    should_stop = True

            elif config.method == "info_gain":
                if prev_entropy is not None:
                    ig = prev_entropy - current_entropy
                    if ig < config.param1:
                        consecutive_low_ig += 1
                    else:
                        consecutive_low_ig = 0
                    if consecutive_low_ig >= 2:
                        should_stop = True
                prev_entropy = current_entropy

            elif config.method == "rank_stability":
                k, n = int(config.param1), int(config.param2)
                current_ranks = tuple(c.cui for c in diagnosis_candidates[:k])
                rank_history.append(current_ranks)
                if len(rank_history) >= n:
                    recent = list(rank_history)[-n:]
                    if all(r == recent[0] for r in recent):
                        should_stop = True

            elif config.method == "evidence_coverage":
                total_ev = len(kg.state.confirmed_cuis) + len(kg.state.denied_cuis)
                if total_ev / 10.0 >= config.param1:
                    should_stop = True

            elif config.method == "disease_narrowing":
                viable = [c for c in diagnosis_candidates if c.score >= config.param2]
                if len(viable) <= int(config.param1):
                    should_stop = True

            elif config.method == "confidence_stability":
                top1 = diagnosis_candidates[0]
                if top1.score >= config.param1:
                    if prev_top1 == top1.cui:
                        stability_count += 1
                    else:
                        stability_count = 1
                    prev_top1 = top1.cui
                    if stability_count >= int(config.param2):
                        should_stop = True
                else:
                    stability_count = 0
                    prev_top1 = None

            elif config.method == "next_question_quality":
                if prev_entropy is not None and prev_entropy > 0:
                    norm_ig = (prev_entropy - current_entropy) / prev_entropy
                    if norm_ig < config.param1 and il >= 3:
                        should_stop = True
                prev_entropy = current_entropy

            if should_stop or il >= max_il:
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
        }

    except Exception:
        kg.close()
        return {"error": True}


def run_config_test(config: TestConfig, patients_data: list, loader_data: dict, ports: list[int]) -> dict:
    """단일 설정에 대해 모든 환자 테스트 (환자 단위 병렬 처리)."""
    correct_at_1 = 0
    correct_at_10 = 0
    total_il = 0
    count = 0
    errors = 0

    # 환자별로 포트 순환 할당
    tasks = []
    for i, patient_data in enumerate(patients_data):
        port = ports[i % len(ports)]
        tasks.append((patient_data, config, loader_data, port))

    # 환자 단위 병렬 처리
    with ProcessPoolExecutor(max_workers=len(ports) * 4) as executor:
        futures = list(executor.map(run_single_patient, tasks, chunksize=100))

    for result in futures:
        if result.get("error"):
            errors += 1
        else:
            correct_at_1 += result["correct_at_1"]
            correct_at_10 += result["correct_at_10"]
            total_il += result["il"]
            count += 1

    return {
        "method": config.method,
        "param1": config.param1,
        "param2": config.param2,
        "description": config.description,
        "count": count,
        "errors": errors,
        "correct_at_1": correct_at_1,
        "correct_at_10": correct_at_10,
        "gtpa_1": correct_at_1 / count if count > 0 else 0,
        "gtpa_10": correct_at_10 / count if count > 0 else 0,
        "avg_il": total_il / count if count > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="그룹별 실험 실행")
    parser.add_argument("--group", type=int, required=True, choices=[1, 2, 3, 4], help="실행할 그룹 번호")
    args = parser.parse_args()

    from src.data_loader import DDXPlusLoader

    group = args.group
    ports = GROUP_PORTS[group]
    configs = get_group_configs(group)

    print("=" * 70)
    print(f"Group {group} Experiments")
    print(f"Ports: {ports}")
    print(f"Configurations: {len(configs)}")
    print("=" * 70)

    # 데이터 로드
    loader = DDXPlusLoader()
    _ = loader.symptom_mapping
    _ = loader.disease_mapping
    _ = loader.fr_to_eng

    test_patients = loader.load_patients(split="test")
    print(f"Total test patients: {len(test_patients):,}")

    loader_data = {
        "symptom_mapping": loader._symptom_mapping,
        "disease_mapping": loader._disease_mapping,
        "fr_to_eng": loader._fr_to_eng,
        "conditions": loader._conditions,
    }

    patients_data = []
    for p in test_patients:
        patients_data.append({
            "age": p.age,
            "sex": p.sex,
            "initial_evidence": p.initial_evidence,
            "evidences": p.evidences,
            "pathology": p.pathology,
            "differential_diagnosis": p.differential_diagnosis,
        })

    # 각 설정 순차 실행 (환자 단위 병렬)
    all_results = []
    start_time = time.time()

    for i, config in enumerate(configs):
        config_start = time.time()
        print(f"\n[{i+1}/{len(configs)}] Testing: {config.description}")

        result = run_config_test(config, patients_data, loader_data, ports)
        all_results.append(result)

        config_elapsed = time.time() - config_start
        print(f"  GTPA@1: {result['gtpa_1']:.2%}, GTPA@10: {result['gtpa_10']:.2%}, Avg IL: {result['avg_il']:.1f}")
        print(f"  Patients: {result['count']:,}, Errors: {result['errors']}, Time: {config_elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Group {group} completed in {elapsed/60:.1f} min")

    # 결과 저장
    output_path = Path(f"results/group{group}_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")

    # 요약
    print(f"\n{'=' * 70}")
    print("TOP 5 by GTPA@1")
    print("=" * 70)
    sorted_results = sorted(all_results, key=lambda x: x["gtpa_1"], reverse=True)[:5]
    print(f"{'Config':>40} {'GTPA@1':>10} {'GTPA@10':>10} {'Avg IL':>10}")
    print("-" * 75)
    for r in sorted_results:
        print(f"{r['description']:>40} {r['gtpa_1']:>10.2%} {r['gtpa_10']:>10.2%} {r['avg_il']:>10.1f}")


if __name__ == "__main__":
    main()
