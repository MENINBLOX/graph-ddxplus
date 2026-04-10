#!/usr/bin/env python3
"""모든 Stopping Criteria 통합 실험 - 병렬 처리.

8개 Neo4j 컨테이너를 활용한 병렬 처리.

테스트 방법:
1. min_il 변화 (0~25)
2. Confidence Only (0.1~0.99)
3. Entropy 기반 (0.5~5.0)
4. Information Gain 기반 (0.001~1.0)
5. Rank Stability 기반 (K=1,3,5, N=2,3,4,5)
6. Evidence Coverage (0.3~1.5)
7. Disease Narrowing (1~10개, min_score 0.01~0.1)
8. Confidence + Stability (conf 0.2~0.6, stability 2~5)
9. Next Question Quality (0.01~0.3)
"""

import json
import math
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

NEO4J_PORTS = [7687, 7688, 7689, 7690, 7691, 7692, 7693, 7694]


@dataclass
class TestConfig:
    method: str
    param1: float
    param2: float = 0.0
    param3: float = 0.0
    description: str = ""


def calculate_entropy(scores: list[float]) -> float:
    """점수 리스트의 entropy 계산."""
    if not scores or sum(scores) == 0:
        return 0.0
    total = sum(scores)
    probs = [s / total for s in scores if s > 0]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def run_single_patient(args: tuple) -> dict | None:
    """단일 환자 진단 - 병렬 처리용."""
    patient_data, loader_data, neo4j_port, config = args

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
    except Exception as e:
        return {"error": str(e)}

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
            return None

        kg.reset_state()
        kg.state.add_confirmed(initial_cui)

        max_il = 50
        il = 0

        # 상태 추적 변수
        prev_top1 = None
        stability_count = 0
        prev_entropy = None
        consecutive_low_ig = 0
        rank_history = deque(maxlen=10)

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

            # 진단 후보 가져오기
            diagnosis_candidates = kg.get_diagnosis_candidates(top_k=10)
            if not diagnosis_candidates:
                break

            should_stop = False
            scores = [c.score for c in diagnosis_candidates if c.score > 0]
            current_entropy = calculate_entropy(scores) if scores else 0

            # ========================================
            # Method 1: min_il (기존 방식)
            # ========================================
            if config.method == "min_il":
                min_il_val = int(config.param1)
                if il >= min_il_val:
                    # 기존 confidence 조건
                    top1_score = diagnosis_candidates[0].score
                    top2_score = diagnosis_candidates[1].score if len(diagnosis_candidates) > 1 else 0
                    if top1_score >= 0.30 or (top1_score - top2_score) >= 0.005:
                        should_stop = True

            # ========================================
            # Method 2: Confidence Only
            # ========================================
            elif config.method == "confidence_only":
                threshold = config.param1
                top1_score = diagnosis_candidates[0].score
                if top1_score >= threshold:
                    should_stop = True

            # ========================================
            # Method 3: Entropy
            # ========================================
            elif config.method == "entropy":
                threshold = config.param1
                if current_entropy < threshold:
                    should_stop = True

            # ========================================
            # Method 4: Information Gain
            # ========================================
            elif config.method == "info_gain":
                threshold = config.param1
                if prev_entropy is not None:
                    info_gain = prev_entropy - current_entropy
                    if info_gain < threshold:
                        consecutive_low_ig += 1
                    else:
                        consecutive_low_ig = 0
                    if consecutive_low_ig >= 2:
                        should_stop = True
                prev_entropy = current_entropy

            # ========================================
            # Method 5: Rank Stability
            # ========================================
            elif config.method == "rank_stability":
                k = int(config.param1)
                n = int(config.param2)
                current_ranks = tuple(c.cui for c in diagnosis_candidates[:k])
                rank_history.append(current_ranks)
                if len(rank_history) >= n:
                    recent = list(rank_history)[-n:]
                    if all(r == recent[0] for r in recent):
                        should_stop = True

            # ========================================
            # Method 6: Evidence Coverage
            # ========================================
            elif config.method == "evidence_coverage":
                threshold = config.param1
                total_evidences = len(kg.state.confirmed_cuis) + len(kg.state.denied_cuis)
                coverage = total_evidences / 10.0
                if coverage >= threshold:
                    should_stop = True

            # ========================================
            # Method 7: Disease Narrowing
            # ========================================
            elif config.method == "disease_narrowing":
                num_threshold = int(config.param1)
                min_score = config.param2
                viable = [c for c in diagnosis_candidates if c.score >= min_score]
                if len(viable) <= num_threshold:
                    should_stop = True

            # ========================================
            # Method 8: Confidence + Stability
            # ========================================
            elif config.method == "confidence_stability":
                conf_threshold = config.param1
                stability_required = int(config.param2)
                top1 = diagnosis_candidates[0]
                if top1.score >= conf_threshold:
                    if prev_top1 == top1.cui:
                        stability_count += 1
                    else:
                        stability_count = 1
                    prev_top1 = top1.cui
                    if stability_count >= stability_required:
                        should_stop = True
                else:
                    stability_count = 0
                    prev_top1 = None

            # ========================================
            # Method 9: Next Question Quality
            # ========================================
            elif config.method == "next_question_quality":
                threshold = config.param1
                if prev_entropy is not None and prev_entropy > 0:
                    normalized_ig = (prev_entropy - current_entropy) / prev_entropy
                    if normalized_ig < threshold and il >= 3:
                        should_stop = True
                prev_entropy = current_entropy

            if should_stop:
                break

            if il >= max_il:
                break

        # 최종 진단
        final_candidates = kg.get_diagnosis_candidates(top_k=10)
        top1_correct = False
        top10_correct = False

        if final_candidates:
            if final_candidates[0].cui == gt_cui:
                top1_correct = True
            for c in final_candidates[:10]:
                if c.cui == gt_cui:
                    top10_correct = True
                    break

        kg.close()

        return {
            "il": il,
            "top1_correct": top1_correct,
            "top10_correct": top10_correct,
        }

    except Exception as e:
        kg.close()
        return {"error": str(e)}


def run_test(patients_data, loader_data, config: TestConfig) -> dict:
    """병렬로 테스트 실행."""
    num_workers = len(NEO4J_PORTS)

    tasks = []
    for idx, pd in enumerate(patients_data):
        port = NEO4J_PORTS[idx % num_workers]
        tasks.append((pd, loader_data, port, config))

    correct_at_1 = 0
    correct_at_10 = 0
    total_il = 0
    count = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_single_patient, t): t for t in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc=config.description[:30]):
            result = future.result()
            if result is None:
                continue
            if "error" in result:
                errors += 1
                continue

            count += 1
            total_il += result["il"]
            if result["top1_correct"]:
                correct_at_1 += 1
            if result["top10_correct"]:
                correct_at_10 += 1

    return {
        "method": config.method,
        "param1": config.param1,
        "param2": config.param2,
        "param3": config.param3,
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
    from src.data_loader import DDXPlusLoader

    print("=" * 70)
    print("All Stopping Criteria Experiment (Parallel)")
    print("=" * 70)

    loader = DDXPlusLoader()
    _ = loader.symptom_mapping
    _ = loader.disease_mapping
    _ = loader.fr_to_eng

    test_patients = loader.load_patients(split="test")
    print(f"Total test patients: {len(test_patients):,}")
    print(f"Using {len(NEO4J_PORTS)} Neo4j instances for parallel processing")

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

    all_results = []
    configs = []

    # ========================================
    # 1. min_il 변화 (0~25, 26개)
    # ========================================
    for min_il in range(0, 26):
        configs.append(TestConfig(
            method="min_il",
            param1=float(min_il),
            description=f"min_il={min_il}"
        ))

    # ========================================
    # 2. Confidence Only (11개)
    # ========================================
    for conf in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
        configs.append(TestConfig(
            method="confidence_only",
            param1=conf,
            description=f"conf>={conf}"
        ))

    # ========================================
    # 3. Entropy (10개)
    # ========================================
    for threshold in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        configs.append(TestConfig(
            method="entropy",
            param1=threshold,
            description=f"entropy<{threshold}"
        ))

    # ========================================
    # 4. Information Gain (10개)
    # ========================================
    for threshold in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        configs.append(TestConfig(
            method="info_gain",
            param1=threshold,
            description=f"IG<{threshold}"
        ))

    # ========================================
    # 5. Rank Stability (12개)
    # ========================================
    for k in [1, 3, 5]:
        for n in [2, 3, 4, 5]:
            configs.append(TestConfig(
                method="rank_stability",
                param1=float(k),
                param2=float(n),
                description=f"Top{k}_stable_{n}"
            ))

    # ========================================
    # 6. Evidence Coverage (10개)
    # ========================================
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
        configs.append(TestConfig(
            method="evidence_coverage",
            param1=threshold,
            description=f"EvidCov>={threshold}"
        ))

    # ========================================
    # 7. Disease Narrowing (18개)
    # ========================================
    for num in [1, 2, 3, 5, 7, 10]:
        for min_score in [0.01, 0.05, 0.1]:
            configs.append(TestConfig(
                method="disease_narrowing",
                param1=float(num),
                param2=min_score,
                description=f"Diseases<={num},s>={min_score}"
            ))

    # ========================================
    # 8. Confidence + Stability (20개)
    # ========================================
    for conf in [0.2, 0.3, 0.4, 0.5, 0.6]:
        for stability in [2, 3, 4, 5]:
            configs.append(TestConfig(
                method="confidence_stability",
                param1=conf,
                param2=float(stability),
                description=f"Conf>={conf},Stab>={stability}"
            ))

    # ========================================
    # 9. Next Question Quality (10개)
    # ========================================
    for threshold in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]:
        configs.append(TestConfig(
            method="next_question_quality",
            param1=threshold,
            description=f"NormIG<{threshold}"
        ))

    print(f"\nTotal configurations to test: {len(configs)}")
    print(f"Estimated time: {len(configs) * 2.5 / 60:.1f} hours")

    # ========================================
    # Run Tests
    # ========================================
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {config.description}")
        start = time.time()
        result = run_test(patients_data, loader_data, config)
        elapsed = time.time() - start

        all_results.append(result)
        print(f"  GTPA@1: {result['gtpa_1']:.2%}, GTPA@10: {result['gtpa_10']:.2%}, "
              f"Avg IL: {result['avg_il']:.1f}, Time: {elapsed/60:.1f}min")

        # 중간 저장 (10개마다)
        if (i + 1) % 10 == 0:
            output_path = Path("results/all_stopping_parallel.json")
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  [Checkpoint saved: {i+1}/{len(configs)}]")

    # ========================================
    # Final Save
    # ========================================
    output_path = Path("results/all_stopping_parallel.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to {output_path}")
    print(f"{'='*70}")

    # ========================================
    # Summary Tables
    # ========================================
    methods = ["min_il", "confidence_only", "entropy", "info_gain", "rank_stability",
               "evidence_coverage", "disease_narrowing", "confidence_stability", "next_question_quality"]

    for method in methods:
        method_results = [r for r in all_results if r["method"] == method]
        if not method_results:
            continue

        print(f"\n### {method} ###")
        print(f"{'Config':>35} {'GTPA@1':>10} {'GTPA@10':>10} {'Avg IL':>10}")
        print("-" * 70)

        # GTPA@1 기준 상위 5개
        sorted_results = sorted(method_results, key=lambda x: x["gtpa_1"], reverse=True)[:5]
        for r in sorted_results:
            print(f"{r['description']:>35} {r['gtpa_1']:>10.2%} {r['gtpa_10']:>10.2%} {r['avg_il']:>10.1f}")

    # 전체 베스트
    print(f"\n{'='*70}")
    print("TOP 10 Overall (by GTPA@1)")
    print(f"{'='*70}")
    print(f"{'Config':>40} {'GTPA@1':>10} {'GTPA@10':>10} {'Avg IL':>10}")
    print("-" * 75)
    sorted_all = sorted(all_results, key=lambda x: x["gtpa_1"], reverse=True)[:10]
    for r in sorted_all:
        print(f"{r['description']:>40} {r['gtpa_1']:>10.2%} {r['gtpa_10']:>10.2%} {r['avg_il']:>10.1f}")


if __name__ == "__main__":
    main()
