#!/usr/bin/env python3
"""Stopping Criteria 실험 v2 - 병렬 처리 + 추가 방법.

테스트 방법:
1. Evidence Coverage: confirmed + denied 수 기반
2. Disease Narrowing: 후보 질환 수 기반
3. Confidence + Stability: 높은 confidence가 N번 유지
4. Next Question Quality: 다음 질문의 예상 정보 이득

8개 Neo4j 컨테이너를 활용한 병렬 처리.
"""

import json
import math
import sys
import time
from collections import Counter
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
    description: str = ""


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

            # ========================================
            # Method 1: Evidence Coverage
            # ========================================
            if config.method == "evidence_coverage":
                threshold = config.param1
                total_evidences = len(kg.state.confirmed_cuis) + len(kg.state.denied_cuis)
                # DDXPlus 평균 증상 수 대비 비율
                coverage = total_evidences / 10.0  # 평균 10개 기준
                if coverage >= threshold:
                    should_stop = True

            # ========================================
            # Method 2: Disease Narrowing
            # ========================================
            elif config.method == "disease_narrowing":
                threshold = int(config.param1)
                min_score = config.param2
                viable = [c for c in diagnosis_candidates if c.score >= min_score]
                if len(viable) <= threshold:
                    should_stop = True

            # ========================================
            # Method 3: Confidence + Stability
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
            # Method 4: Next Question Quality
            # ========================================
            elif config.method == "next_question_quality":
                threshold = config.param1

                # 현재 entropy 계산
                scores = [c.score for c in diagnosis_candidates if c.score > 0]
                if scores:
                    total = sum(scores)
                    probs = [s / total for s in scores]
                    current_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

                    # 예상 정보 이득 = entropy 감소율
                    if prev_entropy is not None:
                        # 정규화된 정보 이득
                        if prev_entropy > 0:
                            normalized_ig = (prev_entropy - current_entropy) / prev_entropy
                        else:
                            normalized_ig = 0

                        # 정보 이득이 threshold 미만이면 종료
                        if normalized_ig < threshold and il >= 3:  # 최소 3회 질문 후
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


def run_test(loader, patients, loader_data, config: TestConfig) -> dict:
    """병렬로 테스트 실행."""
    num_workers = len(NEO4J_PORTS)

    patient_data_list = []
    for p in patients:
        patient_data_list.append({
            "age": p.age,
            "sex": p.sex,
            "initial_evidence": p.initial_evidence,
            "evidences": p.evidences,
            "pathology": p.pathology,
            "differential_diagnosis": p.differential_diagnosis,
        })

    tasks = []
    for idx, pd in enumerate(patient_data_list):
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
    print("Stopping Criteria Experiment v2 (Parallel)")
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

    all_results = []

    # ========================================
    # Test Configurations
    # ========================================
    configs = []

    # 1. Evidence Coverage: 0.3 ~ 1.5 (DDXPlus 평균 10개 대비 비율)
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
        configs.append(TestConfig(
            method="evidence_coverage",
            param1=threshold,
            description=f"EvidenceCov>={threshold}"
        ))

    # 2. Disease Narrowing: 후보 수 1~10, min_score 0.01~0.1
    for num_diseases in [1, 2, 3, 5, 7, 10]:
        for min_score in [0.01, 0.05, 0.1]:
            configs.append(TestConfig(
                method="disease_narrowing",
                param1=float(num_diseases),
                param2=min_score,
                description=f"Diseases<={num_diseases},score>={min_score}"
            ))

    # 3. Confidence + Stability: conf 0.2~0.6, stability 2~5
    for conf in [0.2, 0.3, 0.4, 0.5, 0.6]:
        for stability in [2, 3, 4, 5]:
            configs.append(TestConfig(
                method="confidence_stability",
                param1=conf,
                param2=float(stability),
                description=f"Conf>={conf},Stable>={stability}"
            ))

    # 4. Next Question Quality: IG threshold 0.01 ~ 0.3
    for threshold in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]:
        configs.append(TestConfig(
            method="next_question_quality",
            param1=threshold,
            description=f"NormIG<{threshold}"
        ))

    print(f"\nTotal configurations to test: {len(configs)}")

    # ========================================
    # Run Tests
    # ========================================
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {config.description}")
        start = time.time()
        result = run_test(loader, test_patients, loader_data, config)
        elapsed = time.time() - start

        all_results.append(result)
        print(f"  GTPA@1: {result['gtpa_1']:.2%}, Avg IL: {result['avg_il']:.1f}, Time: {elapsed/60:.1f}min")

    # ========================================
    # Save Results
    # ========================================
    output_path = Path("results/stopping_criteria_v2.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # ========================================
    # Summary Tables
    # ========================================
    print("\n" + "=" * 70)
    print("Summary by Method")
    print("=" * 70)

    for method in ["evidence_coverage", "disease_narrowing", "confidence_stability", "next_question_quality"]:
        print(f"\n### {method} ###")
        print(f"{'Config':>35} {'GTPA@1':>10} {'GTPA@10':>10} {'Avg IL':>10}")
        print("-" * 70)
        for r in all_results:
            if r["method"] == method:
                print(f"{r['description']:>35} {r['gtpa_1']:>10.2%} {r['gtpa_10']:>10.2%} {r['avg_il']:>10.1f}")


if __name__ == "__main__":
    main()
