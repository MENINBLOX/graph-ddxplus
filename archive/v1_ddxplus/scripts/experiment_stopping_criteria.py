#!/usr/bin/env python3
"""다양한 Stopping Criteria 실험.

테스트 방법:
1. Entropy 기반: 진단 분포의 entropy가 임계값 미만이면 종료
2. Information Gain 기반: 마지막 질문의 정보 이득이 임계값 미만이면 종료
3. Rank Stability 기반: Top-K 순위가 N번 연속 동일하면 종료

모든 방법은 min_il=0으로 테스트 (순수하게 해당 기준만으로 stopping)
"""

import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class StoppingResult:
    method: str
    threshold: float
    count: int
    correct_at_1: int
    correct_at_10: int
    gtpa_1: float
    gtpa_10: float
    avg_il: float


def calculate_entropy(scores: list[float]) -> float:
    """점수 리스트의 entropy 계산."""
    if not scores or sum(scores) == 0:
        return 0.0

    # 확률로 정규화
    total = sum(scores)
    probs = [s / total for s in scores if s > 0]

    # Shannon entropy
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy


def get_top_k_ranks(candidates, k: int = 3) -> tuple:
    """Top-K 후보의 CUI 튜플 반환."""
    return tuple(c.cui for c in candidates[:k])


def run_entropy_test(loader, patients, entropy_threshold: float, max_il: int = 50) -> StoppingResult:
    """Entropy 기반 stopping 테스트."""
    from src.umls_kg import UMLSKG

    correct_at_1 = 0
    correct_at_10 = 0
    total_il = 0
    count = 0

    for p in patients:
        kg = UMLSKG()

        gt_disease_eng = loader.fr_to_eng.get(p.pathology, p.pathology)
        gt_cui = loader.get_disease_cui(gt_disease_eng)

        patient_positive_cuis = set()
        for ev_str in p.evidences:
            code = ev_str.split("_@_")[0] if "_@_" in ev_str else ev_str
            cui = loader.get_symptom_cui(code)
            if cui:
                patient_positive_cuis.add(cui)

        initial_cui = loader.get_symptom_cui(p.initial_evidence)
        if not initial_cui:
            kg.close()
            continue

        kg.reset_state()
        kg.state.add_confirmed(initial_cui)

        il = 0
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

            # Entropy 기반 stopping
            diagnosis_candidates = kg.get_diagnosis_candidates(top_k=10)
            if diagnosis_candidates:
                scores = [c.score for c in diagnosis_candidates]
                entropy = calculate_entropy(scores)

                # Entropy가 충분히 낮으면 종료 (확신이 높음)
                if entropy < entropy_threshold:
                    break

            # max_il 도달
            if il >= max_il:
                break

        final_candidates = kg.get_diagnosis_candidates(top_k=10)
        if final_candidates:
            if final_candidates[0].cui == gt_cui:
                correct_at_1 += 1
            for c in final_candidates[:10]:
                if c.cui == gt_cui:
                    correct_at_10 += 1
                    break

        total_il += il
        count += 1
        kg.close()

    return StoppingResult(
        method="entropy",
        threshold=entropy_threshold,
        count=count,
        correct_at_1=correct_at_1,
        correct_at_10=correct_at_10,
        gtpa_1=correct_at_1 / count if count > 0 else 0,
        gtpa_10=correct_at_10 / count if count > 0 else 0,
        avg_il=total_il / count if count > 0 else 0,
    )


def run_info_gain_test(loader, patients, ig_threshold: float, max_il: int = 50) -> StoppingResult:
    """Information Gain 기반 stopping 테스트."""
    from src.umls_kg import UMLSKG

    correct_at_1 = 0
    correct_at_10 = 0
    total_il = 0
    count = 0

    for p in patients:
        kg = UMLSKG()

        gt_disease_eng = loader.fr_to_eng.get(p.pathology, p.pathology)
        gt_cui = loader.get_disease_cui(gt_disease_eng)

        patient_positive_cuis = set()
        for ev_str in p.evidences:
            code = ev_str.split("_@_")[0] if "_@_" in ev_str else ev_str
            cui = loader.get_symptom_cui(code)
            if cui:
                patient_positive_cuis.add(cui)

        initial_cui = loader.get_symptom_cui(p.initial_evidence)
        if not initial_cui:
            kg.close()
            continue

        kg.reset_state()
        kg.state.add_confirmed(initial_cui)

        il = 0
        prev_entropy = None
        consecutive_low_ig = 0

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

            # Information Gain 계산
            diagnosis_candidates = kg.get_diagnosis_candidates(top_k=10)
            if diagnosis_candidates:
                scores = [c.score for c in diagnosis_candidates]
                current_entropy = calculate_entropy(scores)

                if prev_entropy is not None:
                    info_gain = prev_entropy - current_entropy

                    # Information Gain이 임계값 미만이면 카운트
                    if info_gain < ig_threshold:
                        consecutive_low_ig += 1
                    else:
                        consecutive_low_ig = 0

                    # 2번 연속 낮은 IG면 종료
                    if consecutive_low_ig >= 2:
                        break

                prev_entropy = current_entropy

            if il >= max_il:
                break

        final_candidates = kg.get_diagnosis_candidates(top_k=10)
        if final_candidates:
            if final_candidates[0].cui == gt_cui:
                correct_at_1 += 1
            for c in final_candidates[:10]:
                if c.cui == gt_cui:
                    correct_at_10 += 1
                    break

        total_il += il
        count += 1
        kg.close()

    return StoppingResult(
        method="info_gain",
        threshold=ig_threshold,
        count=count,
        correct_at_1=correct_at_1,
        correct_at_10=correct_at_10,
        gtpa_1=correct_at_1 / count if count > 0 else 0,
        gtpa_10=correct_at_10 / count if count > 0 else 0,
        avg_il=total_il / count if count > 0 else 0,
    )


def run_rank_stability_test(loader, patients, stability_k: int, stability_n: int, max_il: int = 50) -> StoppingResult:
    """Rank Stability 기반 stopping 테스트.

    Args:
        stability_k: 확인할 Top-K 개수
        stability_n: 연속 동일해야 하는 횟수
    """
    from src.umls_kg import UMLSKG

    correct_at_1 = 0
    correct_at_10 = 0
    total_il = 0
    count = 0

    for p in patients:
        kg = UMLSKG()

        gt_disease_eng = loader.fr_to_eng.get(p.pathology, p.pathology)
        gt_cui = loader.get_disease_cui(gt_disease_eng)

        patient_positive_cuis = set()
        for ev_str in p.evidences:
            code = ev_str.split("_@_")[0] if "_@_" in ev_str else ev_str
            cui = loader.get_symptom_cui(code)
            if cui:
                patient_positive_cuis.add(cui)

        initial_cui = loader.get_symptom_cui(p.initial_evidence)
        if not initial_cui:
            kg.close()
            continue

        kg.reset_state()
        kg.state.add_confirmed(initial_cui)

        il = 0
        rank_history = deque(maxlen=stability_n)

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

            # Rank Stability 확인
            diagnosis_candidates = kg.get_diagnosis_candidates(top_k=stability_k)
            current_ranks = get_top_k_ranks(diagnosis_candidates, stability_k)
            rank_history.append(current_ranks)

            # N번 연속 동일하면 종료
            if len(rank_history) == stability_n:
                if all(r == rank_history[0] for r in rank_history):
                    break

            if il >= max_il:
                break

        final_candidates = kg.get_diagnosis_candidates(top_k=10)
        if final_candidates:
            if final_candidates[0].cui == gt_cui:
                correct_at_1 += 1
            for c in final_candidates[:10]:
                if c.cui == gt_cui:
                    correct_at_10 += 1
                    break

        total_il += il
        count += 1
        kg.close()

    return StoppingResult(
        method=f"rank_stability_k{stability_k}_n{stability_n}",
        threshold=float(f"{stability_k}.{stability_n}"),  # 표현용
        count=count,
        correct_at_1=correct_at_1,
        correct_at_10=correct_at_10,
        gtpa_1=correct_at_1 / count if count > 0 else 0,
        gtpa_10=correct_at_10 / count if count > 0 else 0,
        avg_il=total_il / count if count > 0 else 0,
    )


def main():
    from src.data_loader import DDXPlusLoader

    print("=" * 70)
    print("Stopping Criteria Experiment")
    print("=" * 70)

    loader = DDXPlusLoader()
    _ = loader.symptom_mapping
    _ = loader.disease_mapping
    _ = loader.fr_to_eng

    test_patients = loader.load_patients(split="test")
    print(f"Total test patients: {len(test_patients):,}")

    all_results = []

    # ========================================
    # 1. Entropy 기반 테스트
    # ========================================
    print("\n" + "=" * 70)
    print("Part 1: Entropy-based Stopping")
    print("=" * 70)

    # 49개 질환의 최대 entropy: log2(49) ≈ 5.6
    # 촘촘하게: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
    entropy_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    for threshold in entropy_thresholds:
        print(f"\nTesting entropy < {threshold}...")
        start = time.time()
        result = run_entropy_test(loader, test_patients, entropy_threshold=threshold)
        elapsed = time.time() - start
        all_results.append(result)
        print(f"  GTPA@1: {result.gtpa_1:.2%}, Avg IL: {result.avg_il:.1f}, Time: {elapsed/60:.1f}min")

    # ========================================
    # 2. Information Gain 기반 테스트
    # ========================================
    print("\n" + "=" * 70)
    print("Part 2: Information Gain-based Stopping")
    print("=" * 70)

    # IG threshold: 매우 작은 값부터 큰 값까지
    ig_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    for threshold in ig_thresholds:
        print(f"\nTesting info_gain < {threshold}...")
        start = time.time()
        result = run_info_gain_test(loader, test_patients, ig_threshold=threshold)
        elapsed = time.time() - start
        all_results.append(result)
        print(f"  GTPA@1: {result.gtpa_1:.2%}, Avg IL: {result.avg_il:.1f}, Time: {elapsed/60:.1f}min")

    # ========================================
    # 3. Rank Stability 기반 테스트
    # ========================================
    print("\n" + "=" * 70)
    print("Part 3: Rank Stability-based Stopping")
    print("=" * 70)

    # (K, N) 조합: Top-K가 N번 연속 동일하면 종료
    stability_configs = [
        (1, 2), (1, 3), (1, 4), (1, 5),  # Top-1 안정성
        (3, 2), (3, 3), (3, 4), (3, 5),  # Top-3 안정성
        (5, 2), (5, 3), (5, 4), (5, 5),  # Top-5 안정성
    ]

    for k, n in stability_configs:
        print(f"\nTesting Top-{k} stable for {n} turns...")
        start = time.time()
        result = run_rank_stability_test(loader, test_patients, stability_k=k, stability_n=n)
        elapsed = time.time() - start
        all_results.append(result)
        print(f"  GTPA@1: {result.gtpa_1:.2%}, Avg IL: {result.avg_il:.1f}, Time: {elapsed/60:.1f}min")

    # ========================================
    # 결과 저장
    # ========================================
    output_path = Path("results/stopping_criteria_experiment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dict = {}
    for r in all_results:
        key = f"{r.method}_{r.threshold}"
        results_dict[key] = {
            "method": r.method,
            "threshold": r.threshold,
            "count": r.count,
            "correct_at_1": r.correct_at_1,
            "correct_at_10": r.correct_at_10,
            "gtpa_1": r.gtpa_1,
            "gtpa_10": r.gtpa_10,
            "avg_il": r.avg_il,
        }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # ========================================
    # 요약 테이블
    # ========================================
    print("\n" + "=" * 70)
    print("Summary: Entropy-based")
    print("=" * 70)
    print(f"{'Threshold':>12} {'GTPA@1':>10} {'GTPA@10':>10} {'Avg IL':>10}")
    print("-" * 45)
    for r in all_results:
        if r.method == "entropy":
            print(f"{r.threshold:>12.1f} {r.gtpa_1:>10.2%} {r.gtpa_10:>10.2%} {r.avg_il:>10.1f}")

    print("\n" + "=" * 70)
    print("Summary: Information Gain-based")
    print("=" * 70)
    print(f"{'Threshold':>12} {'GTPA@1':>10} {'GTPA@10':>10} {'Avg IL':>10}")
    print("-" * 45)
    for r in all_results:
        if r.method == "info_gain":
            print(f"{r.threshold:>12.3f} {r.gtpa_1:>10.2%} {r.gtpa_10:>10.2%} {r.avg_il:>10.1f}")

    print("\n" + "=" * 70)
    print("Summary: Rank Stability-based")
    print("=" * 70)
    print(f"{'Config':>12} {'GTPA@1':>10} {'GTPA@10':>10} {'Avg IL':>10}")
    print("-" * 45)
    for r in all_results:
        if r.method.startswith("rank_stability"):
            config = r.method.replace("rank_stability_", "")
            print(f"{config:>12} {r.gtpa_1:>10.2%} {r.gtpa_10:>10.2%} {r.avg_il:>10.1f}")


if __name__ == "__main__":
    main()
