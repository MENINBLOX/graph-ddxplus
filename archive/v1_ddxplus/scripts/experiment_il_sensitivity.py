#!/usr/bin/env python3
"""min_il과 max_il의 민감도 분석.

목적:
1. min_il 변화에 따른 GTPA@1, Avg IL 변화 측정
2. max_il 변화에 따른 영향 측정
3. IL 분포 분석 (조기 종료 비율)

결과는 논문의 min_il/max_il 설정 근거로 활용.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_single_test(loader, patients, min_il: int, max_il: int = 50) -> dict:
    """단일 설정으로 테스트 실행."""
    from src.umls_kg import UMLSKG
    from collections import Counter

    correct_at_1 = 0
    correct_at_10 = 0
    total_il = 0
    count = 0
    il_distribution = Counter()
    max_il_reached = 0

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

            should_stop, reason = kg.should_stop(
                max_il=max_il,
                min_il=min_il,
                confidence_threshold=0.30,
                gap_threshold=0.005,
                relative_gap_threshold=1.5,
            )
            if should_stop:
                if "max_il" in reason:
                    max_il_reached += 1
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
        il_distribution[il] += 1
        count += 1
        kg.close()

    return {
        "min_il": min_il,
        "max_il": max_il,
        "count": count,
        "correct_at_1": correct_at_1,
        "correct_at_10": correct_at_10,
        "gtpa_1": correct_at_1 / count if count > 0 else 0,
        "gtpa_10": correct_at_10 / count if count > 0 else 0,
        "avg_il": total_il / count if count > 0 else 0,
        "max_il_reached": max_il_reached,
        "max_il_reached_pct": max_il_reached / count if count > 0 else 0,
        "il_distribution": dict(il_distribution),
    }


def main():
    from src.data_loader import DDXPlusLoader

    print("=" * 70)
    print("IL Sensitivity Analysis")
    print("=" * 70)

    loader = DDXPlusLoader()
    _ = loader.symptom_mapping
    _ = loader.disease_mapping
    _ = loader.fr_to_eng

    # 전체 테스트셋 사용
    test_patients = loader.load_patients(split="test")
    total = len(test_patients)
    print(f"Total test patients: {total:,}")

    # 샘플링 (빠른 테스트)
    # sample_size = 5000
    # import random
    # random.seed(42)
    # test_patients = random.sample(test_patients, sample_size)
    # print(f"Sampled: {sample_size:,}")

    results = {}

    # 1. min_il 민감도 분석 (max_il=50 고정)
    print("\n" + "=" * 70)
    print("Part 1: min_il Sensitivity (max_il=50 fixed)")
    print("=" * 70)

    min_il_values = list(range(0, 26))  # 0~25 모든 값 테스트

    for min_il in min_il_values:
        print(f"\nTesting min_il={min_il}...")
        start = time.time()
        result = run_single_test(loader, test_patients, min_il=min_il, max_il=50)
        elapsed = time.time() - start

        results[f"min_il_{min_il}"] = result
        print(f"  GTPA@1: {result['gtpa_1']:.2%}, Avg IL: {result['avg_il']:.1f}, "
              f"max_il reached: {result['max_il_reached_pct']:.2%}, Time: {elapsed/60:.1f}min")

    # 2. max_il 민감도 분석 (min_il=13 고정)
    print("\n" + "=" * 70)
    print("Part 2: max_il Sensitivity (min_il=13 fixed)")
    print("=" * 70)

    max_il_values = [20, 30, 40, 50, 75, 100]

    for max_il in max_il_values:
        print(f"\nTesting max_il={max_il}...")
        start = time.time()
        result = run_single_test(loader, test_patients, min_il=13, max_il=max_il)
        elapsed = time.time() - start

        results[f"max_il_{max_il}"] = result
        print(f"  GTPA@1: {result['gtpa_1']:.2%}, Avg IL: {result['avg_il']:.1f}, "
              f"max_il reached: {result['max_il_reached_pct']:.2%}, Time: {elapsed/60:.1f}min")

    # 3. IL 분포 분석 (min_il=0)
    print("\n" + "=" * 70)
    print("Part 3: Natural IL Distribution (min_il=0)")
    print("=" * 70)

    natural_result = results.get("min_il_0")
    if natural_result:
        il_dist = natural_result["il_distribution"]
        total_cases = sum(il_dist.values())

        print("\nIL Distribution (without min_il constraint):")
        cumulative = 0
        for il in sorted(il_dist.keys()):
            count = il_dist[il]
            cumulative += count
            pct = count / total_cases * 100
            cum_pct = cumulative / total_cases * 100
            print(f"  IL={il:2d}: {count:5d} ({pct:5.1f}%) | cumulative: {cum_pct:5.1f}%")

        under_13 = sum(c for il, c in il_dist.items() if il < 13)
        print(f"\nCases with IL < 13: {under_13:,} ({under_13/total_cases*100:.1f}%)")

    # 결과 저장
    output_path = Path("results/il_sensitivity_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # il_distribution을 문자열 키로 변환 (JSON 호환)
    for key in results:
        if "il_distribution" in results[key]:
            results[key]["il_distribution"] = {
                str(k): v for k, v in results[key]["il_distribution"].items()
            }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # 요약 테이블
    print("\n" + "=" * 70)
    print("Summary: min_il Sensitivity")
    print("=" * 70)
    print(f"{'min_il':>8} {'GTPA@1':>10} {'GTPA@10':>10} {'Avg IL':>10} {'max_il%':>10}")
    print("-" * 50)
    for min_il in min_il_values:
        key = f"min_il_{min_il}"
        if key in results:
            r = results[key]
            print(f"{min_il:>8} {r['gtpa_1']:>10.2%} {r['gtpa_10']:>10.2%} "
                  f"{r['avg_il']:>10.1f} {r['max_il_reached_pct']:>10.2%}")

    print("\n" + "=" * 70)
    print("Summary: max_il Sensitivity")
    print("=" * 70)
    print(f"{'max_il':>8} {'GTPA@1':>10} {'GTPA@10':>10} {'Avg IL':>10} {'max_il%':>10}")
    print("-" * 50)
    for max_il in max_il_values:
        key = f"max_il_{max_il}"
        if key in results:
            r = results[key]
            print(f"{max_il:>8} {r['gtpa_1']:>10.2%} {r['gtpa_10']:>10.2%} "
                  f"{r['avg_il']:>10.1f} {r['max_il_reached_pct']:>10.2%}")


if __name__ == "__main__":
    main()
