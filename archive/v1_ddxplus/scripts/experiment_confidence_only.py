#!/usr/bin/env python3
"""min_il=0으로 confidence threshold만 변화시키는 실험.

목적: min_il 없이 confidence만으로 stopping했을 때 성능 확인
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_single_test(loader, patients, confidence_threshold: float) -> dict:
    """단일 confidence threshold로 테스트."""
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
        max_il = 50  # 안전장치

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

            # min_il=0으로 confidence만 평가
            should_stop, _ = kg.should_stop(
                max_il=max_il,
                min_il=0,  # min_il 없음
                confidence_threshold=confidence_threshold,
                gap_threshold=0.005,
                relative_gap_threshold=1.5,
            )
            if should_stop:
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

    return {
        "confidence_threshold": confidence_threshold,
        "count": count,
        "correct_at_1": correct_at_1,
        "correct_at_10": correct_at_10,
        "gtpa_1": correct_at_1 / count if count > 0 else 0,
        "gtpa_10": correct_at_10 / count if count > 0 else 0,
        "avg_il": total_il / count if count > 0 else 0,
    }


def main():
    from src.data_loader import DDXPlusLoader

    print("=" * 70)
    print("Confidence-Only Stopping (min_il=0)")
    print("=" * 70)

    loader = DDXPlusLoader()
    _ = loader.symptom_mapping
    _ = loader.disease_mapping
    _ = loader.fr_to_eng

    test_patients = loader.load_patients(split="test")
    print(f"Total test patients: {len(test_patients):,}")

    results = {}

    # confidence threshold 범위: 0.1 ~ 0.99
    confidence_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    for conf in confidence_values:
        print(f"\nTesting confidence={conf:.2f}...")
        start = time.time()
        result = run_single_test(loader, test_patients, confidence_threshold=conf)
        elapsed = time.time() - start

        results[f"conf_{conf:.2f}"] = result
        print(f"  GTPA@1: {result['gtpa_1']:.2%}, Avg IL: {result['avg_il']:.1f}, Time: {elapsed/60:.1f}min")

    # 결과 저장
    output_path = Path("results/confidence_only_experiment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # 요약 테이블
    print("\n" + "=" * 70)
    print("Summary: Confidence-Only Stopping (min_il=0)")
    print("=" * 70)
    print(f"{'Confidence':>12} {'GTPA@1':>10} {'GTPA@10':>10} {'Avg IL':>10}")
    print("-" * 45)
    for conf in confidence_values:
        key = f"conf_{conf:.2f}"
        if key in results:
            r = results[key]
            print(f"{conf:>12.2f} {r['gtpa_1']:>10.2%} {r['gtpa_10']:>10.2%} {r['avg_il']:>10.1f}")


if __name__ == "__main__":
    main()
