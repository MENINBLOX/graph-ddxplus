#!/usr/bin/env python3
"""남은 실패 케이스 분석 (매핑 수정 후)."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    # 매핑 수정 후 결과 로드
    with open("results/mapping_only_experiment.json") as f:
        results = json.load(f)

    fixed_stats = results["pls_irreg_fixed"]["disease_stats"]

    # 질환별 실패 분석
    failures = []
    for disease, stats in fixed_stats.items():
        total = stats["total"]
        correct_10 = stats["correct_10"]
        failed = total - correct_10
        if failed > 0:
            rate = correct_10 / total
            failures.append({
                "disease": disease,
                "total": total,
                "correct_10": correct_10,
                "failed": failed,
                "gtpa_10": rate
            })

    # 실패 건수로 정렬
    failures.sort(key=lambda x: -x["failed"])

    print("=" * 70)
    print("남은 GTPA@10 실패 케이스 분석 (매핑 수정 후)")
    print("=" * 70)
    print(f"\n총 실패: {sum(f['failed'] for f in failures)} cases")
    print(f"총 질환: {len(failures)} diseases\n")

    print(f"{'Disease':<40} {'Total':>8} {'Failed':>8} {'GTPA@10':>10}")
    print("-" * 70)

    for f in failures:
        print(f"{f['disease']:<40} {f['total']:>8} {f['failed']:>8} {f['gtpa_10']:>9.2%}")

    # 카테고리별 분류
    print("\n" + "=" * 70)
    print("실패 원인 카테고리화")
    print("=" * 70)

    # 대량 실패 (>10건)
    major_failures = [f for f in failures if f["failed"] > 10]
    minor_failures = [f for f in failures if f["failed"] <= 10]

    print(f"\n주요 실패 (>10건): {len(major_failures)} diseases, {sum(f['failed'] for f in major_failures)} cases")
    for f in major_failures:
        print(f"  - {f['disease']}: {f['failed']} failures ({f['gtpa_10']:.2%} GTPA@10)")

    print(f"\n소수 실패 (<=10건): {len(minor_failures)} diseases, {sum(f['failed'] for f in minor_failures)} cases")
    for f in minor_failures:
        print(f"  - {f['disease']}: {f['failed']} failures")


if __name__ == "__main__":
    main()
