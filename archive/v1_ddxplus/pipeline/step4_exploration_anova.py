#!/usr/bin/env python3
"""Step 4: 증상 탐색 요인 분석 (ANOVA).

4개 요인(co-occurrence, denied threshold, antecedent, selection strategy)의
264개 전 조합(2×11×2×6)을 검증셋 1,000건에서 평가.

이 스크립트는 scripts/analyze_hit_rate_curve_v2.py를 264회 실행하고
결과를 종합하여 ANOVA를 수행한다.

사용법:
  # 전체 264개 조합 실행 (수 시간 소요)
  uv run python pipeline/step4_exploration_anova.py

  # 결과만 분석 (이미 실행된 경우)
  uv run python pipeline/step4_exploration_anova.py --analyze-only
"""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats


CYPHERS = ["cooccur", "coverage"]
THRESHOLDS = list(range(0, 11))  # 0=no filter, 1-10
ANTECEDENTS = [0, 1]  # 0=No, 1=Yes
SELECTIONS = [
    "greedy",
    "ig_binary_split",
    "ig_expected",
    "ig_max",
    "minimax_entropy",
    "minimax_score",
]

N_SAMPLES = 1000
SCRIPT = "scripts/analyze_hit_rate_curve_v2.py"


def run_experiments(ports: str, workers: int) -> None:
    """264개 조합 실행."""
    total = len(CYPHERS) * len(THRESHOLDS) * len(ANTECEDENTS) * len(SELECTIONS)
    done = len(glob.glob("results/hitcurve_val_*_1000.json"))
    print(f"총 {total}개 조합, 완료: {done}, 남은: {total - done}")

    count = 0
    for cypher in CYPHERS:
        for threshold in THRESHOLDS:
            deny_str = f"deny{threshold}" if threshold > 0 else "nodeny"
            for ante in ANTECEDENTS:
                ante_str = "ante" if ante else "noante"
                for selection in SELECTIONS:
                    # 이미 완료된 결과 확인
                    pattern = f"results/hitcurve_val_{cypher}_{deny_str}_{ante_str}_{selection}_{N_SAMPLES}.json"
                    if glob.glob(pattern):
                        continue

                    count += 1
                    name = f"{cypher}_{deny_str}_{ante_str}_{selection}"
                    print(f"\n[{count}] {name}")

                    cmd = [
                        "uv", "run", "python", SCRIPT,
                        "--cypher", cypher,
                        "--deny-threshold", str(threshold),
                        "--antecedent", str(ante),
                        "--selection", selection,
                        "--n-samples", str(N_SAMPLES),
                        "--workers", str(workers),
                        "--ports", ports,
                    ]
                    subprocess.run(cmd)


def analyze_results() -> None:
    """264개 결과를 로드하여 ANOVA 수행."""
    files = glob.glob("results/hitcurve_val_*_1000.json")
    print(f"\n=== ANOVA 분석 ({len(files)}개 결과) ===")

    data = []
    for f in sorted(files):
        with open(f) as fh:
            d = json.load(fh)
        cooccur = "cooccur" if "_cooccur_" in f else "coverage"
        data.append({
            "cooccur": cooccur,
            "threshold": d.get("deny_threshold", 0),
            "antecedent": d.get("antecedent", 0),
            "selection": d.get("selection", "greedy"),
            "hit_rate": d.get("avg_hit_rate", 0),
        })

    if len(data) < 264:
        print(f"  [경고] {264 - len(data)}개 결과 누락. --analyze-only 없이 다시 실행하세요.")

    hr = np.array([d["hit_rate"] for d in data])
    grand_mean = hr.mean()
    ss_total = np.sum((hr - grand_mean) ** 2)

    def calc_ss(key: str) -> tuple[float, int]:
        levels = sorted(set(d[key] for d in data))
        ss = 0.0
        for lev in levels:
            mask = np.array([i for i, d in enumerate(data) if d[key] == lev])
            group_mean = hr[mask].mean()
            ss += len(mask) * (group_mean - grand_mean) ** 2
        return ss, len(levels) - 1

    factors = ["threshold", "antecedent", "selection", "cooccur"]
    results = {}
    for factor in factors:
        ss, df = calc_ss(factor)
        results[factor] = {"ss": ss, "df": df}

    ss_residual = ss_total - sum(r["ss"] for r in results.values())
    df_residual = len(data) - 1 - sum(r["df"] for r in results.values())
    ms_residual = ss_residual / df_residual if df_residual > 0 else 1e-10

    print(f"\nGrand Mean Hit Rate: {grand_mean * 100:.2f}%")
    print(f"\n| Source             | df  | F        | p        | η²     |")
    print(f"|{'-'*20}|{'-'*5}|{'-'*10}|{'-'*10}|{'-'*8}|")

    for factor in ["threshold", "antecedent", "selection", "cooccur"]:
        r = results[factor]
        ms = r["ss"] / r["df"] if r["df"] > 0 else 0
        f_val = ms / ms_residual if ms_residual > 0 else 0
        p_val = 1 - stats.f.cdf(f_val, r["df"], df_residual)
        eta2 = r["ss"] / ss_total if ss_total > 0 else 0

        label = {
            "threshold": "Denied threshold",
            "antecedent": "Antecedent",
            "selection": "Selection strategy",
            "cooccur": "Co-occurrence",
        }[factor]

        p_str = "<.001" if p_val < 0.001 else f"{p_val:.3f}"
        print(f"| {label:<18} | {r['df']:<3} | {f_val:>8.1f} | {p_str:>8} | {eta2:.4f} |")

    print(f"| {'Residual':<18} | {df_residual:<3} |          |          |        |")

    # 결과 저장
    output_path = Path("results") / "anova_exploration.json"
    output = {
        "n_combinations": len(data),
        "grand_mean_hit_rate": float(grand_mean),
        "factors": {
            factor: {
                "ss": float(results[factor]["ss"]),
                "df": int(results[factor]["df"]),
                "eta_squared": float(results[factor]["ss"] / ss_total),
            }
            for factor in factors
        },
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n저장: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ports", type=str, default="7687")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 4: Exploration Factor ANOVA (264 combinations)")
    print("=" * 60)

    if not args.analyze_only:
        run_experiments(args.ports, args.workers)

    analyze_results()


if __name__ == "__main__":
    main()
