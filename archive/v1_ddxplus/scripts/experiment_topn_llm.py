#!/usr/bin/env python3
"""LLM+KG Top-N 실험.

LLM에 제시하는 후보 개수(Top-N)를 변경하며 성능 측정.
단일 모델로 빠르게 테스트.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/experiment_topn_llm.py -n 1000 --severity 2
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/experiment_topn_llm.py -n 27389 --severity 2  # 전체 데이터
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# 단일 모델 사용 (가장 빠른 모델)
TEST_MODEL = "google/gemma-3-270m-it"


def run_topn_experiment(
    top_n: int,
    n_samples: int,
    severity: int,
    scoring: str = "v18_coverage",
) -> dict:
    """단일 Top-N 값으로 벤치마크 실행."""
    from scripts.benchmark_vllm import VLLMBenchmark

    benchmark = VLLMBenchmark(
        model=TEST_MODEL,
        mode="small_llm_kg",
        scoring=scoring,
        severity=severity,
        top_n=top_n,
    )

    try:
        result = benchmark.run(n_samples)
        result["top_n"] = top_n
        return result
    finally:
        benchmark.close()
        del benchmark
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="LLM+KG Top-N Experiment")
    parser.add_argument("-n", "--n-samples", type=int, default=1000)
    parser.add_argument("--severity", type=int, default=2, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--scoring", type=str, default="v18_coverage")
    parser.add_argument("--min-n", type=int, default=2, help="Minimum Top-N value")
    parser.add_argument("--max-n", type=int, default=10, help="Maximum Top-N value")

    args = parser.parse_args()

    print(f"LLM+KG Top-N Experiment")
    print(f"Model: {TEST_MODEL}")
    print(f"Samples: {args.n_samples:,}, severity={args.severity}")
    print(f"Top-N range: {args.min_n} to {args.max_n}")
    print("=" * 70)

    results = []
    start_time = time.time()

    # Top-N 범위 테스트
    for top_n in tqdm(range(args.min_n, args.max_n + 1), desc="Top-N Experiments"):
        tqdm.write(f"\n--- Testing Top-{top_n} ---")

        result = run_topn_experiment(
            top_n=top_n,
            n_samples=args.n_samples,
            severity=args.severity,
            scoring=args.scoring,
        )
        results.append(result)

        tqdm.write(
            f"Top-{top_n}: GTPA@1={result['gtpa_at_1']:.2%}, "
            f"IL={result['avg_il']:.1f}"
        )

    total_time = time.time() - start_time

    # 결과 요약
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (LLM+KG)")
    print("=" * 70)
    print(f"{'Top-N':>6} {'GTPA@1':>10} {'Avg IL':>10}")
    print("-" * 70)

    best_result = max(results, key=lambda x: x["gtpa_at_1"])

    for r in results:
        marker = " *" if r == best_result else ""
        print(
            f"{r['top_n']:>6} {r['gtpa_at_1']:>9.2%} "
            f"{r['avg_il']:>10.2f}{marker}"
        )

    print("-" * 70)
    print(f"Best: Top-{best_result['top_n']} with GTPA@1={best_result['gtpa_at_1']:.2%}")
    print(f"Total time: {total_time/60:.1f} min")

    # 결과 저장
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "topn_llm_experiment.json"

    save_data = {
        "model": TEST_MODEL,
        "n_samples": args.n_samples,
        "severity": args.severity,
        "scoring": args.scoring,
        "total_time_seconds": total_time,
        "results": [
            {
                "top_n": r["top_n"],
                "gtpa_at_1": r["gtpa_at_1"],
                "avg_il": r["avg_il"],
                "ddr": r.get("ddr", 0),
                "ddf1": r.get("ddf1", 0),
            }
            for r in results
        ],
    }

    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
