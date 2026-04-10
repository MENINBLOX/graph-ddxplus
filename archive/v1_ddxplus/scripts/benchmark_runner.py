#!/usr/bin/env python3
"""벤치마크 러너 - 각 모델을 별도 프로세스로 실행하여 GPU 메모리 누수 방지.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_runner.py -n 100
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_runner.py -n 1000 --category 2
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.models_config import SMALL_LLM_MODELS


def run_single_model(model: str, n_samples: int, category: int, output_dir: Path) -> dict:
    """단일 모델 벤치마크 실행 (별도 프로세스)."""
    model_short = model.split("/")[-1].replace("-", "_").replace(".", "_")

    script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")
from scripts.benchmark_vllm import VLLMBenchmark
import json

mode = "small_llm" if {category} == 1 else "small_llm_kg"
benchmark = VLLMBenchmark(
    model="{model}",
    mode=mode,
    scoring="v18_coverage",
    severity=2,
)
result = benchmark.run(n_samples={n_samples})
benchmark.close()

# 결과 출력 (JSON)
print("###RESULT###")
print(json.dumps(result))
'''

    print(f"\n{'='*70}")
    print(f"Running: {model}")
    print("=" * 70)

    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", script],
            capture_output=True,
            text=True,
            timeout=600,  # 10분 타임아웃
            cwd=Path(__file__).parent.parent,
        )

        # 결과 파싱
        output = result.stdout
        if "###RESULT###" in output:
            result_json = output.split("###RESULT###")[1].strip().split("\n")[0]
            return json.loads(result_json)
        else:
            print(f"STDOUT: {output[-2000:]}")
            print(f"STDERR: {result.stderr[-2000:]}")
            return {"model": model, "error": "No result found"}

    except subprocess.TimeoutExpired:
        return {"model": model, "error": "Timeout"}
    except Exception as e:
        return {"model": model, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Benchmark Runner")
    parser.add_argument("-n", "--n-samples", type=int, default=100)
    parser.add_argument("--category", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()

    # 출력 디렉토리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Results will be saved to: {output_dir}")
    print(f"Models: {len(SMALL_LLM_MODELS)}")
    print(f"Samples: {args.n_samples}")
    print(f"Category: {args.category}")

    all_results = []

    for i, model in enumerate(SMALL_LLM_MODELS, 1):
        print(f"\n[{i}/{len(SMALL_LLM_MODELS)}] {model}")
        result = run_single_model(model, args.n_samples, args.category, output_dir)
        all_results.append(result)

        # 중간 저장
        with open(output_dir / "results_partial.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # 최종 요약
    print(f"\n{'='*90}")
    print("FINAL SUMMARY")
    print("=" * 90)
    print(f"{'Model':<40} {'GTPA@1':>8} {'DDR':>8} {'DDF1':>8} {'IL':>6}")
    print("-" * 90)

    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<40} ERROR: {r['error'][:30]}")
        else:
            model_short = r["model"].split("/")[-1][:37]
            print(f"{model_short:<40} {r['gtpa_at_1']:>7.1%} {r['ddr']:>7.1%} {r['ddf1']:>7.1%} {r['avg_il']:>6.1f}")

    print("-" * 90)
    print(f"{'AARLC (baseline)':<40} {'75.39%':>8} {'97.73%':>8} {'78.24%':>8} {'25.75':>6}")

    # 최종 저장
    with open(output_dir / "results_final.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Completed {len(all_results)} benchmarks")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
