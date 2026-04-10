#!/bin/bash
# 전체 벤치마크 실행 스크립트
# GPU0: Top-2, 4, 6, 8, 10 (짝수) - 순차 실행
# GPU1: Top-3, 5, 7, 9 (홀수) - 순차 실행

SAMPLES=27389
SEVERITY=2
CATEGORY=2

echo "=========================================="
echo "Starting full benchmark at $(date)"
echo "Samples: $SAMPLES, Severity: $SEVERITY"
echo "GPU0: Top-2, 4, 6, 8, 10"
echo "GPU1: Top-3, 5, 7, 9"
echo "=========================================="

# GPU0: 짝수 Top-N (순차 실행)
run_gpu0() {
    for N in 2 4 6 8 10; do
        echo "[GPU0] Starting Top-$N at $(date)"
        CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_vllm.py \
            --category $CATEGORY -n $SAMPLES --severity $SEVERITY --top-n $N \
            >> benchmark_gpu0.log 2>&1
        echo "[GPU0] Completed Top-$N at $(date)"
    done
    echo "[GPU0] All done at $(date)"
}

# GPU1: 홀수 Top-N (순차 실행)
run_gpu1() {
    for N in 3 5 7 9; do
        echo "[GPU1] Starting Top-$N at $(date)"
        CUDA_VISIBLE_DEVICES=1 uv run python scripts/benchmark_vllm.py \
            --category $CATEGORY -n $SAMPLES --severity $SEVERITY --top-n $N \
            >> benchmark_gpu1.log 2>&1
        echo "[GPU1] Completed Top-$N at $(date)"
    done
    echo "[GPU1] All done at $(date)"
}

# 두 GPU 병렬 실행
run_gpu0 &
run_gpu1 &

echo "Benchmarks started on both GPUs."
echo "Monitor progress:"
echo "  tail -f benchmark_gpu0.log"
echo "  tail -f benchmark_gpu1.log"

wait
echo "=========================================="
echo "All benchmarks completed at $(date)"
echo "=========================================="
