#!/bin/bash
# reason_tokens 변화에 따른 성능 테스트
# 10,000 샘플, Top-5 기준, severity=2

SAMPLES=10000
SEVERITY=2
TOP_N=5
CATEGORY=2

echo "=========================================="
echo "Reason Tokens 테스트"
echo "Samples: $SAMPLES, Top-N: $TOP_N"
echo "테스트 값: 100, 256, 512, 1024, 2048, 4096"
echo "=========================================="

# GPU0: 100, 512, 2048
run_gpu0() {
    for TOKENS in 100 512 2048; do
        echo "[GPU0] Starting reason_tokens=$TOKENS at $(date)"
        CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_vllm.py \
            --category $CATEGORY -n $SAMPLES --severity $SEVERITY \
            --top-n $TOP_N --reason-tokens $TOKENS \
            >> benchmark_reason_gpu0.log 2>&1
        echo "[GPU0] Completed reason_tokens=$TOKENS at $(date)"
    done
    echo "[GPU0] All done at $(date)"
}

# GPU1: 256, 1024, 4096
run_gpu1() {
    for TOKENS in 256 1024 4096; do
        echo "[GPU1] Starting reason_tokens=$TOKENS at $(date)"
        CUDA_VISIBLE_DEVICES=1 uv run python scripts/benchmark_vllm.py \
            --category $CATEGORY -n $SAMPLES --severity $SEVERITY \
            --top-n $TOP_N --reason-tokens $TOKENS \
            >> benchmark_reason_gpu1.log 2>&1
        echo "[GPU1] Completed reason_tokens=$TOKENS at $(date)"
    done
    echo "[GPU1] All done at $(date)"
}

# 로그 초기화
rm -f benchmark_reason_gpu0.log benchmark_reason_gpu1.log

# 두 GPU 병렬 실행
run_gpu0 &
run_gpu1 &

echo "Tests started on both GPUs."
echo "Monitor progress:"
echo "  tail -f benchmark_reason_gpu0.log"
echo "  tail -f benchmark_reason_gpu1.log"

wait
echo "=========================================="
echo "All tests completed at $(date)"
echo "=========================================="
