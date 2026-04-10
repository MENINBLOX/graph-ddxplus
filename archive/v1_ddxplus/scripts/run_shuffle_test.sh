#!/bin/bash
# Position Bias 테스트 (셔플)
# 1024/2048/8192 테스트 완료 후 실행

echo "=========================================="
echo "Position Bias 테스트 (Shuffle)"
echo "Samples: 10000, Top-N: 5, reason_tokens: 100"
echo "=========================================="

# GPU0 사용 (1024/8192 완료 후)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_vllm.py \
    --category 2 -n 10000 --severity 2 \
    --top-n 5 --reason-tokens 100 --shuffle-candidates

echo "=========================================="
echo "테스트 완료: $(date)"
echo "=========================================="
