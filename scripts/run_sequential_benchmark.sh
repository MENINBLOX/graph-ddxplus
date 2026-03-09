#!/bin/bash
# 순차 벤치마크 실행 스크립트

N_SAMPLES=${1:-10000}

echo "========================================"
echo "Sequential Benchmark (n=$N_SAMPLES)"
echo "========================================"

# 1. qwen3:4b 벤치마크
echo ""
echo "[1/2] Running qwen3:4b-instruct-2507-fp16..."
PYTHONUNBUFFERED=1 uv run python scripts/benchmark_single_model.py \
    -m "qwen3:4b-instruct-2507-fp16" \
    -n $N_SAMPLES

echo ""
echo "[1/2] qwen3:4b completed!"
echo ""

# 2. gpt-oss:20b 벤치마크
echo "[2/2] Running gpt-oss:20b..."
PYTHONUNBUFFERED=1 uv run python scripts/benchmark_single_model.py \
    -m "gpt-oss:20b" \
    -n $N_SAMPLES

echo ""
echo "[2/2] gpt-oss:20b completed!"
echo ""

echo "========================================"
echo "All benchmarks completed!"
echo "========================================"

# 결과 비교
echo ""
echo "=== Results Summary ==="
echo ""
cat results/benchmark_qwen3_4b_instruct_2507_fp16_n${N_SAMPLES}.txt
echo ""
cat results/benchmark_gpt_oss_20b_n${N_SAMPLES}.txt
