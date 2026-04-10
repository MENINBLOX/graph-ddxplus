#!/bin/bash
# 증상 탐색 Hit Rate 곡선 전체 실험
# 4개 Cypher 변형 × 6개 선택 전략 = 24개 조합
# 8포트 활용, 배치 실행

cd /home/max/Graph-DDXPlus
mkdir -p logs/hitcurve2

SCRIPT="scripts/analyze_hit_rate_curve_v2.py"

run_batch() {
    local batch_name=$1
    shift
    echo "=== Batch: $batch_name ==="
    local pids=()
    while [ $# -gt 0 ]; do
        local cypher=$1 selection=$2 port=$3
        shift 3
        local name="${cypher}+${selection}"
        local logfile="logs/hitcurve2/${cypher}_${selection}.log"
        nohup uv run python $SCRIPT \
            --cypher "$cypher" --selection "$selection" \
            --n-samples 1000 --workers 1 --ports "$port" \
            > "$logfile" 2>&1 &
        pids+=($!)
        echo "  $name (PID: $!, port: $port)"
    done
    for pid in "${pids[@]}"; do wait $pid; done
    echo "=== Batch $batch_name 완료 ==="
}

echo "======================================="
echo "Hit Rate 곡선 실험 (24개 조합)"
echo "======================================="

# Batch 1: cooccur × 모든 선택 전략 (6개, greedy는 빠름)
run_batch "cooccur" \
    cooccur greedy 7687 \
    cooccur ig_expected 7688 \
    cooccur ig_max 7689 \
    cooccur ig_binary_split 7690 \
    cooccur minimax_score 7691 \
    cooccur minimax_entropy 7692

# Batch 2: coverage_only × 모든 선택 전략
run_batch "coverage_only" \
    coverage_only greedy 7687 \
    coverage_only ig_expected 7688 \
    coverage_only ig_max 7689 \
    coverage_only ig_binary_split 7690 \
    coverage_only minimax_score 7691 \
    coverage_only minimax_entropy 7692

# Batch 3: cooccur_no_deny_filter × 모든 선택 전략
run_batch "cooccur_no_deny_filter" \
    cooccur_no_deny_filter greedy 7687 \
    cooccur_no_deny_filter ig_expected 7688 \
    cooccur_no_deny_filter ig_max 7689 \
    cooccur_no_deny_filter ig_binary_split 7690 \
    cooccur_no_deny_filter minimax_score 7691 \
    cooccur_no_deny_filter minimax_entropy 7692

# Batch 4: coverage_no_antecedent × 모든 선택 전략
run_batch "coverage_no_antecedent" \
    coverage_no_antecedent greedy 7687 \
    coverage_no_antecedent ig_expected 7688 \
    coverage_no_antecedent ig_max 7689 \
    coverage_no_antecedent ig_binary_split 7690 \
    coverage_no_antecedent minimax_score 7691 \
    coverage_no_antecedent minimax_entropy 7692

echo ""
echo "======================================="
echo "전체 실험 완료! (24개 조합)"
echo "======================================="
