#!/bin/bash
# 스코어링-독립적 종료 조건 전체 실험
# 3개 탐색 × 13개 종료 = 39개 조합
# 8포트 활용, 배치 실행

cd /home/max/Graph-DDXPlus
mkdir -p logs/stopind

SCRIPT="scripts/experiment_stopping_independent.py"

# 종료 조건 목록
STOPPINGS=(
    "consecutive_miss_3"
    "consecutive_miss_5"
    "consecutive_miss_7"
    "marginal_hr_5_10"
    "marginal_hr_5_20"
    "marginal_hr_10_10"
    "marginal_hr_10_20"
    "cumulative_confirmed_3"
    "cumulative_confirmed_5"
    "cumulative_confirmed_7"
    "cumulative_confirmed_10"
    "top3_stable_5"
    "top3_stable_7"
)

# 탐색 방법 목록
EXPLORATIONS=("greedy_cooccur" "ig_expected" "minimax_score")

run_batch() {
    local batch_name=$1
    shift
    echo "=== Batch: $batch_name ==="
    local pids=()
    while [ $# -gt 0 ]; do
        local exp=$1 stop=$2 port=$3
        shift 3
        local logfile="logs/stopind/${exp}_${stop}.log"
        nohup uv run python $SCRIPT \
            --exploration "$exp" --stopping "$stop" \
            --n-samples 1000 --workers 1 --ports "$port" \
            > "$logfile" 2>&1 &
        pids+=($!)
        echo "  ${exp}+${stop} (PID: $!, port: $port)"
    done
    for pid in "${pids[@]}"; do wait $pid; done
    echo "=== Batch $batch_name 완료 ==="
}

echo "======================================="
echo "종료 조건 실험 (3 탐색 × 13 종료 = 39개)"
echo "======================================="

# greedy_cooccur × 13개 종료 (빠름, 2배치)
run_batch "greedy_batch1" \
    greedy_cooccur consecutive_miss_3 7687 \
    greedy_cooccur consecutive_miss_5 7688 \
    greedy_cooccur consecutive_miss_7 7689 \
    greedy_cooccur marginal_hr_5_10 7690 \
    greedy_cooccur marginal_hr_5_20 7691 \
    greedy_cooccur marginal_hr_10_10 7692 \
    greedy_cooccur marginal_hr_10_20 7693

run_batch "greedy_batch2" \
    greedy_cooccur cumulative_confirmed_3 7687 \
    greedy_cooccur cumulative_confirmed_5 7688 \
    greedy_cooccur cumulative_confirmed_7 7689 \
    greedy_cooccur cumulative_confirmed_10 7690 \
    greedy_cooccur top3_stable_5 7691 \
    greedy_cooccur top3_stable_7 7692

# minimax_score × 13개 종료 (느림, 2배치)
run_batch "minimax_batch1" \
    minimax_score consecutive_miss_3 7687 \
    minimax_score consecutive_miss_5 7688 \
    minimax_score consecutive_miss_7 7689 \
    minimax_score marginal_hr_5_10 7690 \
    minimax_score marginal_hr_5_20 7691 \
    minimax_score marginal_hr_10_10 7692 \
    minimax_score marginal_hr_10_20 7693

run_batch "minimax_batch2" \
    minimax_score cumulative_confirmed_3 7687 \
    minimax_score cumulative_confirmed_5 7688 \
    minimax_score cumulative_confirmed_7 7689 \
    minimax_score cumulative_confirmed_10 7690 \
    minimax_score top3_stable_5 7691 \
    minimax_score top3_stable_7 7692

# ig_expected × 13개 종료 (느림, 2배치)
run_batch "ig_batch1" \
    ig_expected consecutive_miss_3 7687 \
    ig_expected consecutive_miss_5 7688 \
    ig_expected consecutive_miss_7 7689 \
    ig_expected marginal_hr_5_10 7690 \
    ig_expected marginal_hr_5_20 7691 \
    ig_expected marginal_hr_10_10 7692 \
    ig_expected marginal_hr_10_20 7693

run_batch "ig_batch2" \
    ig_expected cumulative_confirmed_3 7687 \
    ig_expected cumulative_confirmed_5 7688 \
    ig_expected cumulative_confirmed_7 7689 \
    ig_expected cumulative_confirmed_10 7690 \
    ig_expected top3_stable_5 7691 \
    ig_expected top3_stable_7 7692

echo ""
echo "======================================="
echo "전체 실험 완료! (39개 조합)"
echo "======================================="
