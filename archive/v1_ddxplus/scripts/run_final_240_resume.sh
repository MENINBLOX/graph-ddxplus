#!/bin/bash
# 최종 실험 재시작 - 완료된 것 건너뛰기, workers=2로 속도 향상
# 8포트 × workers=2 = 16 병렬 워커

cd /home/max/Graph-DDXPlus
mkdir -p logs/final results

SCRIPT="scripts/experiment_final_240.py"
RESULT_DIR="results"

EXPLORATIONS=("greedy_cooccur" "minimax_score" "ig_expected")
STOPPINGS=(
    "consecutive_miss_5"
    "marginal_hr_5_10"
    "cumulative_confirmed_5"
    "hr_plateau"
    "top1_stable_5"
    "top3_stable_5"
    "confidence_03"
    "conf_gap_005"
    "entropy_20"
    "confidence_05"
)
SCORINGS=("v15_ratio" "v18_coverage" "naive_bayes" "log_likelihood" "jaccard" "tfidf" "bm25" "cosine")

# 완료 여부 확인 함수
is_done() {
    local name="${1}_${2}_${3}"
    [ -f "${RESULT_DIR}/final_${name}_134529.json" ]
}

total=0
skipped=0
remaining=0

for e in "${EXPLORATIONS[@]}"; do
    for s in "${STOPPINGS[@]}"; do
        for sc in "${SCORINGS[@]}"; do
            total=$((total + 1))
            if is_done "$e" "$s" "$sc"; then
                skipped=$((skipped + 1))
            else
                remaining=$((remaining + 1))
            fi
        done
    done
done

echo "======================================="
echo "최종 실험 재시작 (workers=2, 속도 2배)"
echo "전체: ${total}, 완료: ${skipped}, 남은: ${remaining}"
echo "======================================="

batch_num=0

for exp in "${EXPLORATIONS[@]}"; do
    for stop in "${STOPPINGS[@]}"; do
        # 이 배치에서 실행할 것이 있는지 확인
        batch_tasks=()
        for scoring in "${SCORINGS[@]}"; do
            if ! is_done "$exp" "$stop" "$scoring"; then
                batch_tasks+=("$scoring")
            fi
        done

        if [ ${#batch_tasks[@]} -eq 0 ]; then
            echo "  [SKIP] ${exp}+${stop}: 전부 완료"
            continue
        fi

        batch_num=$((batch_num + 1))
        echo ""
        echo "=== Batch ${batch_num}: ${exp}+${stop} (${#batch_tasks[@]}개) ==="

        pids=()
        port_idx=0
        for scoring in "${batch_tasks[@]}"; do
            port=$((7687 + port_idx))
            port_idx=$(( (port_idx + 1) % 8 ))

            name="${exp}_${stop}_${scoring}"
            logfile="logs/final/${name}.log"

            nohup uv run python $SCRIPT \
                --exploration "$exp" \
                --stopping "$stop" \
                --scoring "$scoring" \
                --workers 4 \
                --ports "$port" \
                > "$logfile" 2>&1 &
            pids+=($!)
            echo "  ${scoring} (PID: $!, port: ${port}, workers=2)"
        done

        for pid in "${pids[@]}"; do
            wait $pid
        done
        echo "=== Batch ${batch_num} 완료 ==="
    done
done

echo ""
echo "======================================="
echo "전체 실험 완료!"
completed=$(ls ${RESULT_DIR}/final_*.json 2>/dev/null | wc -l)
echo "총 결과: ${completed}/240"
echo "======================================="
