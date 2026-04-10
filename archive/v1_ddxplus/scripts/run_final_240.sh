#!/bin/bash
# 최종 실험 240개 - 수정된 버전 (종료 판단은 기존 스코어링, 최종 진단만 커스텀)
# 8포트 × workers=4 = 32 병렬 워커
# 종료 조건 2개씩 묶어서 16개 동시 실행

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

is_done() {
    [ -f "${RESULT_DIR}/final_${1}_${2}_${3}_134529.json" ]
}

total=0
skipped=0
for e in "${EXPLORATIONS[@]}"; do
    for s in "${STOPPINGS[@]}"; do
        for sc in "${SCORINGS[@]}"; do
            total=$((total + 1))
            is_done "$e" "$s" "$sc" && skipped=$((skipped + 1))
        done
    done
done

echo "======================================="
echo "최종 실험 (수정된 버전, workers=4)"
echo "전체: ${total}, 완료: ${skipped}, 남은: $((total - skipped))"
echo "======================================="

for exp in "${EXPLORATIONS[@]}"; do
    echo ""
    echo "########################################"
    echo "# 탐색: ${exp}"
    echo "########################################"

    # 종료 조건을 2개씩 묶어서 동시 실행 (8스코어링×2종료 = 16개 동시)
    stop_idx=0
    while [ $stop_idx -lt ${#STOPPINGS[@]} ]; do
        stop1="${STOPPINGS[$stop_idx]}"
        stop2="${STOPPINGS[$((stop_idx + 1))]:-}"
        stop_idx=$((stop_idx + 2))

        pids=()

        # 종료 1
        port_idx=0
        has_work=0
        for scoring in "${SCORINGS[@]}"; do
            if ! is_done "$exp" "$stop1" "$scoring"; then
                port=$((7687 + port_idx % 8))
                port_idx=$((port_idx + 1))
                name="${exp}_${stop1}_${scoring}"
                nohup uv run python $SCRIPT \
                    --exploration "$exp" --stopping "$stop1" --scoring "$scoring" \
                    --workers 4 --ports "$port" \
                    > "logs/final/${name}.log" 2>&1 &
                pids+=($!)
                has_work=1
            fi
        done

        # 종료 2 (있으면)
        if [ -n "$stop2" ]; then
            for scoring in "${SCORINGS[@]}"; do
                if ! is_done "$exp" "$stop2" "$scoring"; then
                    port=$((7687 + port_idx % 8))
                    port_idx=$((port_idx + 1))
                    name="${exp}_${stop2}_${scoring}"
                    nohup uv run python $SCRIPT \
                        --exploration "$exp" --stopping "$stop2" --scoring "$scoring" \
                        --workers 4 --ports "$port" \
                        > "logs/final/${name}.log" 2>&1 &
                    pids+=($!)
                    has_work=1
                fi
            done
        fi

        if [ $has_work -eq 1 ]; then
            batch_desc="${exp}+${stop1}"
            [ -n "$stop2" ] && batch_desc="${batch_desc} & ${stop2}"
            echo "=== ${batch_desc}: ${#pids[@]}개 동시 실행 ==="
            for pid in "${pids[@]}"; do wait $pid; done
            echo "=== 완료 ==="
        else
            echo "  [SKIP] ${stop1}$([ -n "$stop2" ] && echo " & ${stop2}"): 전부 완료"
        fi
    done
done

completed=$(ls ${RESULT_DIR}/final_*.json 2>/dev/null | wc -l)
echo ""
echo "======================================="
echo "전체 실험 완료! ${completed}/240"
echo "======================================="
