#!/bin/bash
# 최종 실험: 후보 생성 3 × 종료 5 × 스코어링 5 = 75개
# greedy 고정, 134,529건 전체, workers=4

cd /home/max/Graph-DDXPlus
mkdir -p logs/final75 results

SCRIPT="scripts/experiment_final_75.py"
RESULT_DIR="results"

# 후보 생성 3개: (deny_threshold, antecedent)
CANDIDATES=("3 0" "5 0" "5 1")

# 종료 5개
STOPPINGS=("top3_stable_5" "top1_stable_5" "conf_gap_005" "cumulative_confirmed_5" "hr_plateau")

# 스코어링 5개
SCORINGS=("v15_ratio" "v18_coverage" "jaccard" "tfidf" "cosine")

is_done() {
    local dt=$1 an=$2 stop=$3 sc=$4
    local an_str="ante"; [ "$an" -eq 0 ] && an_str="noante"
    [ -f "${RESULT_DIR}/final75_deny${dt}_${an_str}_${stop}_${sc}_134529.json" ]
}

total=0
done_count=0
for cand in "${CANDIDATES[@]}"; do
    read dt an <<< "$cand"
    for stop in "${STOPPINGS[@]}"; do
        for sc in "${SCORINGS[@]}"; do
            total=$((total + 1))
            is_done "$dt" "$an" "$stop" "$sc" && done_count=$((done_count + 1))
        done
    done
done

echo "======================================="
echo "최종 실험 75개 (workers=4)"
echo "완료: ${done_count}, 남은: $((total - done_count))"
echo "======================================="

for cand in "${CANDIDATES[@]}"; do
    read dt an <<< "$cand"
    an_str="ante"; [ "$an" -eq 0 ] && an_str="noante"

    echo ""
    echo "### 후보 생성: deny${dt}_${an_str} ###"

    for stop in "${STOPPINGS[@]}"; do
        # 이 배치에서 실행할 것이 있는지 확인
        batch_tasks=()
        for sc in "${SCORINGS[@]}"; do
            if ! is_done "$dt" "$an" "$stop" "$sc"; then
                batch_tasks+=("$sc")
            fi
        done

        if [ ${#batch_tasks[@]} -eq 0 ]; then
            echo "  [SKIP] ${stop}: 전부 완료"
            continue
        fi

        echo "  === deny${dt}_${an_str}+${stop} (${#batch_tasks[@]}개) ==="
        pids=()
        port_idx=0
        for sc in "${batch_tasks[@]}"; do
            port=$((7687 + port_idx % 8))
            port_idx=$((port_idx + 1))
            name="deny${dt}_${an_str}_${stop}_${sc}"
            nohup uv run python $SCRIPT \
                --deny-threshold "$dt" --antecedent "$an" \
                --stopping "$stop" --scoring "$sc" \
                --workers 4 --ports "$port" \
                > "logs/final75/${name}.log" 2>&1 &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait $pid; done
        echo "  === 완료 ==="
    done
done

completed=$(ls ${RESULT_DIR}/final75_*.json 2>/dev/null | wc -l)
echo ""
echo "======================================="
echo "전체 완료! ${completed}/75"
echo "======================================="
