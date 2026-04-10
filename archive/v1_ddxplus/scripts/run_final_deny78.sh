#!/bin/bash
# 추가 50개 실험: deny7_noante + deny8_noante × 5종료 × 5스코어링

cd /home/max/Graph-DDXPlus
mkdir -p logs/final_deny78 results

SCRIPT="scripts/experiment_final_75.py"

CANDIDATES=("7 0" "8 0")

STOPPINGS=("top3_stable_5" "top1_stable_5" "conf_gap_005" "cumulative_confirmed_5" "hr_plateau")
SCORINGS=("v15_ratio" "v18_coverage" "jaccard" "tfidf" "cosine")

is_done() {
    local dt=$1 an=$2 stop=$3 sc=$4
    local an_str="ante"; [ "$an" -eq 0 ] && an_str="noante"
    [ -f "results/final75_deny${dt}_${an_str}_${stop}_${sc}_134529.json" ]
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
echo "추가 실험 50개 (deny7,8 × noante)"
echo "완료: ${done_count}, 남은: $((total - done_count))"
echo "======================================="

pids=()
count=0

for cand in "${CANDIDATES[@]}"; do
    read dt an <<< "$cand"
    an_str="ante"; [ "$an" -eq 0 ] && an_str="noante"

    for stop in "${STOPPINGS[@]}"; do
        for sc in "${SCORINGS[@]}"; do
            if is_done "$dt" "$an" "$stop" "$sc"; then
                continue
            fi

            port=$((7687 + count % 8))
            count=$((count + 1))
            name="deny${dt}_${an_str}_${stop}_${sc}"

            nohup uv run python $SCRIPT \
                --deny-threshold "$dt" --antecedent "$an" \
                --stopping "$stop" --scoring "$sc" \
                --workers 4 --ports "$port" \
                > "logs/final_deny78/${name}.log" 2>&1 &
            pids+=($!)

            # 8개씩 배치
            if [ $((count % 8)) -eq 0 ]; then
                echo "  배치 ${count}개 실행 중... (대기)"
                for pid in "${pids[@]}"; do wait $pid; done
                pids=()
            fi
        done
    done
done

# 남은 프로세스 대기
if [ ${#pids[@]} -gt 0 ]; then
    echo "  마지막 배치 ${#pids[@]}개 대기..."
    for pid in "${pids[@]}"; do wait $pid; done
fi

completed=$(ls results/final75_deny{7,8}_noante_*.json 2>/dev/null | wc -l)
echo ""
echo "======================================="
echo "추가 완료! ${completed}/50"
echo "======================================="
