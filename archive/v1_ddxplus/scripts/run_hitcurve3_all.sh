#!/bin/bash
# 증상 탐색 완전 요인 설계
# 2(cooccur) × 4(deny: 0,3,5,7) × 2(antecedent) × 6(selection) = 96개
# greedy만 먼저 빠르게 (16개), 나머지는 배치로

cd /home/max/Graph-DDXPlus
mkdir -p logs/hitcurve3 results

SCRIPT="scripts/analyze_hit_rate_curve_v2_full.py"
RESULT_DIR="results"

COOCCURS=(1 0)
DENY_THRESHOLDS=(0 3 5 7)
ANTECEDENTS=(1 0)
SELECTIONS=("greedy" "ig_expected" "ig_max" "ig_binary_split" "minimax_score" "minimax_entropy")

is_done() {
    local co=$1 dt=$2 an=$3 sel=$4
    local co_str="cooccur"; [ "$co" -eq 0 ] && co_str="coverage"
    local dt_str="deny${dt}"; [ "$dt" -eq 0 ] && dt_str="nodeny"
    local an_str="ante"; [ "$an" -eq 0 ] && an_str="noante"
    [ -f "${RESULT_DIR}/hitcurve3_${co_str}_${dt_str}_${an_str}_${sel}_1000.json" ]
}

total=0
done_count=0
for co in "${COOCCURS[@]}"; do
    for dt in "${DENY_THRESHOLDS[@]}"; do
        for an in "${ANTECEDENTS[@]}"; do
            for sel in "${SELECTIONS[@]}"; do
                total=$((total + 1))
                is_done "$co" "$dt" "$an" "$sel" && done_count=$((done_count + 1))
            done
        done
    done
done

echo "======================================="
echo "완전 요인 설계 Hit Rate (96개)"
echo "완료: ${done_count}, 남은: $((total - done_count))"
echo "======================================="

# 선택 전략별로 배치 (greedy 먼저, 나머지 순차)
for sel in "${SELECTIONS[@]}"; do
    echo ""
    echo "### 선택 전략: ${sel} ###"

    pids=()
    port_idx=0
    batch_count=0

    for co in "${COOCCURS[@]}"; do
        for dt in "${DENY_THRESHOLDS[@]}"; do
            for an in "${ANTECEDENTS[@]}"; do
                if is_done "$co" "$dt" "$an" "$sel"; then
                    continue
                fi

                port=$((7687 + port_idx % 8))
                port_idx=$((port_idx + 1))
                batch_count=$((batch_count + 1))

                co_str="cooccur"; [ "$co" -eq 0 ] && co_str="coverage"
                dt_str="deny${dt}"; [ "$dt" -eq 0 ] && dt_str="nodeny"
                an_str="ante"; [ "$an" -eq 0 ] && an_str="noante"
                name="${co_str}_${dt_str}_${an_str}_${sel}"

                nohup uv run python $SCRIPT \
                    --cooccur "$co" --deny-threshold "$dt" --antecedent "$an" \
                    --selection "$sel" --n-samples 1000 --workers 1 --ports "$port" \
                    > "logs/hitcurve3/${name}.log" 2>&1 &
                pids+=($!)

                # 8개씩 배치
                if [ $((batch_count % 8)) -eq 0 ]; then
                    echo "  배치 ${batch_count}개 실행 중... (대기)"
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

    echo "  ${sel} 완료!"
done

echo ""
echo "======================================="
completed=$(ls ${RESULT_DIR}/hitcurve3_*.json 2>/dev/null | wc -l)
echo "전체 완료! ${completed}/96"
echo "======================================="
