#!/bin/bash
# Step 5: 최종 진단 실험 (200개 조합, 134,529건)
#
# ANOVA에서 고정한 요인: co-occurrence=Yes, antecedent=No, selection=greedy
# 변수: denied threshold (1-8) × stopping (5) × scoring (5) = 200개
#
# 사용법:
#   bash pipeline/step5_final_diagnosis.sh

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo "Step 5: Final Diagnosis (200 combinations)"
echo "============================================"

SCRIPT="scripts/experiment_final_75.py"
mkdir -p logs/final results

THRESHOLDS=(1 2 3 4 5 6 7 8)
STOPPINGS=("top3_stable_5" "top1_stable_5" "conf_gap_005" "cumulative_confirmed_5" "hr_plateau")
SCORINGS=("v15_ratio" "v18_coverage" "jaccard" "tfidf" "cosine")

is_done() {
    local dt=$1 stop=$2 sc=$3
    [ -f "results/final75_deny${dt}_noante_${stop}_${sc}_134529.json" ]
}

# 진행 상황 확인
total=0
done_count=0
for dt in "${THRESHOLDS[@]}"; do
    for stop in "${STOPPINGS[@]}"; do
        for sc in "${SCORINGS[@]}"; do
            total=$((total + 1))
            is_done "$dt" "$stop" "$sc" && done_count=$((done_count + 1))
        done
    done
done

echo ""
echo "총 조합: ${total}, 완료: ${done_count}, 남은: $((total - done_count))"
echo ""

if [ "$done_count" -eq "$total" ]; then
    echo "모든 실험이 완료되었습니다."
    exit 0
fi

# 배치 실행
pids=()
count=0

for dt in "${THRESHOLDS[@]}"; do
    for stop in "${STOPPINGS[@]}"; do
        for sc in "${SCORINGS[@]}"; do
            if is_done "$dt" "$stop" "$sc"; then
                continue
            fi

            port=$((7687 + count % 8))
            count=$((count + 1))
            name="deny${dt}_noante_${stop}_${sc}"

            echo "  실행: ${name} (port ${port})"
            nohup uv run python $SCRIPT \
                --deny-threshold "$dt" --antecedent 0 \
                --stopping "$stop" --scoring "$sc" \
                --workers 4 --ports "$port" \
                > "logs/final/${name}.log" 2>&1 &
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

# 완료 확인
completed=$(ls results/final75_deny*_noante_*_134529.json 2>/dev/null | wc -l)
echo ""
echo "============================================"
echo "완료: ${completed}/${total}"
echo "============================================"
