#!/bin/bash
# MEDDxAgent 동일 조건 비교: fixed_il=5,10,15 + adaptive
# 동일 100건 (seed=42), 최적 설정 (deny5_noante + greedy + top3_stable_5 + v15_ratio)

cd /home/max/Graph-DDXPlus
mkdir -p logs/comparison results

SCRIPT="scripts/experiment_fixed_il_comparison.py"
PORTS="7687,7688,7689,7690,7691,7692,7693,7694"

echo "======================================="
echo "MEDDxAgent 동일 조건 비교 실험 (4개)"
echo "======================================="

# 4개 모두 병렬 실행 (각각 workers=2, 서로 다른 포트)
echo "  [1/4] fixed_il=5"
nohup uv run python $SCRIPT \
    --fixed-il 5 --workers 2 --ports "7687,7688" \
    > logs/comparison/fixed_il5.log 2>&1 &
pid1=$!

echo "  [2/4] fixed_il=10"
nohup uv run python $SCRIPT \
    --fixed-il 10 --workers 2 --ports "7689,7690" \
    > logs/comparison/fixed_il10.log 2>&1 &
pid2=$!

echo "  [3/4] fixed_il=15"
nohup uv run python $SCRIPT \
    --fixed-il 15 --workers 2 --ports "7691,7692" \
    > logs/comparison/fixed_il15.log 2>&1 &
pid3=$!

echo "  [4/4] adaptive"
nohup uv run python $SCRIPT \
    --adaptive --workers 2 --ports "7693,7694" \
    > logs/comparison/adaptive.log 2>&1 &
pid4=$!

echo ""
echo "4개 병렬 실행 중... (PID: $pid1, $pid2, $pid3, $pid4)"
echo "로그: logs/comparison/"

# 완료 대기
wait $pid1 && echo "  ✓ fixed_il=5 완료" || echo "  ✗ fixed_il=5 실패"
wait $pid2 && echo "  ✓ fixed_il=10 완료" || echo "  ✗ fixed_il=10 실패"
wait $pid3 && echo "  ✓ fixed_il=15 완료" || echo "  ✗ fixed_il=15 실패"
wait $pid4 && echo "  ✓ adaptive 완료" || echo "  ✗ adaptive 실패"

echo ""
echo "======================================="
echo "결과:"
for f in results/comparison_*.json; do
    if [ -f "$f" ]; then
        mode=$(python3 -c "import json; d=json.load(open('$f')); print(d['mode'])")
        gtpa=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['results']['gtpa_1']:.2%}\")")
        il=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['results']['avg_il']:.1f}\")")
        echo "  $mode: GTPA@1=$gtpa, Avg IL=$il"
    fi
done
echo "======================================="
