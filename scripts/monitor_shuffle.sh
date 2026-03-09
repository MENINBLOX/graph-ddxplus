#!/bin/bash
# 30분 단위 모니터링 스크립트

LOG_FILE="/home/max/Graph-DDXPlus/monitor_shuffle.log"

echo "========================================" >> $LOG_FILE
echo "모니터링 시작: $(date)" >> $LOG_FILE
echo "========================================" >> $LOG_FILE

while true; do
    echo "" >> $LOG_FILE
    echo "======== $(date '+%Y-%m-%d %H:%M:%S') ========" >> $LOG_FILE

    # GPU 상태
    echo "[GPU 상태]" >> $LOG_FILE
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> $LOG_FILE

    # 2048 진행률
    echo "" >> $LOG_FILE
    echo "[2048 tokens + shuffle]" >> $LOG_FILE
    PROGRESS_2048=$(grep "Processed prompts" /home/max/Graph-DDXPlus/benchmark_shuffle_2048.log 2>/dev/null | tail -1 | grep -oP '\d+%|\d+/\d+' | head -2 | tr '\n' ' ')
    echo "  진행률: $PROGRESS_2048" >> $LOG_FILE

    # 2048 round 완료 확인
    ROUNDS_2048=$(grep -c "round=" /home/max/Graph-DDXPlus/benchmark_shuffle_2048.log 2>/dev/null || echo "0")
    echo "  완료된 rounds: $ROUNDS_2048" >> $LOG_FILE

    # 2048 모델 진행 확인
    MODEL_2048=$(grep "Model:" /home/max/Graph-DDXPlus/benchmark_shuffle_2048.log 2>/dev/null | tail -1)
    echo "  현재 모델: $MODEL_2048" >> $LOG_FILE

    # 2048 Debug 출력 (추론 샘플)
    DEBUG_2048=$(grep -A5 "\[Two-Stage Debug" /home/max/Graph-DDXPlus/benchmark_shuffle_2048.log 2>/dev/null | tail -10)
    if [ -n "$DEBUG_2048" ]; then
        echo "  [추론 샘플]" >> $LOG_FILE
        echo "$DEBUG_2048" >> $LOG_FILE
    fi

    # 8192 진행률
    echo "" >> $LOG_FILE
    echo "[8192 tokens + shuffle]" >> $LOG_FILE
    PROGRESS_8192=$(grep "Processed prompts" /home/max/Graph-DDXPlus/benchmark_shuffle_8192.log 2>/dev/null | tail -1 | grep -oP '\d+%|\d+/\d+' | head -2 | tr '\n' ' ')
    echo "  진행률: $PROGRESS_8192" >> $LOG_FILE

    # 8192 round 완료 확인
    ROUNDS_8192=$(grep -c "round=" /home/max/Graph-DDXPlus/benchmark_shuffle_8192.log 2>/dev/null || echo "0")
    echo "  완료된 rounds: $ROUNDS_8192" >> $LOG_FILE

    # 8192 모델 진행 확인
    MODEL_8192=$(grep "Model:" /home/max/Graph-DDXPlus/benchmark_shuffle_8192.log 2>/dev/null | tail -1)
    echo "  현재 모델: $MODEL_8192" >> $LOG_FILE

    # 8192 Debug 출력 (추론 샘플)
    DEBUG_8192=$(grep -A5 "\[Two-Stage Debug" /home/max/Graph-DDXPlus/benchmark_shuffle_8192.log 2>/dev/null | tail -10)
    if [ -n "$DEBUG_8192" ]; then
        echo "  [추론 샘플]" >> $LOG_FILE
        echo "$DEBUG_8192" >> $LOG_FILE
    fi

    # 완료 여부 확인
    if grep -q "All models completed" /home/max/Graph-DDXPlus/benchmark_shuffle_2048.log 2>/dev/null; then
        echo "" >> $LOG_FILE
        echo "★★★ 2048 테스트 완료! ★★★" >> $LOG_FILE
        grep "GTPA@1" /home/max/Graph-DDXPlus/benchmark_shuffle_2048.log | tail -5 >> $LOG_FILE
    fi

    if grep -q "All models completed" /home/max/Graph-DDXPlus/benchmark_shuffle_8192.log 2>/dev/null; then
        echo "" >> $LOG_FILE
        echo "★★★ 8192 테스트 완료! ★★★" >> $LOG_FILE
        grep "GTPA@1" /home/max/Graph-DDXPlus/benchmark_shuffle_8192.log | tail -5 >> $LOG_FILE
    fi

    # 둘 다 완료되면 종료
    if grep -q "All models completed" /home/max/Graph-DDXPlus/benchmark_shuffle_2048.log 2>/dev/null && \
       grep -q "All models completed" /home/max/Graph-DDXPlus/benchmark_shuffle_8192.log 2>/dev/null; then
        echo "" >> $LOG_FILE
        echo "========================================" >> $LOG_FILE
        echo "모든 테스트 완료: $(date)" >> $LOG_FILE
        echo "========================================" >> $LOG_FILE
        break
    fi

    # 30분 대기
    sleep 1800
done
