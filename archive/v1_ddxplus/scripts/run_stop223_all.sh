#!/bin/bash
# 종료 조건 비교 실험 일괄 실행 (max_il=223)
# 8개 Neo4j 포트를 활용하여 4개씩 동시 실행

cd /home/max/Graph-DDXPlus
mkdir -p logs/stop223

SCRIPT="scripts/experiment_stopping_max223.py"

run_batch() {
    local batch_name=$1
    shift
    echo "=== Batch: $batch_name ==="

    local pids=()
    while [ $# -gt 0 ]; do
        local method=$1 p1=$2 p2=$3 desc=$4 workers=$5 ports=$6
        shift 6

        local logfile="logs/stop223/${desc}.log"
        nohup uv run python $SCRIPT \
            --method "$method" --param1 "$p1" --param2 "$p2" \
            --desc "$desc" --workers "$workers" --ports "$ports" \
            > "$logfile" 2>&1 &
        pids+=($!)
        echo "  Started: $desc (PID: $!)"
    done

    # Wait for all
    for pid in "${pids[@]}"; do
        wait $pid
    done
    echo "=== Batch $batch_name 완료 ==="
}

echo "====================================="
echo "종료 조건 비교 실험 (max_il=223)"
echo "====================================="

# Batch 1: Rank Stability (4개 동시)
run_batch "RankStability-1" \
    rank_stability 3 3 "Top3_stable_3" 4 "7687,7688" \
    rank_stability 3 4 "Top3_stable_4" 4 "7689,7690" \
    rank_stability 3 5 "Top3_stable_5" 4 "7691,7692" \
    rank_stability 3 7 "Top3_stable_7" 4 "7693,7694"

# Batch 2: Rank Stability 계속
run_batch "RankStability-2" \
    rank_stability 5 3 "Top5_stable_3" 4 "7687,7688" \
    rank_stability 5 5 "Top5_stable_5" 4 "7689,7690" \
    rank_stability 1 3 "Top1_stable_3" 4 "7691,7692" \
    rank_stability 1 5 "Top1_stable_5" 4 "7693,7694"

# Batch 3: Confidence (4개 동시)
run_batch "Confidence" \
    confidence_only 0.1 0 "conf_ge0.1" 4 "7687,7688" \
    confidence_only 0.3 0 "conf_ge0.3" 4 "7689,7690" \
    confidence_only 0.5 0 "conf_ge0.5" 4 "7691,7692" \
    confidence_only 0.7 0 "conf_ge0.7" 4 "7693,7694"

# Batch 4: Confidence Stability (4개 동시)
run_batch "ConfStability" \
    confidence_stability 0.3 3 "conf0.3_stable3" 4 "7687,7688" \
    confidence_stability 0.3 5 "conf0.3_stable5" 4 "7689,7690" \
    confidence_stability 0.5 3 "conf0.5_stable3" 4 "7691,7692" \
    confidence_stability 0.5 5 "conf0.5_stable5" 4 "7693,7694"

# Batch 5: Entropy (4개 동시)
run_batch "Entropy" \
    entropy 1.0 0 "entropy_lt1.0" 4 "7687,7688" \
    entropy 2.0 0 "entropy_lt2.0" 4 "7689,7690" \
    entropy 3.0 0 "entropy_lt3.0" 4 "7691,7692" \
    info_gain 0.01 3 "IG_lt0.01_3cons" 4 "7693,7694"

# Batch 6: Info Gain + Confidence Gap
run_batch "IG-ConfGap" \
    info_gain 0.001 2 "IG_lt0.001_2cons" 4 "7687,7688" \
    info_gain 0.01 2 "IG_lt0.01_2cons" 4 "7689,7690" \
    confidence_gap 0.3 0.05 "confgap_0.3_0.05" 4 "7691,7692" \
    confidence_gap 0.5 0.1 "confgap_0.5_0.1" 4 "7693,7694"

echo ""
echo "====================================="
echo "전체 실험 완료!"
echo "====================================="
