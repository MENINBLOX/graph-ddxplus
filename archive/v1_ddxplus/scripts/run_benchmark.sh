#!/bin/bash
# =============================================================================
# DDXPlus Benchmark - 통합 실행 스크립트
# =============================================================================
#
# Category 1: LLM only (baseline)
# Category 2: LLM + KG (제안 방법)
#   - Adaptive IL (KG가 진단 타이밍 결정)
#   - Fixed IL = 5, 10, 15 (MEDDxAgent 비교용)
#
# Usage:
#   # GPU 0에서 Category 1 실행
#   ./scripts/run_benchmark.sh --category 1 --gpu 0 -n 27389
#
#   # GPU 1에서 Category 2 실행 (Adaptive + Fixed IL 5,10,15)
#   ./scripts/run_benchmark.sh --category 2 --gpu 1 -n 27389
#
#   # 전체 실행 (Category 1 + 2, 각각 GPU 0, 1)
#   ./scripts/run_benchmark.sh --all -n 27389
#
# =============================================================================

set -e

# 기본값
N_SAMPLES=27389
GPU=0
CATEGORY=""
RUN_ALL=false
SEVERITY=2

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --severity)
            SEVERITY="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -n, --n-samples N    샘플 수 (default: 27389)"
            echo "  --gpu GPU            GPU 번호 (default: 0)"
            echo "  --category CAT       1=LLM only, 2=LLM+KG"
            echo "  --all                Category 1,2 모두 실행 (GPU 0,1 병렬)"
            echo "  --severity SEV       질환 심각도 (default: 2)"
            echo "  -h, --help           도움말"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 타임스탬프
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="/home/max/Graph-DDXPlus/results/${TIMESTAMP}"

# 모델 리스트
MODELS=(
    "Qwen/Qwen3-4B-Thinking-2507"
    "Qwen/Qwen3-VL-4B-Thinking"
    "Qwen/Qwen3-4B-Instruct-2507"
    "mistralai/Ministral-3-3B-Instruct-2512"
    "microsoft/Phi-4-mini-instruct"
    "ai21labs/AI21-Jamba-Reasoning-3B"
    "Qwen/Qwen3-VL-4B-Instruct"
    "google/gemma-3-270m-it"
    "LGAI-EXAONE/EXAONE-4.0-1.2B"
    "LiquidAI/LFM2.5-1.2B-Thinking"
    "meta-llama/Llama-3.1-8B-Instruct"
)

# =============================================================================
# Category 1: LLM only
# =============================================================================
run_category1() {
    local gpu=$1
    local output_dir="${BASE_OUTPUT_DIR}/category1_llm_only"
    mkdir -p "$output_dir"

    echo "=============================================="
    echo "Category 1: LLM only (GPU $gpu)"
    echo "Output: $output_dir"
    echo "Samples: $N_SAMPLES"
    echo "=============================================="

    local results_file="$output_dir/results.csv"
    echo "Model,GTPA@1,DDR,DDF1,IL,Time" > "$results_file"

    for i in "${!MODELS[@]}"; do
        local model="${MODELS[$i]}"
        local model_short=$(echo "$model" | sed 's/.*\///')

        echo ""
        echo "[$(($i+1))/${#MODELS[@]}] $model"
        echo "----------------------------------------------"

        local result_file="$output_dir/result_${i}.json"

        CUDA_VISIBLE_DEVICES=$gpu timeout 14400 uv run python -c "
import sys, json
sys.path.insert(0, '/home/max/Graph-DDXPlus')
from scripts.benchmark_vllm import VLLMBenchmark

try:
    benchmark = VLLMBenchmark(
        model='$model',
        mode='small_llm',
        scoring='v18_coverage',
        severity=$SEVERITY,
    )
    result = benchmark.run(n_samples=$N_SAMPLES)
    benchmark.close()
    with open('$result_file', 'w') as f:
        json.dump(result, f)
    print('SUCCESS')
except Exception as e:
    import traceback
    traceback.print_exc()
    with open('$result_file', 'w') as f:
        json.dump({'error': str(e)}, f)
" 2>&1

        _save_result "$result_file" "$results_file" "$model_short"

        pkill -9 -f "EngineCore" 2>/dev/null || true
        sleep 3
    done

    echo ""
    echo "=============================================="
    echo "Category 1 완료"
    echo "=============================================="
    column -t -s',' "$results_file"
}

# =============================================================================
# Category 2: LLM + KG
# =============================================================================
run_category2() {
    local gpu=$1
    local output_dir="${BASE_OUTPUT_DIR}/category2_llm_kg"
    mkdir -p "$output_dir"

    echo "=============================================="
    echo "Category 2: LLM + KG (GPU $gpu)"
    echo "Output: $output_dir"
    echo "Samples: $N_SAMPLES"
    echo "=============================================="

    # IL 설정: adaptive (None) + fixed (5, 10, 15)
    local IL_SETTINGS=("adaptive" "5" "10" "15")

    for il_setting in "${IL_SETTINGS[@]}"; do
        local il_dir="$output_dir/il_${il_setting}"
        mkdir -p "$il_dir"

        local results_file="$il_dir/results.csv"
        echo "Model,GTPA@1,DDR,DDF1,IL,Time" > "$results_file"

        echo ""
        echo "=============================================="
        echo "IL = $il_setting"
        echo "=============================================="

        for i in "${!MODELS[@]}"; do
            local model="${MODELS[$i]}"
            local model_short=$(echo "$model" | sed 's/.*\///')

            echo ""
            echo "[$(($i+1))/${#MODELS[@]}] $model (IL=$il_setting)"
            echo "----------------------------------------------"

            local result_file="$il_dir/result_${i}.json"

            # max_il 파라미터 설정
            local max_il_param=""
            if [ "$il_setting" != "adaptive" ]; then
                max_il_param="max_il=$il_setting,"
            fi

            CUDA_VISIBLE_DEVICES=$gpu timeout 14400 uv run python -c "
import sys, json
sys.path.insert(0, '/home/max/Graph-DDXPlus')
from scripts.benchmark_vllm import VLLMBenchmark

try:
    benchmark = VLLMBenchmark(
        model='$model',
        mode='small_llm_kg',
        ${max_il_param}
        scoring='v18_coverage',
        severity=$SEVERITY,
    )
    result = benchmark.run(n_samples=$N_SAMPLES)
    benchmark.close()
    with open('$result_file', 'w') as f:
        json.dump(result, f)
    print('SUCCESS')
except Exception as e:
    import traceback
    traceback.print_exc()
    with open('$result_file', 'w') as f:
        json.dump({'error': str(e)}, f)
" 2>&1

            _save_result "$result_file" "$results_file" "$model_short"

            pkill -9 -f "EngineCore" 2>/dev/null || true
            sleep 3
        done

        echo ""
        echo "--- IL=$il_setting 완료 ---"
        column -t -s',' "$results_file"
    done

    echo ""
    echo "=============================================="
    echo "Category 2 전체 완료"
    echo "=============================================="
    _print_category2_summary "$output_dir"
}

# =============================================================================
# 결과 저장 헬퍼
# =============================================================================
_save_result() {
    local result_file=$1
    local results_csv=$2
    local model_short=$3

    if [ -f "$result_file" ]; then
        if grep -q '"gtpa_at_1"' "$result_file" 2>/dev/null; then
            local gtpa=$(python3 -c "import json; d=json.load(open('$result_file')); print(f\"{d['gtpa_at_1']*100:.1f}%\")")
            local ddr=$(python3 -c "import json; d=json.load(open('$result_file')); print(f\"{d['ddr']*100:.1f}%\")")
            local ddf1=$(python3 -c "import json; d=json.load(open('$result_file')); print(f\"{d['ddf1']*100:.1f}%\")")
            local il=$(python3 -c "import json; d=json.load(open('$result_file')); print(f\"{d['avg_il']:.1f}\")")
            local time=$(python3 -c "import json; d=json.load(open('$result_file')); print(f\"{d['elapsed_minutes']:.1f}m\")")

            echo "✓ GTPA@1=$gtpa, DDR=$ddr, DDF1=$ddf1, IL=$il"
            echo "$model_short,$gtpa,$ddr,$ddf1,$il,$time" >> "$results_csv"
        else
            local error=$(python3 -c "import json; d=json.load(open('$result_file')); print(d.get('error', 'Unknown')[:50])" 2>/dev/null || echo "Unknown")
            echo "✗ Error: $error"
            echo "$model_short,ERROR,,,," >> "$results_csv"
        fi
    else
        echo "✗ No result file"
        echo "$model_short,ERROR,,,," >> "$results_csv"
    fi
}

# =============================================================================
# Category 2 요약 출력
# =============================================================================
_print_category2_summary() {
    local output_dir=$1

    echo ""
    echo "=============================================="
    echo "Category 2 Summary (LLM + KG)"
    echo "=============================================="

    for il in "adaptive" "5" "10" "15"; do
        local results_file="$output_dir/il_${il}/results.csv"
        if [ -f "$results_file" ]; then
            echo ""
            echo "--- IL = $il ---"
            column -t -s',' "$results_file"
        fi
    done

    echo ""
    echo "AARLC Baseline: GTPA@1=75.39%, DDR=97.73%, DDF1=78.24%, IL=25.75"
}

# =============================================================================
# 메인 실행
# =============================================================================

if [ "$RUN_ALL" = true ]; then
    echo "=============================================="
    echo "전체 실행: Category 1 (GPU 0) + Category 2 (GPU 1)"
    echo "=============================================="

    mkdir -p "$BASE_OUTPUT_DIR"

    # 병렬 실행
    run_category1 0 > "${BASE_OUTPUT_DIR}/category1.log" 2>&1 &
    PID1=$!
    echo "Category 1 started: PID $PID1 (GPU 0)"

    run_category2 1 > "${BASE_OUTPUT_DIR}/category2.log" 2>&1 &
    PID2=$!
    echo "Category 2 started: PID $PID2 (GPU 1)"

    echo ""
    echo "로그 확인:"
    echo "  tail -f ${BASE_OUTPUT_DIR}/category1.log"
    echo "  tail -f ${BASE_OUTPUT_DIR}/category2.log"

    # 완료 대기
    wait $PID1 $PID2

    echo ""
    echo "=============================================="
    echo "전체 완료"
    echo "결과: $BASE_OUTPUT_DIR"
    echo "=============================================="

elif [ "$CATEGORY" = "1" ]; then
    mkdir -p "$BASE_OUTPUT_DIR"
    run_category1 $GPU

elif [ "$CATEGORY" = "2" ]; then
    mkdir -p "$BASE_OUTPUT_DIR"
    run_category2 $GPU

else
    echo "Error: --category 1 또는 --category 2 또는 --all 옵션이 필요합니다."
    echo "도움말: $0 --help"
    exit 1
fi

echo ""
echo "결과 디렉토리: $BASE_OUTPUT_DIR"
