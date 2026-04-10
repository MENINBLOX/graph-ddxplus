#!/bin/bash
# Category 2: LLM + KG (GPU 1)

N_SAMPLES=${1:-27389}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/home/max/Graph-DDXPlus/results/cat2_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Category 2: LLM + KG (GPU 1)"
echo "Results: $OUTPUT_DIR"
echo "Samples: $N_SAMPLES"
echo ""

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

RESULTS_FILE="$OUTPUT_DIR/all_results.csv"
echo "Model,GTPA@1,DDR,DDF1,IL,Time" > "$RESULTS_FILE"

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_SHORT=$(echo "$MODEL" | sed 's/.*\///')

    echo ""
    echo "=========================================="
    echo "[$(($i+1))/${#MODELS[@]}] $MODEL"
    echo "=========================================="

    sleep 3

    RESULT_FILE="$OUTPUT_DIR/result_${i}.json"

    CUDA_VISIBLE_DEVICES=1 timeout 14400 uv run python -c "
import sys
import json
sys.path.insert(0, '/home/max/Graph-DDXPlus')
from scripts.benchmark_vllm import VLLMBenchmark

try:
    benchmark = VLLMBenchmark(
        model='$MODEL',
        mode='small_llm_kg',  # Category 2: LLM + KG
        scoring='v18_coverage',
        severity=2,
    )
    result = benchmark.run(n_samples=$N_SAMPLES)
    benchmark.close()
    with open('$RESULT_FILE', 'w') as f:
        json.dump(result, f)
    print('SUCCESS')
except Exception as e:
    with open('$RESULT_FILE', 'w') as f:
        json.dump({'error': str(e)}, f)
    print(f'ERROR: {e}')
" 2>&1

    if [ -f "$RESULT_FILE" ]; then
        if grep -q '"gtpa_at_1"' "$RESULT_FILE"; then
            GTPA=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d['gtpa_at_1']*100:.1f}%\")")
            DDR=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d['ddr']*100:.1f}%\")")
            DDF1=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d['ddf1']*100:.1f}%\")")
            IL=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d['avg_il']:.1f}\")")
            TIME=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d['elapsed_minutes']:.1f}m\")")

            echo "GTPA@1=$GTPA, DDR=$DDR, DDF1=$DDF1, IL=$IL"
            echo "$MODEL_SHORT,$GTPA,$DDR,$DDF1,$IL,$TIME" >> "$RESULTS_FILE"
        else
            ERROR=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(d.get('error', 'Unknown')[:50])")
            echo "Error: $ERROR"
            echo "$MODEL_SHORT,ERROR,,,," >> "$RESULTS_FILE"
        fi
    else
        echo "No result file"
        echo "$MODEL_SHORT,ERROR,,,," >> "$RESULTS_FILE"
    fi

    pkill -9 -f "EngineCore" 2>/dev/null
    sleep 2
done

echo ""
echo "=========================================="
echo "CATEGORY 2 FINAL SUMMARY"
echo "=========================================="
echo ""
column -t -s',' "$RESULTS_FILE"
echo ""
echo "Results saved to: $OUTPUT_DIR"
