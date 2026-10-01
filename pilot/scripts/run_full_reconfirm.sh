#!/usr/bin/env bash
# Full-corpus re-confirmation of MODEL and STRATEGY selection (prior runs were on the
# broken 33-doc extract). Sequential on the single 4090. Watch pilot/cache/reconfirm.log.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source .venv/bin/activate
export $(grep -E '^HF_TOKEN=' .env | xargs)
N=200
LOG=pilot/cache/reconfirm.log
: > "$LOG"

run_model() {  # model tag extra
  echo "========MODEL $2 ($1)========" | tee -a "$LOG"
  stdbuf -oL -eL python pilot/scripts/v148_model_panel.py --model "$1" --tag "$2" --n "$N" $3 >> "$LOG" 2>&1
  echo "----MODEL $2 exit=$?----" | tee -a "$LOG"
}

# MODEL PANEL (V4 prompt fixed; gemma-4-E4B already = v147_V4_pred full corpus)
run_model "Qwen/Qwen2.5-7B-Instruct"   "qwen25-7b"   ""
run_model "google/medgemma-4b-it"      "medgemma-4b" ""
run_model "google/gemma-4-12B-it"      "gemma4-12b"  "--load-4bit"
run_model "Qwen/Qwen2.5-14B-Instruct"  "qwen25-14b"  "--load-4bit"

# STRATEGY (gemma-4-E4B fixed; V4 already = v147_V4_pred full corpus)
echo "========STRATEGY v145 (S/C/V1/V2)========" | tee -a "$LOG"
stdbuf -oL -eL python pilot/scripts/v145_strategy_scope.py --n "$N" --backend hf --bs 8 >> "$LOG" 2>&1
echo "----v145 exit=$?----" | tee -a "$LOG"
echo "========STRATEGY v146 (V3/C3)========" | tee -a "$LOG"
stdbuf -oL -eL python pilot/scripts/v146_decomp.py --n "$N" --backend hf --bs 8 >> "$LOG" 2>&1
echo "----v146 exit=$?----" | tee -a "$LOG"

echo "ALL_DONE" >> "$LOG"
