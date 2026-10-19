#!/usr/bin/env bash
# Sequential IE model panel on the V4 prompt (single 4090 -> one model per process).
# Each process loads, downloads if needed, runs n=200, saves preds, prints raw+adj F1, exits
# (clean VRAM between models). Run detached in tmux; watch pilot/cache/v148_panel.log.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source .venv/bin/activate
N="${1:-200}"
LOG=pilot/cache/v148_panel.log
: > "$LOG"

run() {  # model  tag  extra_args
  echo "======== $2 ($1) ========" | tee -a "$LOG"
  stdbuf -oL -eL python pilot/scripts/v148_model_panel.py --model "$1" --tag "$2" --n "$N" $3 >> "$LOG" 2>&1
  echo "---- $2 exit=$? ----" | tee -a "$LOG"
}

run "Qwen/Qwen2.5-7B-Instruct"   "qwen25-7b"   ""
run "google/gemma-4-12B-it"      "gemma4-12b"  "--load-4bit"
run "Qwen/Qwen2.5-14B-Instruct"  "qwen25-14b"  "--load-4bit"

echo "ALL_DONE" >> "$LOG"
