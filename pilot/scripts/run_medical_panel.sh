#!/usr/bin/env bash
# Medical/alt-model round of the IE panel, V4 prompt fixed, all 7-8B -> bf16 (no 4-bit
# confound). Sequential on the single 4090. Watch pilot/cache/v148_medpanel.log.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source .venv/bin/activate
N="${1:-200}"
LOG=pilot/cache/v148_medpanel.log
: > "$LOG"

run() {  # model  tag
  echo "======== $2 ($1) ========" | tee -a "$LOG"
  stdbuf -oL -eL python pilot/scripts/v148_model_panel.py --model "$1" --tag "$2" --n "$N" >> "$LOG" 2>&1
  echo "---- $2 exit=$? ----" | tee -a "$LOG"
}

run "microsoft/Phi-3.5-mini-instruct"     "phi35-mini"
run "m42-health/Llama3-Med42-8B"          "med42-8b"
run "aaditya/Llama3-OpenBioLLM-8B"        "openbio-8b"
run "BioMistral/BioMistral-7B"            "biomistral-7b"

echo "ALL_DONE" >> "$LOG"
