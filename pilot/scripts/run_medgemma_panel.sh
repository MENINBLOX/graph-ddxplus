#!/usr/bin/env bash
# medgemma + medical model round (token-gated). V4 prompt fixed, bf16. Sequential on the
# single 4090. Sources HF_TOKEN from .env for gated/xet downloads. Watch v148_medgemma.log.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source .venv/bin/activate
export $(grep -E '^HF_TOKEN=' .env | xargs)
N="${1:-200}"
LOG=pilot/cache/v148_medgemma.log
: > "$LOG"

run() {  # model  tag  extra
  echo "======== $2 ($1) ========" | tee -a "$LOG"
  stdbuf -oL -eL python pilot/scripts/v148_model_panel.py --model "$1" --tag "$2" --n "$N" $3 >> "$LOG" 2>&1
  echo "---- $2 exit=$? ----" | tee -a "$LOG"
}

run "google/medgemma-4b-it"          "medgemma-4b"   ""
run "m42-health/Llama3-Med42-8B"     "med42-8b"      ""
run "aaditya/Llama3-OpenBioLLM-8B"   "openbio-8b"    ""
run "BioMistral/BioMistral-7B"       "biomistral-7b" ""

echo "ALL_DONE" >> "$LOG"
