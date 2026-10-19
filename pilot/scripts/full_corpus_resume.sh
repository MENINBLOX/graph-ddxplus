#!/usr/bin/env bash
# Idempotent driver for the full-corpus extraction chain.
#
# The host has rebooted mid-run more than once, and each time the chain had to be reassembled by
# hand. This script decides what to run from what is already on disk, so after any interruption
# the same single command continues from the last checkpoint:
#
#   nohup bash pilot/scripts/full_corpus_resume.sh > pilot/cache/prod/full_resume.log 2>&1 &
#
# A stage counts as finished only when its completion line is in its own log -- an output file
# existing is not enough, because every stage checkpoints a partial file that looks complete.
set -uo pipefail
P=pilot/cache/prod
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.70}"
CHAIN="$P/bench_chain.log"

stage_done() { grep -q "$2" "$1" 2>/dev/null; }

run_until_done() {        # retry a resumable stage until its log shows completion
  local log="$1" marker="$2"; shift 2
  local tries=0
  until stage_done "$log" "$marker"; do
    tries=$((tries + 1))
    [ "$tries" -gt 5 ] && { echo "[$(date +%F' '%T)] giving up after 5 attempts: $log"; return 1; }
    echo "[$(date +%F' '%T)] attempt $tries -> $*"
    "$@" >> "$log" 2>&1
    sleep 60                  # let the previous vLLM engine release GPU memory
  done
  echo "[$(date +%F' '%T)] done: $log"
}

# 1. findings IE
run_until_done "$P/full_ie.log" "^diseases=" \
  .venv/bin/python pilot/scripts/v153_production_ie.py --source prod --src-dir "$P/sources_full" \
  --n 100000 --backend vllm --bs 24 --win 8000 --max-new 5120 --chunk 500 --resume \
  --out "$P/ie_pred_full_deep.json" || exit 1

# 2. attribute pass 2 (findings held fixed)
run_until_done "$P/full_p2_resume.log" "^findings unchanged" \
  .venv/bin/python pilot/scripts/attr_enrich_pass2.py --preds "$P/ie_pred_full_deep.json" \
  --out "$P/ie_pred_full_p2.json" --chunk 500 --resume || exit 1

# 3. association IE
run_until_done "$P/full_assoc.log" "^associations " \
  .venv/bin/python pilot/scripts/assoc_ie.py --preds "$P/ie_pred_full_p2.json" \
  --out "$P/ie_pred_full_assoc.json" --chunk 500 --resume || exit 1

stage_done "$CHAIN" FULL_EXTRACTION_DONE || echo FULL_EXTRACTION_DONE >> "$CHAIN"
echo "[$(date +%F' '%T)] full-corpus extraction complete"
