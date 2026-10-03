#!/usr/bin/env bash
# Pipelined IE: extracts every disease already fetched, then waits for more, until the fetch
# finished and nothing is left. Resumable: finished diseases are skipped by ie_v2.py.
#   MODEL=Qwen/Qwen3-4B-Instruct-2507 OUT=pilot/cache/v2/ie_qwen3.jsonl bash run_ie_v2.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
OUT="${OUT:-pilot/cache/v2/ie_qwen3.jsonl}"
while true; do
  fetch_done=$(grep -c FETCH_V2_DONE pilot/cache/v2/fetch.log 2>/dev/null || true)
  bash pilot/scripts/run_vllm.sh pilot/scripts/ie_v2.py --sources pilot/cache/v2/sources \
    --out "$OUT" --model "$MODEL" --limit-diseases 3000
  n_src=$(ls pilot/cache/v2/sources | wc -l); n_out=$(wc -l < "$OUT")
  echo "[$(date +%F' '%T)] sources=$n_src extracted=$n_out fetch_done=$fetch_done"
  if [ "$fetch_done" -ge 1 ] && [ "$n_out" -ge "$n_src" ]; then break; fi
  sleep 120
done
echo IE_V2_DONE
