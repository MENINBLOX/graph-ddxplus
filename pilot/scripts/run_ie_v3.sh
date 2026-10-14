#!/usr/bin/env bash
# Pipelined LLM IE over corpus v3 (collect_v3.py): converts what has been downloaded, extracts it,
# and repeats until the collection finished and every disease is extracted. Resumable.
set -uo pipefail
cd "$(dirname "$0")/../.."
V3=pilot/cache/v3
OUT=$V3/ie_v3.jsonl
while true; do
  done_fetch=$(grep -c COLLECT_V3_DONE $V3/collect.log 2>/dev/null || true)
  .venv/bin/python pilot/scripts/v3_sources.py --raw $V3/raw --out $V3/sources_new
  bash pilot/scripts/run_vllm.sh pilot/scripts/ie_v2.py --sources $V3/sources_new --out $OUT \
    --no-eager --batch 512 --limit-diseases 2000
  n_src=$(ls $V3/sources_new | grep -c '\.json$'); n_out=$(wc -l < $OUT)
  echo "[$(date +%F' '%T)] sources=$n_src extracted=$n_out fetch_done=$done_fetch"
  if [ "$done_fetch" -ge 1 ] && [ "$n_out" -ge "$n_src" ]; then break; fi
  [ "$n_out" -ge "$n_src" ] && sleep 300
done
echo IE_V3_DONE
