#!/usr/bin/env bash
# Dictionary extraction (dict_ie.py) over corpus v3, following the collection: converts what has been
# downloaded (IE sources, and whole-body sources for the v3full variant), matches it, and repeats until
# the collection finished. Resumable; run_dict_umls_v3.sh completes a short last round.
set -uo pipefail
cd "$(dirname "$0")/../.."
V3=pilot/cache/v3
count() { .venv/bin/python pilot/scripts/jsonl_io.py count "$1"; }
while true; do
  done_fetch=$(grep -c COLLECT_V3_DONE $V3/collect.log 2>/dev/null || true)
  .venv/bin/python pilot/scripts/v3_sources.py --raw $V3/raw --out $V3/sources_new
  .venv/bin/python pilot/scripts/v3_sources.py --raw $V3/raw --out $V3/sources_full --full
  .venv/bin/python pilot/scripts/dict_ie.py --sources $V3/sources_new --out $V3/dict_v3.jsonl.gz --workers 8 --resume
  .venv/bin/python pilot/scripts/dict_ie.py --sources $V3/sources_full --out $V3/dict_v3full.jsonl.gz --workers 8 --resume
  echo "[$(date +%F' '%T)] sources=$(ls $V3/sources_new | grep -c '\.json$') matched=$(count $V3/dict_v3.jsonl.gz)" \
       "full=$(count $V3/dict_v3full.jsonl.gz) fetch_done=$done_fetch"
  [ "$done_fetch" -ge 1 ] && break
  sleep 600
done
echo DICT_V3_DONE
