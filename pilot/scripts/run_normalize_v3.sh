#!/usr/bin/env bash
# Links the v3 LLM phrases (ie_v3.jsonl) into phrase2cui_v3.json, a copy of the v2 table extended with
# the same linker, so v3llm/v3union see their phrases linked like the v2 ones. The linker holds about
# 15 GB, so each pass waits until that much memory is available. Incremental; rerun after the IE.
# phrase2cui_v3.covers holds how many ie_v3.jsonl lines the last successful pass covered at least;
# run_graph_variants.sh builds the v3 LLM variants only when it equals the current line count.
set -uo pipefail
cd "$(dirname "$0")/../.."
V3=pilot/cache/v3
T=$V3/phrase2cui_v3.json
[ -s $T ] || cp pilot/cache/v2/phrase2cui_gemma.json $T
until [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 17 ]; do sleep 60; done
n=$(wc -l < $V3/ie_v3.jsonl)
if .venv-sci/bin/python pilot/scripts/normalize_v2.py --ie $V3/ie_v3.jsonl --table $T; then
  echo "$n" > $V3/phrase2cui_v3.covers
  echo "NORMALIZE_V3 $(date +%F' '%T) $n diseases"
else
  echo "NORMALIZE_V3_FAILED $(date +%F' '%T)"
  exit 1
fi
