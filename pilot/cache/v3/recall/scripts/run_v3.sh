#!/bin/bash
# usage: run_v3.sh GRAPH MODE
cd /home/max/thesis
O=pilot/cache/v3/recall
SRC=pilot/cache/v2/sources,pilot/cache/v3/sources_new
COMMON="--sources $SRC --link full --topic-genes --no-self"
case $1 in
  smix)  ARGS="--ie pilot/cache/v3/llmd_v2.jsonl.gz,pilot/cache/v3/llmd_v3.jsonl.gz --table pilot/cache/v3/phrase2cui_v3.json --phrases pilot/cache/v2/ie_gemma_full.jsonl,pilot/cache/v3/ie_v3.jsonl --offtopic-w 0.5" ;;
  sllm)  ARGS="--ie pilot/cache/v2/ie_gemma_full.jsonl,pilot/cache/v3/ie_v3.jsonl --table pilot/cache/v3/phrase2cui_v3.json --offtopic-w 1.0" ;;
  sdict) ARGS="--ie pilot/cache/v2/dict_umls_v2.jsonl.gz,pilot/cache/v3/dict_umls_v3.jsonl.gz --table pilot/cache/v2/phrase2cui_gemma.json --offtopic-w 0.5" ;;
esac
# shellcheck disable=SC2086
/usr/bin/time -v .venv/bin/python pilot/scripts/recall_decomp.py $ARGS $COMMON --feature-cui $2 --out $O/decomp_$1_$2cui.json > $O/decomp_$1_$2cui.log 2>&1
echo "done $1 $2 $?"
