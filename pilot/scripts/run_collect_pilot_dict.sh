#!/usr/bin/env bash
# Collection pilot, LLM-free ablation: the pilot diseases' extra articles and tables read by HPO
# dictionary matching (dict_ie.py) instead of the LLM; the existing corpus keeps its LLM items.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
V=pilot/cache/v2
C=$V/collect_pretest
$PY pilot/scripts/build_graph_v2.py --split-coord --hpo-names --uri bolt://localhost:7688 \
  --ie $V/ie_gemma_full.jsonl,$C/dict_new.jsonl,$C/dict_tables.jsonl --table $C/phrase2cui_collect.json \
  --sources $C/merged_tables --wipe --no-mentions --summary $C/graph_pilot_dict.json > $C/build_pilot_dict.log 2>&1
for OT in 0.0 1.0; do
  $PY pilot/scripts/temporal_eval.py --snapshot 2026-09-01 --cfg gemma_1hop --override "{\"offtopic_w\": $OT}" \
    --tag lit_pilot_dict_ot$OT --out $C/eval_pilot_dict_ot$OT.json > $C/eval_pilot_dict_ot$OT.log 2>&1
done
echo done > $C/DONE_dict
