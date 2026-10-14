#!/usr/bin/env bash
# Collection pilot, table step: IE on the pilot diseases' table documents, then the literature graph
# with paragraphs + tables, ranked like run_collect_pilot.sh. Runs after it (GPU and Neo4j not shared).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
V=pilot/cache/v2
C=$V/collect_pretest
bash pilot/scripts/run_vllm.sh pilot/scripts/ie_v2.py --sources $C/new_tables --out $C/ie_tables.jsonl > $C/ie_tables.log 2>&1
.venv-sci/bin/python pilot/scripts/normalize_v2.py --ie $C/ie_tables.jsonl --table $C/phrase2cui_collect.json > $C/normalize_tables.log 2>&1
for G in pilot_tables pilot_tables_split; do
  SPLIT=""; [ $G != ${G%_split} ] && SPLIT="--split-coord --hpo-names"
  $PY pilot/scripts/build_graph_v2.py $SPLIT --uri bolt://localhost:7688 --ie $V/ie_gemma_full.jsonl,$C/ie_new.jsonl,$C/ie_tables.jsonl \
    --table $C/phrase2cui_collect.json --sources $C/merged_tables --wipe --no-mentions --summary $C/graph_$G.json > $C/build_$G.log 2>&1
  $PY pilot/scripts/temporal_eval.py --snapshot 2026-09-01 --cfg gemma_1hop --tag lit_$G --out $C/eval_$G.json > $C/eval_$G.log 2>&1
  $PY pilot/scripts/temporal_eval.py --snapshot 2026-09-01 --cfg gemma_1hop --override '{"offtopic_w": 1.0}' \
    --tag lit_${G}_ot1 --out $C/eval_${G}_ot1.json > $C/eval_${G}_ot1.log 2>&1
done
echo done > $C/DONE_tables
