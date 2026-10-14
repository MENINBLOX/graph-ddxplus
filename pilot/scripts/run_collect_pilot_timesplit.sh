#!/usr/bin/env bash
# Collection pilot, leakage check: every article published after 2022 removed (base and pilot alike),
# so cases published from 2023 on cannot be described by their own or later reports. *_tg also counts
# titles naming the causal gene as on-topic; it is built on the full corpus too.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
V=pilot/cache/v2
C=$V/collect_pretest
build_eval() {   # name, max-year (0 = all), snapshot, extra builder flags...
  local G=$1 Y=$2 S=$3; shift 3
  if [ ${G%%_*} = base ]; then IE=$V/ie_gemma_full.jsonl; SRC=$V/sources;
  else IE=$V/ie_gemma_full.jsonl,$C/ie_new.jsonl,$C/ie_tables.jsonl; SRC=$C/merged_tables; fi
  $PY pilot/scripts/build_graph_v2.py "$@" --uri bolt://localhost:7688 --ie $IE --table $C/phrase2cui_collect.json \
    --sources $SRC --wipe --no-mentions --max-year $Y --summary $C/graph_$G.json > $C/build_$G.log 2>&1
  for OT in 0.0 1.0; do
    $PY pilot/scripts/temporal_eval.py --snapshot $S --cfg gemma_1hop --override "{\"offtopic_w\": $OT}" \
      --tag lit_${G}_ot$OT --out $C/eval_${G}_ot$OT.json > $C/eval_${G}_ot$OT.log 2>&1
  done
}
build_eval base_y2022 2022 2022-12-31
build_eval pilot_tables_split_y2022 2022 2022-12-31 --split-coord --hpo-names
build_eval pilot_tables_split_tg_y2022 2022 2022-12-31 --split-coord --hpo-names --topic-genes
build_eval pilot_tables_split_tg 0 2026-09-01 --split-coord --hpo-names --topic-genes
build_eval base_split_tg 0 2026-09-01 --split-coord --hpo-names --topic-genes
echo done > $C/DONE_timesplit
