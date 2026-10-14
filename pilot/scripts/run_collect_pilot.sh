#!/usr/bin/env bash
# Collection pilot: literature graph with and without the extra articles of the 50 pilot diseases,
# ranked over every dated phenopacket case with the frozen 1-hop config (and offtopic_w 1, since
# abstracts and gene-found articles rarely name the disease subtype in the title); *_split adds
# coordinated-phrase splitting and HPO-name relinking.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
V=pilot/cache/v2
C=$V/collect_pretest
[ -s $C/phrase2cui_collect.json ] || cp $V/phrase2cui_gemma.json $C/phrase2cui_collect.json
.venv-sci/bin/python pilot/scripts/normalize_v2.py --ie $C/ie_new.jsonl --table $C/phrase2cui_collect.json > $C/normalize.log 2>&1
for G in base pilot base_split pilot_split; do
  if [ ${G%_split} = base ]; then IE=$V/ie_gemma_full.jsonl; SRC=$V/sources; else IE=$V/ie_gemma_full.jsonl,$C/ie_new.jsonl; SRC=$C/merged; fi
  SPLIT=""; [ $G != ${G%_split} ] && SPLIT="--split-coord --hpo-names"
  $PY pilot/scripts/build_graph_v2.py $SPLIT --uri bolt://localhost:7688 --ie $IE --table $C/phrase2cui_collect.json \
    --sources $SRC --wipe --no-mentions --summary $C/graph_$G.json > $C/build_$G.log 2>&1
  $PY pilot/scripts/temporal_eval.py --snapshot 2026-09-01 --cfg gemma_1hop --tag lit_$G --out $C/eval_$G.json > $C/eval_$G.log 2>&1
  $PY pilot/scripts/temporal_eval.py --snapshot 2026-09-01 --cfg gemma_1hop --override '{"offtopic_w": 1.0}' \
    --tag lit_${G}_ot1 --out $C/eval_${G}_ot1.json > $C/eval_${G}_ot1.log 2>&1
done
echo done > $C/DONE_pilot
