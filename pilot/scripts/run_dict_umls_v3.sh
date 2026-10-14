#!/usr/bin/env bash
# Dictionary extraction over the finished corpus v3: completes the HPO-name pass of run_dict_v3.sh
# where its last round fell short, then the UMLS-vocabulary pass (dict_ie.py --vocab umls). Every
# output must hold one line per source file. Resumable.
set -uo pipefail
cd "$(dirname "$0")/../.."
V3=pilot/cache/v3
until grep -q DICT_V3_DONE $V3/dict_v3.log 2>/dev/null; do sleep 300; done
fail=0
for spec in "hpo sources_new dict_v3" "hpo sources_full dict_v3full" \
            "umls sources_new dict_umls_v3" "umls sources_full dict_umls_v3full"; do
  read -r voc src out <<< "$spec"
  n=$(ls $V3/$src | grep -c '\.json$')
  [ -f $V3/$out.jsonl ] && [ "$(wc -l < $V3/$out.jsonl)" -ge "$n" ] && continue
  .venv/bin/python pilot/scripts/dict_ie.py --vocab $voc --sources $V3/$src --out $V3/$out.jsonl --workers 8 --resume
  got=$(wc -l < $V3/$out.jsonl)
  echo "$out: $got lines for $n sources"
  [ "$got" -ge "$n" ] || fail=1
done
[ $fail = 0 ] && echo DICT_UMLS_V3_DONE || echo DICT_UMLS_V3_INCOMPLETE
