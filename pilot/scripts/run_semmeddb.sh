#!/usr/bin/env bash
# SemMedDB disease-finding graph with the same ranker (research design §9.32 P0-5).
# Each variant is loaded into the comparator instance (build_reference_graph.py --source semmeddb) and the
# nine development sets are ranked with (a) the frozen gemma 1-hop configuration and (b) the configuration
# tune_dev.py selects on this graph by the design D3 rule (same grid and start as HPOA, PrimeKG and MeSH
# co-occurrence, run_ref_dev.sh). offtopic_w has no effect (n_topic_docs = n_docs) and is fixed at 1.0 for
# the frozen run. The main variant (semantic_drop) is tuned; the sensitivity variants are ranked with the
# frozen configuration and with the main variant's selection.
#   semantic_drop  predicate orientation (MANIFESTATION_OF finding -> disease, the other two either way),
#                  SemMedDB generic concepts (novelty 0) left out
#   semantic_keep  same, generic concepts kept
#   subject_drop   disease as subject only for COEXISTS_WITH and ASSOCIATED_WITH
#
#   bash pilot/scripts/run_semmeddb.sh semantic_drop semantic_keep subject_drop
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
O=pilot/cache/v3/p0/semmeddb
URI=bolt://localhost:7688
MAIN=semantic_drop
BENCH=ddxplus-test,rarebench,pumch,phenopacket,symcat
mkdir -p $O
exec 9>pilot/cache/v2/devD/.neo4j_7688.lock
flock 9
read -r A M D U B < <($PY -c "import json;s=json.load(open('pilot/cache/v2/devD/frozen_config_D.json'))['gemma_1hop'];print(s['alpha'],s['mu'],s['decay'],s['unified_w'],s['support_beta'])")

dev() {   # tag cfg alpha mu decay unified_w support_beta offtopic_w
  [ -f $O/dev_$1_$2.json ] && return 0
  $PY pilot/scripts/graph_dx.py --uri $URI --bench $BENCH --ddx_n 3000 --alpha "$3" --mu "$4" --decay "$5" \
    --unified_w "$6" --support_beta "$7" --offtopic_w "$8" --out $O/dev_$1_$2.json > $O/dev_$1_$2.log 2>&1
}

for V in "$@"; do
  case $V in
    semantic_drop) OPT="--semmed-orient semantic --semmed-generic drop" ;;
    semantic_keep) OPT="--semmed-orient semantic --semmed-generic keep" ;;
    subject_drop)  OPT="--semmed-orient subject --semmed-generic drop" ;;
    *) echo "unknown variant $V"; exit 1 ;;
  esac
  # shellcheck disable=SC2086
  $PY pilot/scripts/build_reference_graph.py --source semmeddb --uri $URI $OPT --summary $O/graph_$V.json \
    > $O/build_$V.log 2>&1
  dev "$V" frozen "$A" "$M" "$D" "$U" "$B" 1.0
  if [ "$V" = $MAIN ] && [ ! -f $O/tune_$V.json ]; then
    $PY pilot/scripts/tune_dev.py --uri $URI --out $O/tune_$V.json > $O/tune_$V.log 2>&1
  fi
  [ -f $O/tune_$MAIN.json ] || { echo "no tune_$MAIN.json: run $MAIN first"; exit 1; }
  read -r TA TM TD TU TB TG < <($PY -c "import json;s=json.load(open('$O/tune_$MAIN.json'))['selected'];print(s['alpha'],s['mu'],s['decay'],s['unified_w'],s['support_beta'],s['offtopic_w'])")
  dev "$V" tuned "$TA" "$TM" "$TD" "$TU" "$TB" "$TG"
  echo "done $V"
done
