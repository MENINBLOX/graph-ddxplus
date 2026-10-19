#!/usr/bin/env bash
# Exploratory: the frozen scoring with unified-evidence diseases outside the k*4 window also scored on their
# findings (graph_dx.py --full_unified), on the frozen graph and the two UMLS dictionary graphs.
set -uo pipefail
cd /home/max/thesis
W=pilot/cache/v2/devD/window
C="--alpha 0.35 --mu 2.0 --decay 0.3 --unified_w 2.0 --support_beta 0.5"
BENCH="--bench ddxplus-test,rarebench,pumch,phenopacket,symcat --ddx_n 3000"
run() { .venv/bin/python pilot/scripts/graph_dx.py --uri $1 $BENCH $C --offtopic_w $2 $3 --out $W/$4.json > $W/$4.log 2>&1; echo "done $4 $?"; }
run bolt://localhost:7687 0.0 "" base_nm_check_ot0.0 &
run bolt://localhost:7687 0.0 --full_unified base_nm_uf_ot0.0 &
run bolt://localhost:7688 0.5 --full_unified v2umls_uf_ot0.5 &
run bolt://localhost:7689 0.5 --full_unified v2umls_ns_uf_ot0.5 &
wait
echo FULL_UNIFIED_DONE
