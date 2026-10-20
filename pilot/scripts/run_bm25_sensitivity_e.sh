#!/usr/bin/env bash
# §9.33 V1 (POST HOC): BM25 sensitivity on confirmation set E. Lineage is read from 7689 (read-only); 7687/7688 unused.
set -uo pipefail
cd "$(dirname "$0")/../.."
O=pilot/cache/v3/framing/v1_bm25
PY=.venv/bin/python
mkdir -p $O
export CONFIRM_E_UNSEAL=1
$PY pilot/scripts/bm25_sensitivity_e.py run --index v2 --lineage-uri bolt://localhost:7689 > $O/run_v2.log 2>&1 &
P2=$!
$PY pilot/scripts/bm25_sensitivity_e.py run --index v3 --lineage-uri bolt://localhost:7689 > $O/run_v3.log 2>&1
S3=$?
wait $P2
S2=$?
[ $S3 -eq 0 ] && [ $S2 -eq 0 ] || { echo "V1_FAILED v3=$S3 v2=$S2"; exit 1; }
$PY pilot/scripts/bm25_sensitivity_e.py stats > $O/stats.log 2>&1 || { echo V1_FAILED stats; exit 1; }
echo V1_DONE
