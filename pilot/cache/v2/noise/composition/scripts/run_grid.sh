#!/usr/bin/env bash
# Composition grid: nominal ratios, precision-matched ratios (matched_ratios.json) and 20% (ratios_p20.json).
# Uses the comparator Neo4j at bolt://localhost:7688 (wiped by every build); finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."
run() {  # sib keeps ratios
  KEEPS=$2 RATIOS=$3 SIB=$1 O=pilot/cache/v2/noise/composition/sib$1 bash pilot/scripts/run_noise_sensitivity.sh
}
for SIB in 1 0 0.7; do run $SIB "1 0.3 0.15" "2.33 1"; done
run 1 1 2.86; run 1 0.3 2.63; run 1 0.15 2.58
run 0 1 1.82; run 0 0.3 2.02; run 0 0.15 2.08
run 1 1 1.15; run 1 0.3 1.08; run 1 0.15 1.06
run 0 1 0.82; run 0 0.3 0.91; run 0 0.15 0.93
run 0.7 0.3 4.95; run 0.7 0.15 4.77
run 1 0.3 6; run 1 0.15 5.59
run 0 0.3 4.05; run 0 0.15 4.03
