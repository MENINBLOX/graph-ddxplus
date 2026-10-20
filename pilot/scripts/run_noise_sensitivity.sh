#!/usr/bin/env bash
# Ranker sensitivity to wrong edges (and, with KEEPS, to missing edges): HPOA graph (v2024-12-12) with
# wrong edges added at several precision levels, ranked with the frozen HPOA 1-hop config.
# SIB sets the wrong-edge composition (build_reference_graph.py --sib-share); O the output directory.
#   KEEPS="1 0.3 0.15" RATIOS=2.33 SIB=1 O=pilot/cache/v2/noise/composition/sib1 run_noise_sensitivity.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
O=${O:-pilot/cache/v2/noise}
SIB=${SIB:-0.7}
T=v2024-12-12
mkdir -p $O
for K in ${KEEPS:-1}; do
for R in ${RATIOS:-0.43 1 2.33}; do
  for S in ${SEEDS:-0}; do
    G=r${R}_s$S
    [ "$K" != 1 ] && G=k${K}_$G
    [ -s $O/$G.json ] && continue
    $PY pilot/scripts/build_reference_graph.py --source hpoa --hpoa data/external_kg/hpoa_history/$T.hpoa \
      --keep $K --noise $R --sib-share $SIB --seed $S --summary $O/${G}_summary.json > $O/${G}_build.log 2>&1
    $PY pilot/scripts/temporal_eval.py --snapshot ${T:1} --cfg hpoa --tag noise_$G --out $O/$G.json > $O/$G.log 2>&1
  done
done
done
echo done > $O/DONE
