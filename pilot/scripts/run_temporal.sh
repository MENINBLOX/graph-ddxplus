#!/usr/bin/env bash
# Research design T: rebuild time-sliced knowledge graphs in the comparator instance (7688) and rank
# every dated phenopacket case with the frozen design D 1-hop configurations.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
V=pilot/cache/v2
O=$V/temporal
mkdir -p $O
for Y in ${YEARS:-2019 2020 2021 2022 2023 2024 2025}; do
  [ -s $O/lit_$Y.json ] && continue
  $PY pilot/scripts/build_graph_v2.py --uri bolt://localhost:7688 --ie $V/ie_gemma_full.jsonl --table $V/phrase2cui_gemma.json \
    --sources $V/sources --wipe --no-mentions --max-year $Y --summary $O/lit_${Y}_summary.json > $O/lit_${Y}_build.log 2>&1
  $PY pilot/scripts/temporal_eval.py --snapshot $Y-12-31 --cfg gemma_1hop --tag lit_$Y --out $O/lit_$Y.json > $O/lit_$Y.log 2>&1
done
for T in ${HPOA_TAGS:-v2022-02-14 v2022-12-15 v2024-01-11 v2024-12-12 v2025-11-24 v2026-09-01}; do
  [ -s $O/hpoa_$T.json ] && continue
  $PY pilot/scripts/build_reference_graph.py --source hpoa --hpoa data/external_kg/hpoa_history/$T.hpoa \
    --summary $O/hpoa_${T}_summary.json > $O/hpoa_${T}_build.log 2>&1
  $PY pilot/scripts/temporal_eval.py --snapshot ${T:1} --cfg hpoa --tag hpoa_$T --out $O/hpoa_$T.json > $O/hpoa_$T.log 2>&1
done
echo done > $O/DONE
