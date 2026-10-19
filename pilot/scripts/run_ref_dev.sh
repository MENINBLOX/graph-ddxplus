#!/usr/bin/env bash
# Design D3/D10: comparator knowledge sources re-selected on the development sets (same rule as ours),
# plus PPR on curated graphs (E-H4, restart-only search).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
for src in "$@"; do
  O=pilot/cache/v2/devD/ref_$src; mkdir -p $O
  $PY pilot/scripts/build_reference_graph.py --source $src > $O/build.log 2>&1
  $PY pilot/scripts/tune_dev.py --uri bolt://localhost:7688 --out $O/tune.json > $O/tune.log 2>&1
  read -r A M D U B G < <($PY -c "import json;s=json.load(open('$O/tune.json'))['selected'];print(s['alpha'],s['mu'],s['decay'],s['unified_w'],s['support_beta'],s['offtopic_w'])")
  $PY pilot/scripts/graph_dx.py --uri bolt://localhost:7688 --bench ddxplus-test,rarebench,pumch,phenopacket,symcat --ddx_n 3000 \
    --alpha $A --mu $M --decay $D --unified_w $U --support_beta $B --offtopic_w $G --out $O/dev.json > $O/dev.log 2>&1
  if [ "$src" != "cooc" ]; then
    $PY pilot/scripts/ppr_dx.py export --uri bolt://localhost:7688 --out $O/graph.npz > $O/export.log 2>&1
    $PY pilot/scripts/ppr_dx.py tune --uri bolt://localhost:7688 --graph $O/graph.npz --only restart --out $O/tune_ppr.json > $O/tune_ppr.log 2>&1
    $PY pilot/scripts/ppr_dx.py test --bench dev --uri bolt://localhost:7688 --graph $O/graph.npz --cfg $O/tune_ppr.json --boot 0 --out $O/dev_ppr.json > $O/dev_ppr.log 2>&1
  fi
  echo "done $src"
done
