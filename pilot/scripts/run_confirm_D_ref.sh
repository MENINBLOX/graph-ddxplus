#!/usr/bin/env bash
# Design D confirmation for comparator knowledge sources (run after run_confirm_D.sh, once).
# Rebuilds each comparator graph in the comparator instance (7688) and evaluates its frozen 1-hop
# and PPR configurations on the sealed confirmation sets.
set -euo pipefail
cd "$(dirname "$0")/../.."
F=pilot/cache/v2/devD/frozen_config_D.json
O=pilot/cache/v2/confirmD
[ -e "$O/DONE" ] || { echo "run run_confirm_D.sh first"; exit 1; }
[ -e "$O/DONE_ref" ] && { echo "comparator confirmation already run"; exit 1; }
export CONFIRM_UNSEAL=1
PY=.venv/bin/python
for src in hpoa primekg cooc; do
  $PY pilot/scripts/build_reference_graph.py --source $src > $O/ref_${src}_build.log 2>&1
  read -r A M D U B G < <($PY -c "import json;s=json.load(open('$F'))['comparators']['$src']['onehop'];print(s['alpha'],s['mu'],s['decay'],s['unified_w'],s['support_beta'],s['offtopic_w'])")
  $PY pilot/scripts/graph_dx.py --uri bolt://localhost:7688 --bench confirm --alpha $A --mu $M --decay $D --unified_w $U \
    --support_beta $B --offtopic_w $G --boot 1000 --out $O/ref_${src}_1hop.json > $O/ref_${src}_1hop.log 2>&1
  if [ "$src" != "cooc" ]; then
    $PY -c "import json;json.dump({'selected':json.load(open('$F'))['comparators']['$src']['ppr']},open('$O/ref_${src}_ppr_cfg.json','w'))"
    $PY pilot/scripts/ppr_dx.py test --bench confirm --uri bolt://localhost:7688 --graph pilot/cache/v2/devD/ref_$src/graph.npz \
      --cfg $O/ref_${src}_ppr_cfg.json --out $O/ref_${src}_ppr.json > $O/ref_${src}_ppr.log 2>&1
  fi
done
date -Iseconds > $O/DONE_ref
echo "comparator confirmation done"
