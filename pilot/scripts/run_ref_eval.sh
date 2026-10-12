#!/usr/bin/env bash
# Comparator knowledge sources through the same ranker: build -> validate-only selection -> test once.
#   bash pilot/scripts/run_ref_eval.sh hpoa primekg cooc
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
for src in "$@"; do
  O=pilot/cache/v2/eval_ref_$src; mkdir -p $O
  $PY pilot/scripts/build_reference_graph.py --source $src > $O/build.log 2>&1
  $PY pilot/scripts/tune_v2.py --uri bolt://localhost:7688 --out $O/tune.json > $O/tune.log 2>&1
  read -r A M D U B G < <($PY -c "import json;s=json.load(open('$O/tune.json'))['selected'];print(s['alpha'],s['mu'],s['decay'],s['unified_w'],s['support_beta'],s['offtopic_w'])")
  $PY pilot/scripts/graph_dx.py --uri bolt://localhost:7688 --bench ddxplus-test,rarebench,pumch,phenopacket,symcat \
    --ddx_n 3000 --alpha $A --mu $M --decay $D --unified_w $U --support_beta $B --offtopic_w $G --boot 1000 \
    --out $O/test.json > $O/test.log 2>&1
  echo "done $src"
done
