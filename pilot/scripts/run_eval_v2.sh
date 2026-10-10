#!/usr/bin/env bash
# End-to-end evaluation of the graph currently loaded in Neo4j (research design §5, D12):
#   1. select scoring configuration on DDXPlus validate only (coordinate search)
#   2. select IL configuration on DDXPlus validate only (benchmark space, budget 20)
#   3. run test sets once with the selected configuration (+ bootstrap CIs)
#   4. post-hoc graph-feature ablation on test (never used to re-select)
#   5. IL test (both action spaces, budgets 0/5/10/20)
#
#   TAG=qwen3_v21 bash pilot/scripts/run_eval_v2.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${TAG:?set TAG, e.g. qwen3_v21}"
PY=.venv/bin/python
OUT=pilot/cache/v2/eval_$TAG
mkdir -p "$OUT"
TEST=ddxplus-test,rarebench,pumch,phenopacket,symcat

$PY pilot/scripts/graph_dx.py --bench ddxplus-validate --ddx_n 3000 --alpha 0.35 --mu 2.0 --decay 0.5 --unified_w 1.0 \
  > "$OUT/validate_default.log" 2>&1
$PY pilot/scripts/tune_v2.py --out "$OUT/tune.json" > "$OUT/tune.log" 2>&1
read -r A M D U B G < <($PY -c "import json;s=json.load(open('$OUT/tune.json'))['selected'];print(s['alpha'],s['mu'],s['decay'],s['unified_w'],s['support_beta'],s['offtopic_w'])")
echo "selected alpha=$A mu=$M decay=$D unified_w=$U support_beta=$B offtopic_w=$G"

# IL selection on validate
best=""; bestkey="-1 -1"
for nw in 0 0.5 1.0; do for tp in 10 20 40; do
  f="$OUT/il_val_nw${nw}_top${tp}.json"
  $PY pilot/scripts/il_sim.py --bench ddxplus-validate --space benchmark --n 500 --budget 20 --neg_w $nw --top $tp \
    --alpha $A --mu $M --decay $D --unified_w $U --support_beta $B --offtopic_w $G --out "$f" > /dev/null 2>&1
  key=$($PY -c "import json;r=json.load(open('$f'))['result'];print(r['lineage@1'],r['lineage@3'])")
  echo "IL validate neg_w=$nw top=$tp -> $key"
  if $PY -c "import sys;a=list(map(float,'$key'.split()));b=list(map(float,'$bestkey'.split()));sys.exit(0 if a>b else 1)"; then
    best="$nw $tp"; bestkey="$key"; fi
done; done
read -r NW TP <<< "$best"
echo "IL selected neg_w=$NW top=$TP ($bestkey)"
$PY -c "import json;json.dump({'neg_w':$NW,'top':$TP,'validate':'$bestkey'},open('$OUT/il_selected.json','w'))"

# test (once)
$PY pilot/scripts/graph_dx.py --bench $TEST --ddx_n 3000 --alpha $A --mu $M --decay $D --unified_w $U --support_beta $B --offtopic_w $G --boot 1000 \
  --out "$OUT/test.json" > "$OUT/test.log" 2>&1

# post-hoc ablation: switch one graph feature relative to the selected configuration
abl() { $PY pilot/scripts/graph_dx.py --bench $TEST --ddx_n 3000 "$@" --out "$OUT/ablation_$name.json" > "$OUT/ablation_$name.log" 2>&1; }
if [ "$D" = "0.0" ]; then name=hier_on; abl --alpha $A --mu $M --decay 0.3 --unified_w $U --support_beta $B --offtopic_w $G
else name=no_hierarchy; abl --alpha $A --mu $M --decay 0.0 --unified_w $U --support_beta $B --offtopic_w $G; fi
name=no_unified;     abl --alpha $A --mu $M --decay $D --unified_w 0.0 --support_beta $B --offtopic_w $G
name=no_coverage;    abl --alpha $A --mu 0.0 --decay $D --unified_w $U --support_beta $B --offtopic_w $G
if [ "$B" != "0.0" ]; then name=no_support; abl --alpha $A --mu $M --decay $D --unified_w $U --support_beta 0.0 --offtopic_w $G; fi
if [ "$G" != "1.0" ]; then name=no_topic; abl --alpha $A --mu $M --decay $D --unified_w $U --support_beta $B --offtopic_w 1.0; fi
name=no_length_norm; abl --alpha 0.0 --mu $M --decay $D --unified_w $U --support_beta $B --offtopic_w $G

# IL test
for sp in benchmark open; do for b in ddxplus-test phenopacket; do for k in 0 5 10 20; do
  $PY pilot/scripts/il_sim.py --bench $b --space $sp --n 1000 --budget $k --neg_w $NW --top $TP \
    --alpha $A --mu $M --decay $D --unified_w $U --support_beta $B --offtopic_w $G --out "$OUT/il_test_${sp}_${b}_b$k.json" > /dev/null 2>&1
done; done; done
echo "done $TAG"
