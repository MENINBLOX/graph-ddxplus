#!/usr/bin/env bash
# Design D final confirmation run (research design D2, D7). Run ONCE, after the freeze commit.
#   bash pilot/scripts/run_confirm_D.sh
# Requires: pilot/cache/v2/devD/frozen_config_D.json committed, clean tracked tree, gemma graph on 7687.
set -euo pipefail
cd "$(dirname "$0")/../.."
F=pilot/cache/v2/devD/frozen_config_D.json
git ls-files --error-unmatch "$F" > /dev/null || { echo "frozen config not committed"; exit 1; }
git diff --quiet HEAD -- pilot/scripts "$F" || { echo "uncommitted changes in scripts/config"; exit 1; }
O=pilot/cache/v2/confirmD
[ -e "$O/DONE" ] && { echo "confirmation already run ($O/DONE)"; exit 1; }
mkdir -p $O
git rev-parse HEAD > $O/freeze_commit.txt
export CONFIRM_UNSEAL=1
PY=.venv/bin/python
cfg() { $PY -c "import json;print(json.load(open('$F'))$1)"; }
# 1. main system: gemma graph, 1-hop ranker (main instance)
read -r A M D U B G < <($PY -c "import json;s=json.load(open('$F'))['gemma_1hop'];print(s['alpha'],s['mu'],s['decay'],s['unified_w'],s['support_beta'],s['offtopic_w'])")
$PY pilot/scripts/graph_dx.py --uri bolt://localhost:7687 --bench confirm --alpha $A --mu $M --decay $D --unified_w $U \
  --support_beta $B --offtopic_w $G --boot 1000 --out $O/gemma_1hop.json > $O/gemma_1hop.log 2>&1
# 2. BM25 and MedCPT (same corpus), 3. graph+BM25 fusion
$PY pilot/scripts/retrieval_baseline.py test --bench confirm --cfg pilot/cache/v2/devD/tune_bm25.json --out $O/bm25.json > $O/bm25.log 2>&1
cp pilot/cache/v2/devD/fuse_tune_gemma.json $O/fuse_tune.json
$PY pilot/scripts/retrieval_baseline.py fuse_test --bench confirm --graph_uri bolt://localhost:7687 \
  --graph_cfg pilot/cache/v2/devD/tune_gemma.json --cfg pilot/cache/v2/devD/tune_bm25.json --out $O/fuse_test.json > $O/fuse.log 2>&1
# 4. PPR (HippoRAG-style) on the gemma graph
$PY pilot/scripts/ppr_dx.py test --bench confirm --uri bolt://localhost:7687 --graph pilot/cache/v2/devD/graph_gemma.npz \
  --cfg pilot/cache/v2/devD/tune_ppr_gemma.json --out $O/ppr_gemma.json > $O/ppr.log 2>&1
# 5. paired comparisons (T1, T4, E-H1)
for k in 10 1; do
  $PY pilot/scripts/paired_stats.py $O/bm25.json $O/gemma_1hop.json --k $k --label "T1 BM25 vs graph" --out $O/paired_T1_k$k.json
  $PY pilot/scripts/paired_stats.py $O/gemma_1hop.json $O/fuse_test.json --k $k --label "T4 graph vs fusion" --out $O/paired_T4g_k$k.json
  $PY pilot/scripts/paired_stats.py $O/bm25.json $O/fuse_test.json --k $k --label "T4 BM25 vs fusion" --out $O/paired_T4b_k$k.json
  $PY pilot/scripts/paired_stats.py $O/gemma_1hop.json $O/ppr_gemma.json --k $k --label "E-H1 1-hop vs PPR" --out $O/paired_EH1_k$k.json
done > $O/paired.txt
date -Iseconds > $O/DONE
echo "confirmation done"
