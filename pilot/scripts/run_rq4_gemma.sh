#!/usr/bin/env bash
# RQ4: replace the IE model (Qwen3-4B -> gemma-4-E4B) with everything else fixed, then restore Qwen3 graph.
#   1. build gemma graph (same corpus, normalizer, build rules)
#   2. reviewer-free provenance check (curated agreement)
#   3. test with the Qwen3-selected configuration (IE effect alone), and the full validate-only
#      selection + test chain for gemma (each pipeline tuned on validate)
#   4. rebuild the Qwen3 graph
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
V=pilot/cache/v2
$PY pilot/scripts/build_graph_v2.py --ie $V/ie_gemma_full.jsonl --table $V/phrase2cui_gemma.json \
  --sources $V/sources --wipe --summary $V/graph_summary_gemma_v21.json > $V/build_gemma_v21.log 2>&1
$PY pilot/scripts/curated_agreement.py --out $V/curated_agreement_gemma_v21.json > /dev/null
mkdir -p $V/eval_gemma_v21
read -r A M D U B G < <($PY -c "import json;s=json.load(open('$V/eval_qwen3_v21/tune.json'))['selected'];print(s['alpha'],s['mu'],s['decay'],s['unified_w'],s['support_beta'],s['offtopic_w'])")
$PY pilot/scripts/graph_dx.py --bench ddxplus-validate,ddxplus-test,rarebench,pumch,phenopacket,symcat --ddx_n 3000 \
  --alpha $A --mu $M --decay $D --unified_w $U --support_beta $B --offtopic_w $G --boot 1000 \
  --out $V/eval_gemma_v21/test_qwen_config.json > $V/eval_gemma_v21/test_qwen_config.log 2>&1
TAG=gemma_v21 bash pilot/scripts/run_eval_v2.sh
$PY pilot/scripts/build_graph_v2.py --ie $V/ie_qwen3.jsonl,$V/ie_qwen3_delta.jsonl --table $V/phrase2cui_qwen3.json \
  --sources $V/sources --wipe --summary $V/graph_summary_qwen3_v21.json > $V/build_qwen3_v21_restore.log 2>&1
echo "rq4 done"
