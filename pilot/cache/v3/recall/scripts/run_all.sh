#!/usr/bin/env bash
# One corpus at a time: D regression, S-mix, S-llm, S-dict, S-mix without --phrases.
set -uo pipefail
cd /home/max/thesis
PY=.venv/bin/python
SP=pilot/cache/v3/recall/supp
V=pilot/cache/v2
V3=pilot/cache/v3
R=$V3/recall
SRC3="$V/sources,$V3/sources_new"
V3FLAGS="--sources $SRC3 --link full --topic-genes --no-self"
LLM3="$V/ie_gemma_full.jsonl,$V3/ie_v3.jsonl"
run() {  # name log args...
  local name=$1 log=$2; shift 2
  echo "start $name $(date +%T)"
  /usr/bin/time -v $PY pilot/scripts/recall_decomp.py "$@" > "$log" 2>&1
  echo "end $name rc=$? $(date +%T) $(grep -E 'Maximum resident' "$log" | tr -s ' ')"
}
run D $SP/regress_D2.log --out $SP/regress_D2/decomp.json
run smix $R/decomp_smix.log $V3FLAGS --ie $V3/llmd_v2.jsonl.gz,$V3/llmd_v3.jsonl.gz --table $V3/phrase2cui_v3.json \
  --phrases $LLM3 --out $R/decomp_smix.json
run sllm $R/decomp_sllm.log $V3FLAGS --ie $LLM3 --table $V3/phrase2cui_v3.json --out $R/decomp_sllm.json
run sdict $R/decomp_sdict.log $V3FLAGS --ie $V/dict_umls_v2.jsonl.gz,$V3/dict_umls_v3.jsonl.gz --table $V/phrase2cui_gemma.json \
  --out $R/decomp_sdict.json
run smix_nophr $SP/smix_nophrases.log $V3FLAGS --ie $V3/llmd_v2.jsonl.gz,$V3/llmd_v3.jsonl.gz --table $V3/phrase2cui_v3.json \
  --out $SP/smix_nophrases/decomp.json
echo ALLDONE
