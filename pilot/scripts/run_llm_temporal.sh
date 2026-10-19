#!/usr/bin/env bash
# Research design T, LLM axis: zero-shot diagnosis of every dated phenopacket case per local model.
set -uo pipefail
cd "$(dirname "$0")/../.."
O=pilot/cache/v2/temporal
while read -r tag model flags; do
  [ -z "$tag" ] && continue
  [ -s $O/llm_${tag}.json ] && continue
  bash pilot/scripts/run_vllm.sh pilot/scripts/llm_temporal.py generate --model $model $flags \
    --out $O/llm_${tag}_raw.json > $O/llm_${tag}_gen.log 2>&1 || { echo "$tag generate failed"; continue; }
  .venv/bin/python pilot/scripts/llm_temporal.py score --raw $O/llm_${tag}_raw.json --out $O/llm_${tag}.json > $O/llm_${tag}.log 2>&1
  tail -n 1 $O/llm_${tag}.log
done <<'LIST'
qwen25_7b Qwen/Qwen2.5-7B-Instruct
qwen3_4b Qwen/Qwen3-4B-Instruct-2507
phi35 microsoft/Phi-3.5-mini-instruct
med42_8b m42-health/Llama3-Med42-8B
medgemma4b google/medgemma-4b-it --eager
gemma4_e4b google/gemma-4-E4B-it --eager
gemma4_12b google/gemma-4-12B-it --eager --quant fp8
qwen25_14b Qwen/Qwen2.5-14B-Instruct --quant fp8
LIST
echo done > $O/DONE_llm
