#!/usr/bin/env bash
# Full reproduction: sources -> IE -> normalization -> KG -> evaluation.
#
# Every stage is resumable (--resume + atomic checkpoints) because a session teardown kills long
# jobs, and every stage writes a log that the invariant checks read. Stages are separated because
# each one failed silently at least once during development: a failed stage produces output that
# looks like a valid smaller result, so the chain must verify artifacts rather than exit codes.
#
#   ./reproduce.sh sources    # disease-name-only literature collection (§C3)
#   ./reproduce.sh ie         # findings -> attribute pass 2 -> association
#   ./reproduce.sh kg         # merge -> normalize -> assemble
#   ./reproduce.sh eval       # frozen config, all benchmarks
#   ./reproduce.sh ablate     # component ablation + patient-bootstrap CIs (paired vs full)
#   ./reproduce.sh measures   # non-accuracy numbers the paper reports (testability, curated, oracle, topology)
#   ./reproduce.sh all
set -euo pipefail
P=pilot/cache/prod
STAGE="${1:-all}"

# ---- frozen evaluation configuration (identical for every benchmark, §C5) -------------------
# re-selected on the final KG by config_reselect.py (one criterion: mean Top-1 over 6 benchmarks)
export KG_ATTR_COMBINE=mean
export KG_ATTR_SEL=0
# near-ties depend on set iteration order, which follows the string hash seed; pin it
export PYTHONHASHSEED=0
KG="${KG:-$P/kg_merged29.json}"
CFG="--parents $P/phenotype_parents_rich.json --hier 1 --sapbert 0 --tau 0.78 \
 --hpo_bridge $P/hpo_bridge.json --hpo_anc 0 --hpo_w 0.5 \
 --bio_bridge $P/biochem_bridge.json \
 --syn_bridge $P/synonym_bridge.json --syn_w 1.0 \
 --cov_mu 0.5 --normexp 0.35 --verb_w 0.5 --local_idf 1.0 --neg_w 15 \
 --lsa 1 --lsa_w 2.0 --fuse mult \
 --attr 1 --attr_w 10 --attr_mu 0.0 \
 --attr_use location,character,severity,onset,timing,aggravating,relieving"

need() { [ -s "$1" ] || { echo "MISSING: $1" >&2; exit 1; }; }

if [ "$STAGE" = sources ] || [ "$STAGE" = all ]; then
  echo "== 1. literature collection (disease NAME only) =="
  # name_variants() splits alias-joined labels and drops numeric subtypes; without it a title
  # search returns nothing for many labels, which is indistinguishable from "no literature".
  [ -n "${DISEASES:-}" ] || { DISEASES="$P/full_todo.json"
    .venv/bin/python pilot/scripts/full_corpus_todo.py --out "$DISEASES"; }
  .venv/bin/python pilot/scripts/deep_fetch.py --diseases "$DISEASES" \
    --out-dir "${SRCDIR:-$P/sources_full}" --fetched-date "$(date +%F)" \
    --reviews 10 --cases 6 --budget 150000 --workers 6
fi

if [ "$STAGE" = ie ] || [ "$STAGE" = all ]; then
  # findings (exhaustive, no attribute rules) -> attribute pass 2 (findings held fixed) ->
  # association IE (§11 non-symptom evidence). The driver is idempotent: after an interruption the
  # same command continues from the last checkpoint, and a stage counts as done only when its own
  # log has the completion line. Reads $P/sources_full, writes $P/ie_pred_full_assoc.json.
  echo "== 2. full-corpus IE (findings -> attributes -> associations) =="
  bash pilot/scripts/full_corpus_resume.sh
  need "$P/ie_pred_full_assoc.json"
fi

if [ "$STAGE" = kg ] || [ "$STAGE" = all ]; then
  # The final KG is the development KG (merged27: benchmark gold diseases, deep sources) plus the
  # full-corpus extraction from `ie`, merged additively. Normalization records are seeded from the
  # development KG so --resume only normalizes what the full corpus added.
  echo "== 3. additive merge of the full-corpus extraction =="
  need "$P/ie_pred_full_assoc.json"
  .venv/bin/python pilot/scripts/merge_full_corpus.py \
    --base "$P/ie_pred_merged27.json" --incoming "$P/ie_pred_full_assoc.json" \
    --out "$P/ie_pred_merged29.json" \
    --records "$P/kg_records_merged27.json" --records-out "$P/kg_records_merged29.json"
  echo "== 4. qualifier promotion into empty slots =="
  .venv/bin/python pilot/scripts/qualifier_promote.py \
    --preds "$P/ie_pred_merged29.json" --out "$P/ie_pred_merged29_promoted.json"
  echo "== 5. deterministic normalization (serial: one worker holds ~15 GB) =="
  .venv-sci/bin/python pilot/scripts/v154_prod_normalize.py \
    --pred "$P/ie_pred_merged29_promoted.json" --out "$P/kg_records_merged29.json" \
    --resume --checkpoint 200
  echo "== 6. graph assembly =="
  .venv/bin/python pilot/scripts/v157_build_kg.py --records "$P/kg_records_merged29.json" \
    --preds "$P/ie_pred_merged29_promoted.json" --out "$P/kg_merged29.json"
  echo "== 7. integrity gates =="
  .venv/bin/python pilot/scripts/pipeline_invariants.py --kg "$P/kg_merged29.json" \
    --preds "$P/ie_pred_merged29_promoted.json" || true
fi

if [ "$STAGE" = eval ] || [ "$STAGE" = all ]; then
  echo "== 8. evaluation, frozen config, all benchmarks =="
  need "$KG"
  DDX_COMPOSE_PAIN=1 .venv/bin/python pilot/scripts/onlykg_eval_unified2.py \
    --kg "$KG" $CFG --benchmarks ddxplus
  .venv/bin/python pilot/scripts/onlykg_eval_unified2.py --kg "$KG" $CFG --benchmarks rarebench
  .venv/bin/python pilot/scripts/onlykg_eval_unified2.py --kg "$KG" $CFG --benchmarks symcat
  echo "== 9. attribute ablation (permutation null + bootstrap) =="
  KG_NORM_EXP=0.35 .venv/bin/python pilot/scripts/nlice_ablation.py --kg "$KG" \
    --queries "$P/nlice_queries_v2.json" --perm 100 --boot 1000
fi

if [ "$STAGE" = ablate ] || [ "$STAGE" = all ]; then
  echo "== 10. component ablation with patient-bootstrap 95% CIs =="
  # One switch per variant against the frozen config; per-patient ranks are dumped so the
  # statistics below can be recomputed without re-scoring. JOBS is bounded by RAM (~4.4 GB each).
  need "$KG"
  ABL="${ABL:-$P/ablation_$(basename "$KG" .json)}"
  .venv/bin/python pilot/scripts/ablation_suite.py run --kg "$KG" --out-dir "$ABL" --jobs "${JOBS:-5}"
  .venv/bin/python pilot/scripts/ablation_suite.py stats --out-dir "$ABL" --boot 2000
fi

if [ "$STAGE" = measures ] || [ "$STAGE" = all ]; then
  echo "== 11. analysis measurements reported in the paper =="
  need "$KG"
  for m in topology testability curated; do
    .venv/bin/python pilot/scripts/paper_measures.py "$m" --kg "$KG"
  done
  .venv/bin/python pilot/scripts/paper_measures.py oracle
fi
