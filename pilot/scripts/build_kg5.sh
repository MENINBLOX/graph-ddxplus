#!/usr/bin/env bash
# Rebuild the KG from the pass-2 attribute-enriched IE and re-measure everything that the
# seven silent-loss defects had made unmeasurable. Run from the repo root.
#
# Chain (each stage was a place where attribute signal used to disappear without an error):
#   ie_pred_merged4_p2.json  pass-1 findings + pass-2 attributes   (attr_enrich_pass2.py)
#     -> kg_records_merged5   deterministic CUI normalization      (v154, --resume)
#     -> kg_merged5           graph assembly + documentation depth (v157 --preds)
#     -> kg_merged5_semio     semiology attribute enrichment       (semiology_build_edges.py)
#     -> invariants / ablation / benchmarks
set -euo pipefail

PRED=pilot/cache/prod/ie_pred_merged4_p3.json
REC=pilot/cache/prod/kg_records_merged5.json
KG=pilot/cache/prod/kg_merged5.json
KGS=pilot/cache/prod/kg_merged5_semio.json
Q=pilot/cache/prod/nlice_queries_v2.json

echo "== 1. assemble graph (records -> KG, with documentation-depth provenance)"
.venv/bin/python pilot/scripts/v157_build_kg.py --records "$REC" --preds "$PRED" --out "$KG"

echo "== 2. semiology attribute enrichment"
.venv/bin/python pilot/scripts/semiology_build_edges.py --kg "$KG" --out "$KGS"

echo "== 3. seam invariants (known-failing §12 degree check is left visible on purpose)"
.venv/bin/python pilot/scripts/pipeline_invariants.py --preds "$PRED" --records "$REC" --kg "$KGS" || true

echo "== 4. six-attribute leave-one-out ablation (compare to nlice_ablation_pre_p2.json)"
.venv/bin/python pilot/scripts/nlice_ablation.py --kg "$KGS" --queries "$Q" --perm 200 --boot 2000

echo "== 5. diagnosis, rare + common (single algorithm, coverage smoothing on)"
COMMON="--parents pilot/cache/prod/phenotype_parents_rich.json --hier 1 --lsa 1 --lsa_w 0.4 \
 --sapbert 0 --tau 0.78 --hpo_bridge pilot/cache/prod/hpo_bridge.json --hpo_anc 0 --hpo_w 0.5 \
 --bio_bridge pilot/cache/prod/biochem_bridge.json --bio_w 1.0 --cov_mu 0.5"
# shellcheck disable=SC2086
.venv/bin/python pilot/scripts/onlykg_eval_unified2.py --kg "$KGS" $COMMON --benchmarks rarebench
# shellcheck disable=SC2086
DDX_COMPOSE_PAIN=1 .venv/bin/python pilot/scripts/onlykg_eval_unified2.py --kg "$KGS" $COMMON \
  --benchmarks ddxplus,symcat --attr 1 \
  --attr_use character,severity,onset,timing,aggravating,relieving --attr_w 10 --attr_mu 0.0
