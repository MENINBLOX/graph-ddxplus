#!/usr/bin/env bash
# Build and evaluate the open-schema attribute KG. Run from the repo root.
#
# What changed versus the slot-based build: the attribute inventory is no longer six slots chosen
# from Bates. The IE extracts qualifiers with no vocabulary in the prompt (attr_open_ie.py), and the
# controlled vocabulary is applied downstream -- HPO Clinical modifier, HP:0012823 (hpo_modifier.py).
# The schema is therefore MEASURED: across 32,186 diseases the families the literature actually
# supplies are Clinical course (13,746), Severity (12,898) and Position (11,084), while the ones the
# old schema was built around are near-absent (Aggravated by 343, Pain characteristic 253,
# Ameliorated by 11).
#
# Chain: ie_pred_merged7 (slot ∪ open) -> records -> KG -> ablation -> benchmarks
set -euo pipefail

PRED=pilot/cache/prod/ie_pred_merged7.json
REC=pilot/cache/prod/kg_records_merged7.json
KG=pilot/cache/prod/kg_merged7.json
KGS=pilot/cache/prod/kg_merged7_semio.json
Q=pilot/cache/prod/nlice_queries_v2.json

echo "== 1. assemble graph (carries open qualifiers + documentation depth)"
.venv/bin/python pilot/scripts/v157_build_kg.py --records "$REC" --preds "$PRED" --out "$KG"

echo "== 2. semiology attribute enrichment"
.venv/bin/python pilot/scripts/semiology_build_edges.py --kg "$KG" --out "$KGS"

echo "== 3. seam invariants (the §12 degree check is expected to fail; it is the open bottleneck)"
.venv/bin/python pilot/scripts/pipeline_invariants.py --preds "$PRED" --records "$REC" --kg "$KGS" || true

echo "== 4. leave-one-out ablation over HPO modifier families (+ the UMLS anatomy axis)"
.venv/bin/python pilot/scripts/hpo_family_ablation.py --kg "$KGS" --queries "$Q" --perm 30

echo "== 5. diagnosis: rare then common (single algorithm, coverage smoothing on)"
COMMON="--parents pilot/cache/prod/phenotype_parents_rich.json --hier 1 --lsa 1 --lsa_w 0.4 \
 --sapbert 0 --tau 0.78 --hpo_bridge pilot/cache/prod/hpo_bridge.json --hpo_anc 0 --hpo_w 0.5 \
 --bio_bridge pilot/cache/prod/biochem_bridge.json --bio_w 1.0 --cov_mu 0.5"
# shellcheck disable=SC2086
.venv/bin/python pilot/scripts/onlykg_eval_unified2.py --kg "$KGS" $COMMON --benchmarks rarebench
# shellcheck disable=SC2086
DDX_COMPOSE_PAIN=1 .venv/bin/python pilot/scripts/onlykg_eval_unified2.py --kg "$KGS" $COMMON \
  --benchmarks ddxplus,symcat --attr 1 --attr_use character,severity,onset \
  --attr_w 10 --attr_mu 0.0
