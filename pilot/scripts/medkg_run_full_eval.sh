#!/usr/bin/env bash
# Full evaluation: rebuild medkg KG with all downloaded data, then eval on 3 benchmarks.
# Run after download completes.
set -e
cd /home/max/Graph-DDXPlus

echo "=== STEP 1: Section extraction ==="
.venv/bin/python pilot/scripts/medkg_extract_sections.py 2>&1 | tail -3

echo ""
echo "=== STEP 2: Multi-source LLM IE (vLLM batch) ==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python pilot/scripts/medkg_ie_multi_source.py 2>&1 | tail -5

echo ""
echo "=== STEP 3: Normalize + noise filter ==="
.venv/bin/python pilot/scripts/medkg_normalize_phenotypes.py 2>&1 | tail -5

echo ""
echo "=== STEP 4: Multi-source merge ==="
.venv/bin/python pilot/scripts/medkg_merge_sources.py 2>&1 | tail -5

echo ""
echo "=== STEP 5: KG summary stats ==="
.venv/bin/python pilot/scripts/medkg_summary_stats.py 2>&1 | tail -25

echo ""
echo "=== STEP 6: DDXPlus eval prep (disease_features.json) ==="
.venv/bin/python pilot/scripts/medkg_eval_ddxplus.py 2>&1 | tail -5

echo ""
echo "=== STEP 7: DDXPlus v110 medkg full eval (5K patients) ==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python pilot/scripts/kg_diagnose_v110_medkg.py 2>&1 | tail -10

echo ""
echo "=== STEP 8: SymCat v110 medkg eval ==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python pilot/scripts/diagnose_symcat_v110_medkg.py 50 2>&1 | tail -10

echo ""
echo "=== STEP 9: RareBench v110 medkg eval ==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python pilot/scripts/diagnose_rarebench_v110_medkg.py 2>&1 | tail -10

echo ""
echo "=== STEP 10: Compare v87 vs v110 ==="
.venv/bin/python pilot/scripts/medkg_compare_results.py 2>&1

echo ""
echo "Pipeline complete."
