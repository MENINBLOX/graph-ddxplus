#!/usr/bin/env bash
# End-to-end medkg pipeline. Run after download finishes.
set -e
cd /home/max/Graph-DDXPlus

echo "Step 1: Section extraction"
.venv/bin/python pilot/scripts/medkg_extract_sections.py 2>&1 | tail -3

echo ""
echo "Step 2: Orphanet structured edges"
.venv/bin/python pilot/scripts/medkg_parse_orphanet.py 2>&1 | tail -3

echo ""
echo "Step 3: LLM IE on sections (vLLM)"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python pilot/scripts/medkg_ie_multi_source.py 2>&1 | tail -5

echo ""
echo "Step 4: Normalize phenotypes (HPO mapping via Orphanet index)"
.venv/bin/python pilot/scripts/medkg_normalize_phenotypes.py 2>&1 | tail -3

echo ""
echo "Step 5: Multi-source merge"
.venv/bin/python pilot/scripts/medkg_merge_sources.py 2>&1 | tail -10

echo ""
echo "Step 6: DDXPlus eval (multi-source KG)"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python pilot/scripts/medkg_eval_ddxplus.py 2>&1 | tail -10

echo "Pipeline complete"
