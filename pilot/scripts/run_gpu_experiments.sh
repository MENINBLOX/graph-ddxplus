#!/bin/bash
# GPU 실험 순차 실행: medgemma → v16_diff
set -e

echo "=== [1/2] medgemma-1.5-4b re-ranking ==="
/home/max/Graph-DDXPlus/.venv/bin/python -u pilot/scripts/kg_diagnose_v15_medgemma.py 2>&1

echo ""
echo "=== [2/2] v16 감별 프롬프트 (고유 증상) ==="
/home/max/Graph-DDXPlus/.venv/bin/python -u pilot/scripts/kg_diagnose_v16_diff.py 2>&1
