#!/bin/bash
# 벤치마크 평가 순차 실행
# SymCat KG 완료 후 평가 → RareBench KG 구축 → 평가
set -e

PY=/home/max/Graph-DDXPlus/.venv/bin/python

# Wait for SymCat KG cache to exist
while [ ! -f /home/max/Graph-DDXPlus/pilot/results/kg_symcat_cache.json ]; do
    echo "[$(date +%H:%M:%S)] SymCat KG 대기..."
    sleep 60
done
echo "[$(date +%H:%M:%S)] SymCat KG 발견!"

echo ""
echo "=== SymCat 평가 ==="
$PY -u pilot/scripts/diagnose_symcat.py

echo ""
echo "=== RareBench KG 구축 ==="
$PY -u pilot/scripts/build_kg_rarebench.py

echo ""
echo "=== 모든 벤치마크 완료 ==="
