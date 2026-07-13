#!/bin/bash
PY=/home/max/Graph-DDXPlus/.venv/bin/python

# Wait for prompts file
while [ ! -f /home/max/Graph-DDXPlus/pilot/results/kg_rarebench_prompts.json ]; do
    echo "[$(date +%H:%M:%S)] RareBench prompts 대기..."
    sleep 30
done
echo "[$(date +%H:%M:%S)] Prompts 발견!"

echo ""
echo "=== RareBench KG (vLLM) ==="
$PY -u pilot/scripts/run_prepared_kg.py rarebench

echo ""
echo "=== RareBench 평가 ==="
$PY -u pilot/scripts/diagnose_rarebench.py
