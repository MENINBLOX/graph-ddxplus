#!/bin/bash
PY=/home/max/Graph-DDXPlus/.venv/bin/python

# Wait for RareBench KG to finish
while [ ! -f /home/max/Graph-DDXPlus/pilot/results/kg_rarebench_cache.json ]; do
    sleep 60
done

# Wait for any vLLM to release GPU
while ps aux | grep -E "run_prepared_kg|kg_diagnose" | grep -v grep > /dev/null; do
    sleep 30
done

echo "[$(date +%H:%M:%S)] GPU free, running v22 CoT..."
$PY -u pilot/scripts/kg_diagnose_v22_cot.py
