#!/bin/bash
PY=/home/max/Graph-DDXPlus/.venv/bin/python

# Wait for v25 to finish
while ps aux | grep -E "kg_diagnose_v25" | grep -v grep > /dev/null; do
    sleep 30
done
echo "[$(date +%H:%M:%S)] v25 done. Verifying KG..."
$PY -u pilot/scripts/kg_verify_pairs.py
