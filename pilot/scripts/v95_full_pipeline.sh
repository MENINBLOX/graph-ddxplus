#!/bin/bash
# v95 full pipeline: merge shards -> UMLS map -> build KG -> precheck -> evaluate
# Run after all 3 IE shards complete.

set -e
cd /home/max/Graph-DDXPlus

PY=".venv/bin/python"
LOG_DIR=logs
mkdir -p $LOG_DIR

echo "=== Step 3: Merge 3 shards into v95_full_rare_ie.jsonl ==="
cat pilot/data/cache/v95_full_shard0.jsonl \
    pilot/data/cache/v95_full_shard1.jsonl \
    pilot/data/cache/v95_full_shard2.jsonl > pilot/data/cache/v95_full_rare_ie.jsonl
wc -l pilot/data/cache/v95_full_rare_ie.jsonl

echo ""
echo "=== Step 4: UMLS direct mapping -> v95_full_cui_edges.jsonl ==="
$PY pilot/scripts/v95_map_rare.py \
    --ie_path pilot/data/cache/v95_full_rare_ie.jsonl \
    --out pilot/data/cache/v95_full_cui_edges.jsonl \
    --pool pilot/data/cache/v95_remaining_pool.json 2>&1 | tee $LOG_DIR/v95_full_map.log

echo ""
echo "=== Step 5: Build KG: v93 base + (v95_cui_edges + v95_full_cui_edges) at scale=5 ==="
# Combine both new IE outputs (v95 + v95_full)
cat pilot/data/cache/v95_cui_edges.jsonl pilot/data/cache/v95_full_cui_edges.jsonl \
    > pilot/data/cache/v95_combined_cui_edges.jsonl
wc -l pilot/data/cache/v95_combined_cui_edges.jsonl

# Build on top of v85 (v93 = v85 + v92) so v95_full = v85 + v92 + v95_combined
# Use scale=5 (matches v93 recipe + project rule "scale=5 when augmenting v93/v95")
$PY pilot/scripts/v80_integrate_kg.py \
    --base_kg pilot/data/onlykg_graph_v93_s3.pkl \
    --llm_edges pilot/data/cache/v95_combined_cui_edges.jsonl \
    --out pilot/data/onlykg_graph_v95_full_s3.pkl \
    --scale 5 2>&1 | tee $LOG_DIR/v95_full_build.log

echo ""
echo "=== Step 6: Coverage precheck ==="
$PY pilot/scripts/precheck_benchmark_vocab.py \
    --graph pilot/data/onlykg_graph_v95_full_s3.pkl 2>&1 | tee $LOG_DIR/v95_full_precheck.log

echo ""
echo "=== Step 8: KG topology inspect ==="
$PY pilot/scripts/kg_topology_inspect.py \
    --graph pilot/data/onlykg_graph_v95_full_s3.pkl 2>&1 | tee $LOG_DIR/v95_full_topology.log

echo ""
echo "=== Step 7a: DDXPlus 5K eval (tau=1.5 lam=0.4) ==="
$PY pilot/scripts/eval_ddxplus_full_metrics.py \
    --graph pilot/data/onlykg_graph_v95_full_s3.pkl \
    --n 5000 --tau 1.5 --lam 0.4 2>&1 | tee $LOG_DIR/v95_full_ddx5k.log

echo ""
echo "=== Step 7b: PhenoBrain eval ==="
$PY pilot/scripts/eval_v93_phenobrain.py \
    --graph pilot/data/onlykg_graph_v95_full_s3.pkl 2>&1 | tee $LOG_DIR/v95_full_pheno.log

echo ""
echo "=== Step 7c: MIMIC-RD eval ==="
$PY pilot/scripts/eval_v93_mimic_rd.py \
    --graph pilot/data/onlykg_graph_v95_full_s3.pkl 2>&1 | tee $LOG_DIR/v95_full_mimicrd.log

echo ""
echo "=== Step 7d: DDXPlus 30K eval ==="
$PY pilot/scripts/eval_ddxplus_full_metrics.py \
    --graph pilot/data/onlykg_graph_v95_full_s3.pkl \
    --n 30000 --tau 1.5 --lam 0.4 2>&1 | tee $LOG_DIR/v95_full_ddx30k.log

echo ""
echo "=== ALL DONE ==="
