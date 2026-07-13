#!/bin/bash
# v103 batch IE — 49 DDXPlus diseases, 3 GPU parallel
set -e
cd /home/max/Graph-DDXPlus
source .venv/bin/activate
mkdir -p pilot/data/cache/v103_per_disease

# Get all 49 disease CUIs and names
python -c "
import json
icd = json.load(open('data/ddxplus/disease_icd10_cui_mapping.json'))
diseases = [(info['cui'], dn) for dn, info in icd.items() if info.get('cui')]
print(f'Total: {len(diseases)}')
# Save list
with open('pilot/data/cache/v103_disease_pool.txt', 'w') as f:
    for cui, dn in diseases:
        f.write(f'{cui}\t{dn}\n')
"

# Sharding for 3 GPUs
python -c "
diseases = [l.strip().split('\t') for l in open('pilot/data/cache/v103_disease_pool.txt')]
n = len(diseases)
per_shard = (n + 2) // 3
for i in range(3):
    s, e = i*per_shard, min(n, (i+1)*per_shard)
    with open(f'pilot/data/cache/v103_shard{i}.txt', 'w') as f:
        for cui, dn in diseases[s:e]:
            f.write(f'{cui}\t{dn}\n')
    print(f'  shard{i}: {e-s} diseases [{s}:{e}]')
"

mkdir -p logs

# Launch 3 GPU shards
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python pilot/scripts/v103_run_shard.py \
    --shard_file pilot/data/cache/v103_shard${i}.txt \
    --out_dir pilot/data/cache/v103_per_disease \
    > logs/v103_shard${i}.log 2>&1 &
  echo "Launched shard $i (PID $!)"
done

wait
echo "All shards complete"
ls pilot/data/cache/v103_per_disease/ | wc -l
