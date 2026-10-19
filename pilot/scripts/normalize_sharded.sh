#!/usr/bin/env bash
# Normalize the corpus as N independent shards in parallel, then merge the shard record files.
#
# scispaCy linking is single-process, so a full-corpus pass used one core of twenty-four and took
# an estimated eleven hours. The partition is crc32(cui) % N, which is stable across processes and
# restarts, so shards are disjoint, each is independently resumable, and a killed worker only costs
# its own last checkpoint.
#
#   N=8 nohup setsid bash pilot/scripts/normalize_sharded.sh > pilot/cache/prod/norm_sharded.log 2>&1 &
set -uo pipefail
P=pilot/cache/prod
N=${N:-8}
PRED=${PRED:-$P/ie_pred_merged29_promoted.json}
OUT=${OUT:-$P/kg_records_merged29.json}
mkdir -p "$P/shards"

for i in $(seq 0 $((N - 1))); do
  .venv-sci/bin/python pilot/scripts/v154_prod_normalize.py \
    --pred "$PRED" --out "$P/shards/rec_$i.json" \
    --shard "$i" --nshards "$N" --resume --checkpoint 200 \
    > "$P/shards/norm_$i.log" 2>&1 &
done
wait

# Merge shards. Disjoint by construction, so a key appearing twice means the partition broke and
# the run must not be trusted.
.venv/bin/python - "$OUT" "$P"/shards/rec_*.json <<'PY'
import json, sys
out_path, *shards = sys.argv[1:]
merged, dupes = {}, 0
for p in shards:
    part = json.load(open(p, encoding="utf-8"))
    for k, v in part.items():
        if k in merged:
            dupes += 1
        merged[k] = v
    print(f"  {p}: {len(part)}")
if dupes:
    sys.exit(f"ABORT: {dupes} diseases appeared in more than one shard — partition is broken")
json.dump(merged, open(out_path, "w"))
print(f"merged {len(merged)} normalized diseases -> {out_path}")
PY
echo NORM_SHARDED_DONE >> "$P/bench_chain.log"
