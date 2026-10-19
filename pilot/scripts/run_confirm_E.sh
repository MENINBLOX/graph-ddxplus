#!/usr/bin/env bash
# Design E confirmation run (docs/research_design.md E3-E5). Run ONCE, after the freeze commit.
#   bash pilot/scripts/run_confirm_E.sh
# Each graph system is rebuilt on the comparator instance (7688) and evaluated only if its fingerprint and loaded
# edge count match frozen_config_E.json. After an interruption, rerun: systems whose result file exists are not
# run again (E5); every attempt is logged in attempts.log.
set -euo pipefail
cd "$(dirname "$0")/../.."
F=pilot/cache/v2/devD/frozen_config_E.json
O=pilot/cache/v2/confirmE
R=pilot/cache/v3/retrieval
URI=bolt://localhost:7688
PY=.venv/bin/python
git ls-files --error-unmatch "$F" > /dev/null || { echo "frozen config not committed"; exit 1; }
git diff --quiet HEAD -- pilot/scripts "$F" data/confirm_e || { echo "uncommitted changes in scripts/config/manifest"; exit 1; }
[ -e "$O/DONE" ] && { echo "confirmation already run ($O/DONE)"; exit 1; }
mkdir -p $O
echo "$(date -Iseconds) start $(git rev-parse HEAD)" >> $O/attempts.log
export CONFIRM_E_UNSEAL=1
get() { $PY -c "import json;f=json.load(open('$F'));print($1)"; }
read -r A M D U B < <(get "' '.join(str(f['scoring'][k]) for k in ('alpha','mu','decay','unified_w','support_beta'))")

# 1. graph systems; S-mix last so the fusion step finds it loaded
for S in S-D S-llm S-dict S-union S-mix; do
  out=$O/$S.json
  [ -s "$out" ] && { echo "$S: result exists, skipped"; continue; }
  V=$(get "f['graph_systems']['$S']['variant']")
  OT=$(get "f['graph_systems']['$S']['offtopic_w']")
  BUILD_ONLY=1 URI=$URI bash pilot/scripts/run_graph_variants.sh "$V" >> $O/build.log 2>&1
  want=$(get "f['graph_systems']['$S']['fingerprint']")
  got=$($PY -c "import json;print(json.load(open('pilot/cache/v2/devD/variants/graph_$V.json'))['fingerprint'])")
  [ "$want" = "$got" ] || { echo "$S: fingerprint $got != frozen $want" | tee -a $O/attempts.log; exit 1; }
  edges=$($PY -c "
from neo4j import GraphDatabase
d=GraphDatabase.driver('$URI',auth=('neo4j','thesis-dev-only'))
with d.session() as s: print(s.run('MATCH ()-[e:HAS_FINDING]->() RETURN count(e) AS n').single()['n'])")
  [ "$edges" = "$(get "f['graph_systems']['$S']['edges']")" ] || { echo "$S: loaded edges $edges differ" | tee -a $O/attempts.log; exit 1; }
  $PY pilot/scripts/graph_dx.py --uri $URI --bench confirm_e --alpha $A --mu $M --decay $D --unified_w $U \
    --support_beta $B --offtopic_w $OT --boot 1000 --out $out.part > $O/$S.log 2>&1
  mv $out.part $out
  echo "$(date -Iseconds) $S done" >> $O/attempts.log
done

# 2. S-bm25: the frozen index, checked file by file
$PY - "$F" "$R" <<'EOF'
import hashlib, json, sys
from pathlib import Path
f, r = json.load(open(sys.argv[1])), Path(sys.argv[2])
for name, want in f["S-bm25"]["index_sha256"].items():
    h = hashlib.sha256(open(r / name, "rb").read()).hexdigest()
    if h != want:
        sys.exit(f"S-bm25 index file {name} differs from the freeze")
EOF
get "json.dumps({'selected': f['S-bm25']['selected']})" > $O/bm25_cfg.json
if [ ! -s $O/S-bm25.json ]; then
  $PY pilot/scripts/retrieval_baseline.py test --bench confirm_e --cfg $O/bm25_cfg.json --index $R --out $O/S-bm25.json.part > $O/S-bm25.log 2>&1
  mv $O/S-bm25.json.part $O/S-bm25.json
fi

# 3. S-comb: S-mix (still loaded on 7688) fused with S-bm25 at the frozen weight
OTM=$(get "f['graph_systems']['S-mix']['offtopic_w']")
get "json.dumps({'selected': {**f['scoring'], 'offtopic_w': $OTM}})" > $O/smix_cfg.json
get "json.dumps({'selected': {'w': f['S-comb']['w']}})" > $O/fuse_tune.json
if [ ! -s $O/fuse_test.json ]; then
  $PY pilot/scripts/retrieval_baseline.py fuse_test --bench confirm_e --graph_uri $URI --graph_cfg $O/smix_cfg.json \
    --cfg $O/bm25_cfg.json --index $R --out $O/fuse_test.json > $O/S-comb.log 2>&1
fi
cp $O/fuse_test.json $O/S-comb.json

# 4. targets E-T1..E-T5 (paired_stats refuses a missing or unpaired judged set)
SETS=ConfirmE-DDXPlus,ConfirmE-SymCat
cmp() {  # A B label [--macro]
  for k in 10 1; do
    $PY pilot/scripts/paired_stats.py $O/$1.json $O/$2.json --k $k --require $SETS ${4:+--macro $SETS} \
      --label "$3" --out $O/paired_${1}_vs_${2}_k$k.json
  done
}
{
  cmp S-D S-mix "E-T1 S-mix vs S-D"
  cmp S-llm S-mix "E-T2 S-mix vs S-llm" macro
  cmp S-dict S-mix "E-T2 S-mix vs S-dict" macro
  cmp S-union S-mix "E-T2 S-mix vs S-union" macro
  cmp S-bm25 S-mix "E-T3 S-mix vs S-bm25"
  cmp S-mix S-comb "E-T5 S-comb vs S-mix" macro
  cmp S-bm25 S-comb "E-T5 S-comb vs S-bm25" macro
} > $O/paired.txt
date -Iseconds > $O/DONE
echo "$(date -Iseconds) done" >> $O/attempts.log
echo "confirmation E done"
