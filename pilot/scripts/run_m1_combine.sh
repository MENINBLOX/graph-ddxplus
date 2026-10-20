#!/usr/bin/env bash
# §9.33 M1 (development sets, post hoc): HPOA + S-mix combinations (m1_combine.py has the definitions).
#   1. S-mix ranked lists on the read-only S-mix instance (7689), in the background.
#   2. On the comparator instance (7688, under its flock): HPOA and the P0-4 HPOA-LOPO graph
#      (build_reference_graph.py; loaded edges must equal deletions.json) and their ranked lists.
#   3. union-build with no HPOA edge, ranked on RareBench + PUMCH: the S-mix written by union-build must reproduce the
#      stored S-mix ranks before any union graph is ranked.
#   4. S-mix ∪ HPOA ranked on the nine development sets and S-mix ∪ HPOA-LOPO on the publication-based sets, both
#      with the frozen S-mix configuration. 7688 is wiped at the end.
#   5. fuse (RRF, w chosen by D3) and stats.
# Steps whose output exists are skipped; every attempt is logged in attempts.log.
#   bash pilot/scripts/run_m1_combine.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
M1=pilot/scripts/m1_combine.py
O=pilot/cache/v3/framing/m1
URI=bolt://localhost:7688
SMIX_URI=bolt://localhost:7689
mkdir -p $O
echo "$(date -Iseconds) start $(git rev-parse HEAD)" >> $O/attempts.log

SPID=""
if [ ! -s $O/lists_S-mix.json.gz ]; then
  $PY $M1 lists --system S-mix --uri $SMIX_URI > $O/lists_S-mix.log 2>&1 &
  SPID=$!
fi

exec 9>"pilot/cache/v2/devD/.neo4j_${URI##*:}.lock"
flock 9
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
$PY $M1 lopo-file --out "$W/lopo.hpoa" > $O/lopo_file.log 2>&1
HPOA_FILE=data/external_kg/phenotype.hpoa
read -r SMIX_ARGS < <($PY $M1 args --system S-mix)

cypher() {
  $PY -c "
from neo4j import GraphDatabase
d=GraphDatabase.driver('$URI',auth=('neo4j','thesis-dev-only'))
with d.session() as s:
    print(s.run('''$1''').single()[0])"
}

edges_of() { $PY -c "import json;print(json.load(open('pilot/cache/v3/p0/hpoa_lopo/deletions.json'))['variants']['$1']['edges'])"; }

for S in HPOA HPOA-LOPO; do
  [ -s $O/lists_$S.json.gz ] && { echo "$S: lists exist, skipped"; continue; }
  if [ $S = HPOA ]; then f=$HPOA_FILE; want=$(edges_of base); else f=$W/lopo.hpoa; want=$(edges_of lopo); fi
  echo "$(date -Iseconds) $S build" >> $O/attempts.log
  $PY pilot/scripts/build_reference_graph.py --source hpoa --hpoa "$f" --uri $URI --summary $O/graph_$S.json \
    > $O/build_$S.log 2>&1
  got=$($PY -c "import json;print(json.load(open('$O/graph_$S.json'))['edges'])")
  loaded=$(cypher "MATCH ()-[e:HAS_FINDING]->() RETURN count(e)")
  [ "$want" = "$got" ] && [ "$want" = "$loaded" ] \
    || { echo "$S: edges expected $want, summary $got, loaded $loaded" | tee -a $O/attempts.log; exit 1; }
  $PY $M1 lists --system $S --uri $URI > $O/lists_$S.log 2>&1 \
    || { echo "$S: lists failed (see lists_$S.log)" | tee -a $O/attempts.log; exit 1; }
  echo "$(date -Iseconds) $S lists saved (edges $loaded)" | tee -a $O/attempts.log
done

union() {  # $1 name, $2 phenotype.hpoa or none, $3 graph_dx.py benches, $4 output
  [ -s "$4" ] && { echo "$1: result exists, skipped"; return; }
  echo "$(date -Iseconds) $1 build" >> $O/attempts.log
  $PY $M1 union-build --hpoa "$2" --uri $URI --smix-uri $SMIX_URI --summary $O/graph_$1.json > $O/build_$1.log 2>&1 \
    || { echo "$1: union-build failed (see build_$1.log)" | tee -a $O/attempts.log; exit 1; }
  # shellcheck disable=SC2086
  $PY pilot/scripts/graph_dx.py --uri $URI --bench "$3" --ddx_n 3000 $SMIX_ARGS --out "$4.part" > $O/rank_$1.log 2>&1
  mv "$4.part" "$4"
  echo "$(date -Iseconds) $1 ranked" | tee -a $O/attempts.log
}

if [ ! -s $O/mirror_check.log ] || ! grep -q "^ok" $O/mirror_check.log; then
  union smix_only none rarebench,pumch $O/mirror_smix_only.json
  $PY $M1 check-mirror --run $O/mirror_smix_only.json > $O/mirror_check.log 2>&1 \
    || { echo "mirror: S-mix written by union-build differs from the stored ranks" | tee -a $O/attempts.log; exit 1; }
  echo "ok" >> $O/mirror_check.log
fi
union union $HPOA_FILE ddxplus-test,rarebench,pumch,phenopacket,symcat $O/union.json
union union_lopo "$W/lopo.hpoa" rarebench,phenopacket $O/union_lopo.json

for q in "MATCH ()-[r]->() RETURN r|DELETE r" "MATCH (n) RETURN n|DETACH DELETE n"; do
  cypher "CALL apoc.periodic.iterate('${q%%|*}', '${q##*|}', {batchSize: 2000}) YIELD failedBatches RETURN failedBatches" > /dev/null
done
echo "$(date -Iseconds) wiped $URI ($(cypher "MATCH (n) RETURN count(n)") nodes left)" | tee -a $O/attempts.log
flock -u 9

if [ -n "$SPID" ]; then
  wait "$SPID" || { echo "S-mix lists failed (see lists_S-mix.log)" | tee -a $O/attempts.log; exit 1; }
fi
$PY $M1 fuse > $O/fuse.log 2>&1
$PY $M1 stats > $O/stats.log 2>&1
echo "$(date -Iseconds) done" | tee -a $O/attempts.log
