#!/usr/bin/env bash
# P0-3 (docs/research_design.md §9.32): re-rank confirmation set E, POST HOC, for the four E-T2 graph systems so that
# the per-patient strict ranks are kept (graph_dx.py ranks_strict). Each system is rebuilt on the comparator
# instance and ranked only if its fingerprint, loaded HAS_FINDING edges and Disease nodes equal frozen_config_E.json;
# scoring is the frozen one with the system's frozen offtopic_w. p0_strict_stats.py then checks that ranks_lineage
# equals the confirmation run's file exactly.
#   bash pilot/scripts/run_p0_strict.sh [S-llm S-dict S-union S-mix]
set -euo pipefail
cd "$(dirname "$0")/../.."
F=pilot/cache/v2/devD/frozen_config_E.json
O=pilot/cache/v3/p0/strict
URI=bolt://localhost:7688
PY=.venv/bin/python
mkdir -p $O
export CONFIRM_E_UNSEAL=1
get() { $PY -c "import json;f=json.load(open('$F'));print($1)"; }
read -r A M D U B < <(get "' '.join(str(f['scoring'][k]) for k in ('alpha','mu','decay','unified_w','support_beta'))")

loaded() {  # HAS_FINDING edges and Disease nodes on $URI
  $PY -c "
from neo4j import GraphDatabase
d=GraphDatabase.driver('$URI',auth=('neo4j','thesis-dev-only'))
with d.session() as s:
    e=s.run('MATCH ()-[e:HAS_FINDING]->() RETURN count(e) AS n').single()['n']
    n=s.run('MATCH (d:Disease) RETURN count(d) AS n').single()['n']
print(e, n)"
}

SYSTEMS=${*:-S-llm S-dict S-union S-mix}
for S in $SYSTEMS; do
  out=$O/$S.json
  [ -s "$out" ] && { echo "$S: result exists, skipped"; continue; }
  V=$(get "f['graph_systems']['$S']['variant']")
  OT=$(get "f['graph_systems']['$S']['offtopic_w']")
  t0=$O/.t_$S
  touch "$t0"
  echo "$(date -Iseconds) $S build $V" >> $O/attempts.log
  BUILD_ONLY=1 URI=$URI bash pilot/scripts/run_graph_variants.sh "$V" >> $O/build.log 2>&1
  sum=pilot/cache/v2/devD/variants/graph_$V.json
  [ "$sum" -nt "$t0" ] || { echo "$S: build summary not rewritten" | tee -a $O/attempts.log; exit 1; }
  want=$(get "f['graph_systems']['$S']['fingerprint']")
  got=$($PY -c "import json;print(json.load(open('$sum'))['fingerprint'])")
  [ "$want" = "$got" ] || { echo "$S: fingerprint $got != frozen $want" | tee -a $O/attempts.log; exit 1; }
  read -r E N < <(loaded)
  [ "$E $N" = "$(get "f['graph_systems']['$S']['edges'], f['graph_systems']['$S']['diseases']")" ] \
    || { echo "$S: loaded edges/diseases $E $N differ from the freeze" | tee -a $O/attempts.log; exit 1; }
  echo "$(date -Iseconds) $S verified fingerprint $got edges $E diseases $N" >> $O/attempts.log
  $PY pilot/scripts/graph_dx.py --uri $URI --bench confirm_e --alpha $A --mu $M --decay $D --unified_w $U \
    --support_beta $B --offtopic_w "$OT" --boot 1000 --out "$out.part" > $O/$S.log 2>&1
  mv "$out.part" "$out"
  rm -f "$t0"
  echo "$(date -Iseconds) $S ranked" >> $O/attempts.log
done
echo "p0 strict ranking done"
