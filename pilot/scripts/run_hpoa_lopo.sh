#!/usr/bin/env bash
# P0-4 (docs/research_design.md §9.32): HPOA leave-one-publication-out on the development sets. hpoa_lopo.py prepare
# writes the filtered phenotype.hpoa files into a work directory that is removed on exit; each is loaded into the
# comparator instance with build_reference_graph.py --source hpoa and the nine development sets are ranked with the
# frozen HPOA 1-hop configuration (frozen_config_D.json comparators.hpoa.onehop, §9.22). A graph is ranked only when
# its build summary and the HAS_FINDING edges loaded in Neo4j equal the edge count prepare computed. Results are
# kept across runs (the filtered files are deterministic; deletions.json records their sha256). The 10-fold graphs
# lopo_f<k> rank only fold k's Phenopacket-store cases (hpoa_lopo.py rank-fold); lopo_check first reloads the lopo
# graph and requires rank-fold to reproduce graph_dx.py's stored ranks for fold 0. The instance is wiped at the
# end. hpoa_lopo.py stats then compares every graph with S-mix.
#   bash pilot/scripts/run_hpoa_lopo.sh [base lopo rand_s0 ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
O=pilot/cache/v3/p0/hpoa_lopo
URI=bolt://localhost:7688
mkdir -p $O
exec 9>"pilot/cache/v2/devD/.neo4j_${URI##*:}.lock"
flock 9
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
$PY pilot/scripts/hpoa_lopo.py prepare --hpoa-dir "$W" > $O/prepare.log 2>&1
read -r A M D U B G < <($PY -c "import json;s=json.load(open('pilot/cache/v2/devD/frozen_config_D.json'))['comparators']['hpoa']['onehop'];print(*(s[k] for k in ('alpha','mu','decay','unified_w','support_beta','offtopic_w')))")

cypher() {
  $PY -c "
from neo4j import GraphDatabase
d=GraphDatabase.driver('$URI',auth=('neo4j','thesis-dev-only'))
with d.session() as s:
    print(s.run('''$1''').single()[0])"
}

FOLDS=$(for k in $(seq 0 9); do printf 'lopo_f%s ' "$k"; done)
for V in ${*:-base lopo rand_s0 rand_dz_s0 lopo_strip nopmid rand_nopmid rand_s1 rand_s2 lopo_check $FOLDS}; do
  case $V in
    lopo_check) G_=lopo; out=$O/fold_check.json ;;
    lopo_f*) G_=$V; out=$O/fold_${V#lopo_f}.json ;;
    *) G_=$V; out=$O/dev_$V.json ;;
  esac
  [ -s "$out" ] && { echo "$V: result exists, skipped"; continue; }
  want=$($PY -c "import json;print(json.load(open('$O/deletions.json'))['variants']['$G_']['edges'])")
  echo "$(date -Iseconds) $V build" >> $O/attempts.log
  $PY pilot/scripts/build_reference_graph.py --source hpoa --hpoa "$W/$G_.hpoa" --uri $URI \
    --summary $O/graph_$V.json > $O/build_$V.log 2>&1
  got=$($PY -c "import json;print(json.load(open('$O/graph_$V.json'))['edges'])")
  loaded=$(cypher "MATCH ()-[e:HAS_FINDING]->() RETURN count(e)")
  [ "$want" = "$got" ] && [ "$want" = "$loaded" ] \
    || { echo "$V: edges prepare $want, summary $got, loaded $loaded" | tee -a $O/attempts.log; exit 1; }
  case $V in
    lopo_check) $PY pilot/scripts/hpoa_lopo.py rank-fold --fold 0 --check $O/dev_lopo.json --uri $URI > $O/$V.log 2>&1 ;;
    lopo_f*) $PY pilot/scripts/hpoa_lopo.py rank-fold --fold "${V#lopo_f}" --uri $URI > $O/$V.log 2>&1 ;;
    *) $PY pilot/scripts/graph_dx.py --uri $URI --bench ddxplus-test,rarebench,pumch,phenopacket,symcat --ddx_n 3000 \
         --alpha "$A" --mu "$M" --decay "$D" --unified_w "$U" --support_beta "$B" --offtopic_w "$G" \
         --out "$out.part" > $O/dev_$V.log 2>&1
       mv "$out.part" "$out" ;;
  esac
  echo "$(date -Iseconds) $V ranked (edges $loaded)" | tee -a $O/attempts.log
done
for q in "MATCH ()-[r]->() RETURN r|DELETE r" "MATCH (n) RETURN n|DETACH DELETE n"; do
  cypher "CALL apoc.periodic.iterate('${q%%|*}', '${q##*|}', {batchSize: 2000}) YIELD failedBatches RETURN failedBatches" > /dev/null
done
echo "$(date -Iseconds) wiped $URI ($(cypher "MATCH (n) RETURN count(n)") nodes left)" | tee -a $O/attempts.log
