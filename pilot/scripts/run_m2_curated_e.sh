#!/usr/bin/env bash
# §9.33 M2 [post hoc]: the design-D curated comparators (HPOA, PrimeKG; 1-hop at frozen_config_D.json
# comparators.<src>.onehop, the setting of the D confirmation reference run run_confirm_D_ref.sh) ranked ONCE on
# confirmation set E. Each source is rebuilt on the comparator instance (7688) with build_reference_graph.py; E is
# ranked only when the build and the loaded HAS_FINDING edges equal the D reference build and m2_curated_e.py check
# finds the stored development ranks and the per-disease edge counts of the development graph reproduced. A source
# whose E result exists is not ranked again. The instance is wiped at the end; m2_curated_e.py stats then compares.
#   bash pilot/scripts/run_m2_curated_e.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
F=pilot/cache/v2/devD/frozen_config_D.json
O=pilot/cache/v3/framing/m2
URI=bolt://localhost:7688
mkdir -p $O
exec 9>"pilot/cache/v2/devD/.neo4j_${URI##*:}.lock"
flock 9
echo "$(date -Iseconds) start $(git rev-parse HEAD)" >> $O/attempts.log

cypher() {
  $PY -c "
from neo4j import GraphDatabase
d=GraphDatabase.driver('$URI',auth=('neo4j','thesis-dev-only'))
with d.session() as s:
    print(s.run('''$1''').single()[0])"
}

for src in hpoa primekg; do
  out=$O/E_$src.json
  [ -s "$out" ] && { echo "$src: E result exists, skipped"; continue; }
  want=$($PY -c "import re;print(re.search(r'edges (\d+)',open('pilot/cache/v2/confirmD/ref_${src}_build.log').read())[1])")
  $PY pilot/scripts/build_reference_graph.py --source $src --uri $URI --summary $O/graph_$src.json > $O/build_$src.log 2>&1
  got=$($PY -c "import json;print(json.load(open('$O/graph_$src.json'))['edges'])")
  loaded=$(cypher "MATCH ()-[e:HAS_FINDING]->() RETURN count(e)")
  [ "$want" = "$got" ] && [ "$want" = "$loaded" ] \
    || { echo "$src: edges D build $want, summary $got, loaded $loaded" | tee -a $O/attempts.log; exit 1; }
  read -r A M D U B G < <($PY -c "import json;s=json.load(open('$F'))['comparators']['$src']['onehop'];print(*(s[k] for k in ('alpha','mu','decay','unified_w','support_beta','offtopic_w')))")
  args=(--uri $URI --alpha "$A" --mu "$M" --decay "$D" --unified_w "$U" --support_beta "$B" --offtopic_w "$G")
  if [ ! -s $O/dev_$src.json ]; then
    $PY pilot/scripts/graph_dx.py "${args[@]}" --bench ddxplus-test,rarebench,pumch,phenopacket,symcat --ddx_n 3000 \
      --out $O/dev_$src.json.part > $O/dev_$src.log 2>&1
    mv $O/dev_$src.json.part $O/dev_$src.json
  fi
  $PY pilot/scripts/m2_curated_e.py check --src $src --dev $O/dev_$src.json --uri $URI > $O/check_$src.log 2>&1 \
    || { echo "$src: check failed (see check_$src.log)" | tee -a $O/attempts.log; exit 1; }
  CONFIRM_E_UNSEAL=1 $PY pilot/scripts/graph_dx.py "${args[@]}" --bench confirm_e --boot 1000 \
    --out $out.part > $O/E_$src.log 2>&1
  mv $out.part $out
  echo "$(date -Iseconds) $src ranked on E (edges $loaded)" | tee -a $O/attempts.log
done
for q in "MATCH ()-[r]->() RETURN r|DELETE r" "MATCH (n) RETURN n|DETACH DELETE n"; do
  cypher "CALL apoc.periodic.iterate('${q%%|*}', '${q##*|}', {batchSize: 2000}) YIELD failedBatches RETURN failedBatches" > /dev/null
done
CONFIRM_E_UNSEAL=1 $PY pilot/scripts/m2_curated_e.py stats > $O/stats.log 2>&1
echo "$(date -Iseconds) done" >> $O/attempts.log
echo "M2 done"
