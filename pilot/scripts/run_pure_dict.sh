#!/usr/bin/env bash
# P0-2 (research design §9.32): the pure dictionary graph v3umlsnt_ns = S-dict (v3umls_ns) built from the same
# UMLS dictionary items, documents and flags, with an empty phrase table. In build_graph_v2.py the table is read
# in two places: linked_forms() for items that carry no CUI (a UMLS string naming several finding concepts with
# no unique preferred name), and the Concept.tui property. With the empty table such an item links only through
# the HPO name index, a dictionary; Concept.tui is left blank, which neither graph_dx.py nor the build
# fingerprint reads.
# Development: the nine sets with the frozen gemma 1-hop scoring at offtopic_w .5 (S-dict's frozen value) and
# the tune_v2 grid {0, .25, 1}, as run_graph_variants.sh. Confirmation set E (POST HOC) at offtopic_w .5 as
# run_confirm_E.sh. Results are kept while the build fingerprint and the scoring are unchanged.
#
#   [URI=bolt://localhost:7688] [OTS="0.5 0.0 0.25 1.0"] [CONFIRM=1] bash pilot/scripts/run_pure_dict.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
V=pilot/cache/v2
V3=pilot/cache/v3
O=$V3/p0/pure_dict
URI=${URI:-bolt://localhost:7688}
G=v3umlsnt_ns
mkdir -p $O
exec 9>"$V/devD/.neo4j_${URI##*:}.lock"
flock 9
read -r A M D U B < <($PY -c "import json;s=json.load(open('$V/devD/frozen_config_D.json'))['gemma_1hop'];print(s['alpha'],s['mu'],s['decay'],s['unified_w'],s['support_beta'])")
UMLS3="$V/dict_umls_v2.jsonl.gz,$V3/dict_umls_v3.jsonl.gz"
SRC3="$V/sources,$V3/sources_new"
echo '{}' > $O/empty_table.json

prev=$O/graph_$G.prev.json
rm -f "$prev"; [ -f $O/graph_$G.json ] && cp $O/graph_$G.json "$prev"
$PY pilot/scripts/build_graph_v2.py --split-coord --hpo-names --topic-genes --no-self --uri $URI --ie $UMLS3 \
  --table $O/empty_table.json --sources $SRC3 --max-year 0 --wipe --no-mentions --summary $O/graph_$G.json \
  > $O/build_$G.log 2>&1
stale=$($PY - $O/graph_$G.json "$prev" "$A $M $D $U $B" <<'EOF'
import json, os, sys
new = json.load(open(sys.argv[1]))
new["scoring"] = sys.argv[3]
old = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else None
json.dump(new, open(sys.argv[1], "w"), indent=1)
print(int(old is None or old["fingerprint"] != new["fingerprint"] or old.get("scoring") != new["scoring"]))
EOF
)
rm -f "$prev"
[ "$stale" = 1 ] && rm -f "${O:?}"/dev_${G}_ot*.json "${O:?}"/E_${G}_ot*.json

dev() {
  [ -f $O/dev_${G}_ot$1.json ] || $PY pilot/scripts/graph_dx.py --uri $URI --bench ddxplus-test,rarebench,pumch,phenopacket,symcat \
    --ddx_n 3000 --alpha $A --mu $M --decay $D --unified_w $U --support_beta $B --offtopic_w $1 \
    --out $O/dev_${G}_ot$1.json > $O/dev_${G}_ot$1.log 2>&1
}
confirm() {  # POST HOC: E was judged once (design E7); this run selects nothing
  [ -f $O/E_${G}_ot$1.json ] || CONFIRM_E_UNSEAL=1 $PY pilot/scripts/graph_dx.py --uri $URI --bench confirm_e \
    --alpha $A --mu $M --decay $D --unified_w $U --support_beta $B --offtopic_w $1 --boot 1000 \
    --out $O/E_${G}_ot$1.json > $O/E_${G}_ot$1.log 2>&1
}
pids=()
for W in ${OTS:-0.5 0.0 0.25 1.0}; do dev "$W" & pids+=($!); done
[ "${CONFIRM:-1}" = 1 ] && { confirm 0.5 & pids+=($!); }
for p in "${pids[@]}"; do wait "$p"; done
echo "done $G"
