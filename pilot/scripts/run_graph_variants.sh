#!/usr/bin/env bash
# Literature-graph variants evaluated the same way: built into a comparator instance without Mention
# nodes, then the nine development sets with the frozen gemma 1-hop configuration and offtopic_w as the
# only re-selected scoring key (design D3 rule, tune_v2 grid), and the dated phenopacket cases
# (temporal_eval.py) at offtopic_w 0 and 1. Variants ending in _y2022 keep only documents published by
# 2022, use gene-disease links only for diseases HPOA annotated by then, and are read on the cases
# published from 2023 on (time-split leakage check). Suffixes before _y2022: _ns adds --no-self, _nq
# --no-qualifiers, _kn --keep-named-generic.
# Results are kept while the rebuilt graph's fingerprint (build_graph_v2.py summary) and the scoring
# configuration are unchanged. One runner at a time per instance (lock file per port).
# v3 variants extract the v2 and the v3 documents the same way: dict (HPO names), umls (UMLS finding
# strings), llm (gemma IE, phrases linked by phrase2cui_v3.json, which run_normalize_v3.sh must have
# extended over the whole ie_v3.jsonl), union (llm + dict), unionu (llm + umls), llmd (llm items the phrase table
# leaves unlinked, re-linked by the umls dictionary inside the phrase: llm_dict_relink.py; llmda keeps the
# unlinked item as well). *full read the PMC
# bodies uncut (sources_full) instead of the 12k clinical sections.
#
#   [URI=bolt://localhost:7689] bash pilot/scripts/run_graph_variants.sh base_nm base_split_tg v2dict v2union
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
V=pilot/cache/v2
V3=pilot/cache/v3
O=$V/devD/variants
URI=${URI:-bolt://localhost:7688}
mkdir -p $O
exec 9>"$V/devD/.neo4j_${URI##*:}.lock"
flock 9
read -r A M D U B < <($PY -c "import json;s=json.load(open('$V/devD/frozen_config_D.json'))['gemma_1hop'];print(s['alpha'],s['mu'],s['decay'],s['unified_w'],s['support_beta'])")
ALL="--split-coord --hpo-names --topic-genes"
LLM=$V/ie_gemma_full.jsonl
LLM3="$LLM,$V3/ie_v3.jsonl"
DICT3="$V/dict_v2.jsonl.gz,$V3/dict_v3.jsonl.gz"
UMLS3="$V/dict_umls_v2.jsonl.gz,$V3/dict_umls_v3.jsonl.gz"
SRC3="$V/sources,$V3/sources_new"
SRC3F="$V3/sources_full,$V/sources"
for G in "$@"; do
  name=${G%_y2022}; YEAR=0; SNAP=2026-09-01; CUT=""; OPT=""
  [ "$name" != "$G" ] && { YEAR=2022; SNAP=2022-12-31; CUT="--gene-cut data/external_kg/hpoa_history/v2022-12-15.hpoa"; }
  while :; do
    case $name in
      *_ns) OPT="$OPT --no-self"; name=${name%_ns} ;;
      *_nq) OPT="$OPT --no-qualifiers"; name=${name%_nq} ;;
      *_kn) OPT="$OPT --keep-named-generic"; name=${name%_kn} ;;
      *) break ;;
    esac
  done
  SRC=$V/sources; TABLE=$V/phrase2cui_gemma.json; FLAGS=$ALL
  case $name in
    base_nm)       IE=$LLM; FLAGS="" ;;
    base_split)    IE=$LLM; FLAGS="--split-coord --hpo-names" ;;
    base_split_tg) IE=$LLM ;;
    v2dict)        IE=$V/dict_v2.jsonl.gz ;;
    v2union)       IE=$LLM,$V/dict_v2.jsonl.gz ;;
    v2umls)        IE=$V/dict_umls_v2.jsonl.gz ;;
    v2unionu)      IE=$LLM,$V/dict_umls_v2.jsonl.gz ;;
    v3dict)        IE=$DICT3; SRC=$SRC3 ;;
    v3dictfull)    IE=$V/dict_v2.jsonl.gz,$V3/dict_v3full.jsonl.gz; SRC=$SRC3F ;;
    v3umls)        IE=$UMLS3; SRC=$SRC3 ;;
    v3umlsfull)    IE=$V/dict_umls_v2.jsonl.gz,$V3/dict_umls_v3full.jsonl.gz; SRC=$SRC3F ;;
    v3llm)         IE=$LLM3; SRC=$SRC3; TABLE=$V3/phrase2cui_v3.json ;;
    v3union)       IE=$LLM3,$DICT3; SRC=$SRC3; TABLE=$V3/phrase2cui_v3.json ;;
    v3unionu)      IE=$LLM3,$UMLS3; SRC=$SRC3; TABLE=$V3/phrase2cui_v3.json ;;
    v3llmd)        IE=$V3/llmd_v2.jsonl.gz,$V3/llmd_v3.jsonl.gz; SRC=$SRC3; TABLE=$V3/phrase2cui_v3.json ;;
    v3llmda)       IE=$V3/llmda_v2.jsonl.gz,$V3/llmda_v3.jsonl.gz; SRC=$SRC3; TABLE=$V3/phrase2cui_v3.json ;;
    *) echo "unknown variant $G"; exit 1 ;;
  esac
  [ -s "$TABLE" ] || { echo "missing phrase table $TABLE"; exit 1; }
  if [ "$TABLE" = "$V3/phrase2cui_v3.json" ] && [ "$(cat $V3/phrase2cui_v3.covers 2>/dev/null)" != "$(wc -l < $V3/ie_v3.jsonl)" ]; then
    echo "$G: phrase2cui_v3.json does not cover ie_v3.jsonl; run run_normalize_v3.sh"; exit 1
  fi
  prev=$O/graph_$G.prev.json
  rm -f "$prev"; [ -f $O/graph_$G.json ] && cp $O/graph_$G.json "$prev"
  # shellcheck disable=SC2086
  $PY pilot/scripts/build_graph_v2.py $FLAGS $CUT $OPT --uri $URI --ie $IE --table $TABLE \
    --sources $SRC --max-year $YEAR --wipe --no-mentions --summary $O/graph_$G.json > $O/build_$G.log 2>&1
  # Old summaries carry no fingerprint or scoring key; for them the counts decide.
  stale=$($PY - $O/graph_$G.json "$prev" "$A $M $D $U $B" <<'EOF'
import json, os, sys
new = json.load(open(sys.argv[1]))
new["scoring"] = sys.argv[3]
old = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else None
json.dump(new, open(sys.argv[1], "w"), indent=1)
if old is None:
    print(1)
elif "fingerprint" in old:
    print(int(old["fingerprint"] != new["fingerprint"] or old.get("scoring") != new["scoring"]))
else:
    print(int(any(old.get(k) != new.get(k) for k in ("diseases", "diseases_with_findings", "concepts", "edges", "mentions", "docs"))))
EOF
)
  rm -f "$prev"
  # BUILD_ONLY=1 (confirmation runs): load the graph and report whether it matches the last summary; touch no result
  if [ "${BUILD_ONLY:-0}" = 1 ]; then echo "built $G stale=$stale"; continue; fi
  if [ "$stale" = 1 ]; then
    rm -f "${O:?}"/dev_${G}_ot*.json "${O:?}"/temporal_${G}_ot*.json
  fi
  if [ $YEAR = 0 ]; then
    for G2 in 0.0 0.25 0.5 1.0; do
      [ -f $O/dev_${G}_ot$G2.json ] || $PY pilot/scripts/graph_dx.py --uri $URI --bench ddxplus-test,rarebench,pumch,phenopacket,symcat --ddx_n 3000 \
        --alpha $A --mu $M --decay $D --unified_w $U --support_beta $B --offtopic_w $G2 --out $O/dev_${G}_ot$G2.json > $O/dev_${G}_ot$G2.log 2>&1
    done
  fi
  for G2 in 0.0 1.0; do
    [ -f $O/temporal_${G}_ot$G2.json ] || $PY pilot/scripts/temporal_eval.py --uri $URI --snapshot $SNAP --cfg gemma_1hop --override "{\"offtopic_w\": $G2}" \
      --tag lit_${G}_ot$G2 --out $O/temporal_${G}_ot$G2.json > $O/temporal_${G}_ot$G2.log 2>&1
  done
  echo "done $G"
done
