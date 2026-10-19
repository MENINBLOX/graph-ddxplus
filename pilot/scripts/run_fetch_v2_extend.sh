#!/usr/bin/env bash
# Corpus v2 depth extension (design D17): every disease re-queried with reviews=15, cases=10;
# only articles not yet stored are added (and written to sources_delta for incremental IE).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env 2>/dev/null; set +a
for i in 1 2 3 4 5; do
  .venv/bin/python pilot/scripts/fetch_v2.py --universe pilot/cache/v2/disease_universe.json \
    --out-dir pilot/cache/v2/sources --extend-delta pilot/cache/v2/sources_delta \
    --exclude-pmids pilot/cache/v2/exclude_pmids_phenopacket.json \
    --reviews 15 --cases 10 --per-article 12000 --workers 6 && break
  sleep 60
done
echo FETCH_EXTEND_DONE
