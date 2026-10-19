#!/usr/bin/env bash
# Resumable corpus-v2 collection: re-running continues where it stopped (finished diseases are skipped).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env 2>/dev/null; set +a
for i in 1 2 3 4 5; do
  .venv/bin/python pilot/scripts/fetch_v2.py --universe pilot/cache/v2/disease_universe.json \
    --out-dir pilot/cache/v2/sources --exclude-pmids pilot/cache/v2/exclude_pmids_phenopacket.json \
    --reviews 6 --cases 4 --per-article 12000 --workers 6 && break
  sleep 60
done
echo FETCH_V2_DONE
