#!/bin/bash
cd /home/max/thesis
source .venv/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1
echo "[orchestrator] $(date) waiting for PMC crawl to finish..."
until grep -q "^done:" pilot/cache/prod/pmc_fetch.log 2>/dev/null; do sleep 60; done
NE=$(python3 -c "import glob,json; print(sum(1 for f in glob.glob('pilot/cache/prod/sources_pmc/*.json') if json.load(open(f)).get('text')))")
echo "[orchestrator] crawl done. non-empty PMC blobs: $NE"
echo "[1/4] IE on PMC sources $(date)"
python3 pilot/scripts/genereviews_ie.py --src-dir pilot/cache/prod/sources_pmc --out pilot/cache/prod/ie_pred_pmc.json --bs 16 --max-new 2048 || { echo "IE FAILED"; exit 1; }
echo "[2/4] merge $(date)"
python3 pilot/scripts/merge_ie_preds.py --base pilot/cache/prod/ie_pred_merged2.json --add pilot/cache/prod/ie_pred_pmc.json --out pilot/cache/prod/ie_pred_merged3.json || { echo "MERGE FAILED"; exit 1; }
echo "[3/4] normalize $(date)"
python3 pilot/scripts/v155_normalize_umls.py --pred pilot/cache/prod/ie_pred_merged3.json --out pilot/cache/prod/kg_records_merged3.json || { echo "NORM FAILED"; exit 1; }
echo "[4/4] build KG $(date)"
python3 pilot/scripts/v157_build_kg.py --records pilot/cache/prod/kg_records_merged3.json --out pilot/cache/prod/kg_merged3.json || { echo "BUILD FAILED"; exit 1; }
echo "[orchestrator] DONE $(date) -> kg_merged3.json"
